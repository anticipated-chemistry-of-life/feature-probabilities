"""``generate-groundtruth``: MassSpecGym-vs-SIRIUS ground-truth CLI.

Issue #20's happy-path slice (given the shared TOML config (#11), fetches
and chunks a pinned MassSpecGym revision (#13, #16), runs every chunk
through the cache-aware SIRIUS pipeline (#19) with
``source_kind='ground_truth_massspecgym'``/``extract_id=None``, sets each
resulting feature's ``true_inchikey`` from the chunk's known MassSpecGym
structure) plus issue #21's remaining batch-control surface:

- ``--force`` sets the SIRIUS ``JobSubmission.recompute`` field and passes
  ``force=True`` through to `run_cache.get_or_create_run`, bypassing the
  cache lookup for every chunk.
- ``--instrument-type {Orbitrap,QTOF,all}`` filters the parsed MassSpecGym
  spectra to one instrument type before chunking (default ``all``); the
  excluded instrument type's chunks are never written or sent to SIRIUS.
- A single chunk's failure (any exception raised while running or
  persisting it) is caught, recorded into the run's summary, and rolled
  back in isolation -- `generate_groundtruth` now `commit`s after every
  chunk (success or failure) instead of only `flush`ing and leaving one
  final `commit` to `main`, so an earlier chunk's successful,
  already-committed work survives a later chunk's failure, and a rerun
  only reprocesses the failed/missing chunks (`run_cache`'s cache lookup
  finds the successful ones). `main` exits non-zero and prints every
  failed chunk's name and error in the final summary whenever
  `GroundtruthSummary.failures` is non-empty.

A feature's true structure is looked up by ``external_feature_id`` against
a MassSpecGym-``identifier``-keyed map built once from the fetched TSV --
SIRIUS's pre-picked import path carries each MGF spectrum's ``FEATURE_ID``
(set from that same ``identifier`` by ``massspecgym._row_to_spectrum``)
through unchanged onto ``AlignedFeature.external_feature_id``.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import PySirius
from PySirius import JobSubmission
from sqlalchemy.orm import Session

from feature_probabilities.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
)
from feature_probabilities.massspecgym import (
    MassSpecGymFetchError,
    MassSpecGymParseError,
    fetch_massspecgym_tsv,
    parse_massspecgym_spectra,
    spectrum_instrument_type,
    write_sirius_chunks,
)
from feature_probabilities.run_cache import (
    RunCacheError,
    SiriusRunRequest,
    get_or_create_run,
)
from feature_probabilities.schema import (
    SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
    create_database,
)
from feature_probabilities.sirius import (
    Sirius,
    SiriusCredentialsError,
    SiriusStartupError,
    SiriusVersionMismatchError,
    SiriusVersionUnavailableError,
    require_matching_sirius_version,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matchms import Spectrum

    from feature_probabilities.sirius import SiriusInterface

#: MassSpecGym currently exposes only positive-mode adducts (see
#: `massspecgym._add_required_metadata_for_sirius`), so every ground-truth
#: run this CLI creates is unconditionally recorded as positive-mode.
_IONIZATION_MODE = "positive"


#: Every error this CLI's `main` treats as a clean, actionable failure
#: rather than a raw traceback.
_CLICK_EXCEPTION_ERRORS = (
    ConfigError,
    MassSpecGymFetchError,
    MassSpecGymParseError,
    RunCacheError,
    SiriusCredentialsError,
    SiriusStartupError,
    SiriusVersionMismatchError,
    SiriusVersionUnavailableError,
)


@dataclass(frozen=True, slots=True)
class ChunkFailure:
    """One chunk whose SIRIUS run or persistence raised, per issue #21.

    `error` is `str(exc)` rather than the exception object itself, so
    `GroundtruthSummary` (and anything printing it) stays a plain,
    repr-friendly data value.
    """

    chunk_path: Path
    instrument_type: str
    error: str


@dataclass(frozen=True, slots=True)
class GroundtruthSummary:
    """A basic success/failure summary of what one `generate_groundtruth` call did."""

    chunks_processed: int
    cache_hits: int
    features_seen: int
    annotations_seen: int
    failures: tuple[ChunkFailure, ...] = ()


def _true_inchikey_by_identifier(spectra: Sequence[Spectrum]) -> dict[str, str]:
    """MassSpecGym `identifier` -> known-true `inchikey`, across every spectrum.

    `identifier` is globally unique across the whole MassSpecGym TSV and is
    carried unchanged onto each MGF spectrum's `feature_id`/`FEATURE_ID`
    field (`massspecgym._row_to_spectrum`), which SIRIUS's pre-picked import
    path reports back as `AlignedFeature.external_feature_id`.
    """
    return {
        str(spectrum.get("feature_id")): str(spectrum.get("inchikey"))
        for spectrum in spectra
    }


def generate_groundtruth(
    session: Session,
    sirius: SiriusInterface,
    config: Config,
    *,
    work_dir: Path,
    force: bool = False,
    instrument_type: str = "all",
) -> GroundtruthSummary:
    """Run the full pipeline against an already-open `session`.

    Fetches `config.massspecgym_revision`, chunks it by instrument type
    (filtered to `instrument_type` unless it's `"all"`), and runs every
    chunk through `run_cache.get_or_create_run` with
    `source_kind='ground_truth_massspecgym'`/`extract_id=None`/
    `force=force`, setting each resulting feature's `true_inchikey`.
    `force=True` also sets the SIRIUS `JobSubmission.recompute` field.

    `commit`s `session` after every chunk -- on success, so an earlier
    chunk's work survives a later chunk's failure; on failure, `rollback`s
    first, so a partially-persisted failed chunk leaves no rows behind,
    then records a `ChunkFailure` and moves on to the next chunk rather
    than aborting the whole run. Only errors raised *before* any chunk is
    processed (SIRIUS version mismatch, MassSpecGym fetch/parse failure)
    propagate out of this function.

    Raises:
        SiriusVersionMismatchError: the installed SIRIUS version doesn't
            match `config.required_sirius_version`.
        MassSpecGymFetchError: MassSpecGym couldn't be fetched.
        MassSpecGymParseError: MassSpecGym's TSV is malformed.
    """
    require_matching_sirius_version(sirius, config.required_sirius_version)

    tsv_path = fetch_massspecgym_tsv(config.massspecgym_revision)
    spectra = parse_massspecgym_spectra(tsv_path)
    if instrument_type != "all":
        spectra = [
            spectrum
            for spectrum in spectra
            if spectrum_instrument_type(spectrum) == instrument_type
        ]
    true_inchikey_by_identifier = _true_inchikey_by_identifier(spectra)
    chunk_paths_by_instrument_type = write_sirius_chunks(spectra, work_dir / "chunks")

    analysis_params = JobSubmission(
        **{**config.sirius.analysis_params, "recompute": force}
    )

    chunks_processed = 0
    cache_hits = 0
    features_seen = 0
    annotations_seen = 0
    failures: list[ChunkFailure] = []
    for chunk_instrument_type, chunk_paths in chunk_paths_by_instrument_type.items():
        for chunk_path in chunk_paths:
            request = SiriusRunRequest(
                input_file=chunk_path,
                project_path=work_dir / "projects" / chunk_path.stem,
                source_kind=SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
                extract_id=None,
                import_params=None,
                analysis_params=analysis_params,
                sirius_version=config.required_sirius_version,
                pysirius_client_version=PySirius.__version__,
                ionization_mode=_IONIZATION_MODE,
                instrument_type=chunk_instrument_type,
            )
            try:
                result = get_or_create_run(session, sirius, request, force=force)
                for feature in result.features:
                    true_inchikey = true_inchikey_by_identifier.get(
                        feature.external_feature_id
                    )
                    if true_inchikey is not None:
                        feature.true_inchikey = true_inchikey
                session.commit()
            except Exception as exc:  # noqa: BLE001 -- one bad chunk must not abort the batch
                session.rollback()
                failures.append(
                    ChunkFailure(
                        chunk_path=chunk_path,
                        instrument_type=chunk_instrument_type,
                        error=str(exc),
                    )
                )
                continue

            chunks_processed += 1
            cache_hits += 1 if result.from_cache else 0
            features_seen += len(result.features)
            annotations_seen += len(result.annotations)

    return GroundtruthSummary(
        chunks_processed=chunks_processed,
        cache_hits=cache_hits,
        features_seen=features_seen,
        annotations_seen=annotations_seen,
        failures=tuple(failures),
    )


def _format_summary(summary: GroundtruthSummary) -> str:
    newly_run = summary.chunks_processed - summary.cache_hits
    lines = [
        (
            f"generate-groundtruth: processed {summary.chunks_processed} chunk(s) "
            f"({summary.cache_hits} cache hit(s), {newly_run} newly run), covering "
            f"{summary.features_seen} feature(s) and {summary.annotations_seen} "
            "annotation(s)."
        )
    ]
    if summary.failures:
        lines.append(f"{len(summary.failures)} chunk(s) FAILED:")
        lines.extend(
            f"  - {failure.chunk_path.name} ({failure.instrument_type}): "
            f"{failure.error}"
            for failure in summary.failures
        )
    return "\n".join(lines)


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help=f"Path to the TOML config file (default: {DEFAULT_CONFIG_PATH}).",
)
@click.option(
    "--db",
    "db_path",
    type=str,
    default=None,
    help="Override the config's db_path.",
)
@click.option(
    "--massspecgym-revision",
    "massspecgym_revision",
    type=str,
    default=None,
    help="Override the config's pinned MassSpecGym HuggingFace revision.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Bypass the sirius_runs cache and set JobSubmission.recompute so "
        "every chunk is recomputed by SIRIUS."
    ),
)
@click.option(
    "--instrument-type",
    "instrument_type",
    type=click.Choice(["Orbitrap", "QTOF", "all"]),
    default="all",
    help="Only process chunks for this instrument type (default: all).",
)
def main(
    config_path: Path | None,
    db_path: str | None,
    massspecgym_revision: str | None,
    force: bool,
    instrument_type: str,
) -> None:
    """Fetch MassSpecGym, run it through cache-aware SIRIUS, populate ground truth."""
    try:
        config = load_config(
            config_path if config_path is not None else DEFAULT_CONFIG_PATH,
            overrides={
                "db_path": db_path,
                "massspecgym_revision": massspecgym_revision,
            },
        )
        engine = create_database(config.db_path)
        try:
            sirius = Sirius(headless=True)
            with (
                tempfile.TemporaryDirectory(prefix="generate-groundtruth-") as tmp,
                Session(engine) as session,
            ):
                summary = generate_groundtruth(
                    session,
                    sirius,
                    config,
                    work_dir=Path(tmp),
                    force=force,
                    instrument_type=instrument_type,
                )
        finally:
            engine.dispose()
    except _CLICK_EXCEPTION_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_format_summary(summary))
    if summary.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

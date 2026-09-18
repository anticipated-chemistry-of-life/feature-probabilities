"""``generate-groundtruth``: the happy-path MassSpecGym-vs-SIRIUS ground-truth CLI.

Issue #20's first end-to-end, runnable slice of this executable: given the
shared TOML config (#11), fetches and chunks a pinned MassSpecGym revision
(#13, #16), runs every chunk through the cache-aware SIRIUS pipeline (#19)
with ``source_kind='ground_truth_massspecgym'``/``extract_id=None``, sets
each resulting feature's ``true_inchikey`` from the chunk's known
MassSpecGym structure, and prints a basic success summary.

A feature's true structure is looked up by ``external_feature_id`` against
a MassSpecGym-``identifier``-keyed map built once from the fetched TSV --
SIRIUS's pre-picked import path carries each MGF spectrum's ``FEATURE_ID``
(set from that same ``identifier`` by ``massspecgym._row_to_spectrum``)
through unchanged onto ``AlignedFeature.external_feature_id``.

``generate_groundtruth`` only ``flush``es, matching ``run_cache``'s and
``metadata``'s convention that a caller controls the transaction; ``main``,
its one caller, ``commit``s once the whole run has succeeded.

``--force``, ``--instrument-type`` filtering, and per-chunk batch-failure
handling (log-and-continue, non-zero exit with a failure summary) are
deliberately deferred to the next ticket -- ``main`` aborts the whole run
on the first error instead, per issue #20's explicitly narrowed scope.
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
    SiriusVersionUnavailableError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matchms import Spectrum

    from feature_probabilities.sirius import SiriusInterface

#: MassSpecGym currently exposes only positive-mode adducts (see
#: `massspecgym._add_required_metadata_for_sirius`), so every ground-truth
#: run this CLI creates is unconditionally recorded as positive-mode.
_IONIZATION_MODE = "positive"


class SiriusVersionMismatchError(Exception):
    """Raised when the installed SIRIUS version doesn't match the config's."""


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
class GroundtruthSummary:
    """A basic success summary of what one `generate_groundtruth` call did."""

    chunks_processed: int
    cache_hits: int
    features_seen: int
    annotations_seen: int


def _require_matching_sirius_version(
    sirius: SiriusInterface, required_version: str
) -> None:
    """Fail fast if the running SIRIUS instance isn't `required_version`.

    Raises:
        SiriusVersionMismatchError: the installed and required versions differ.
    """
    installed_version = sirius.get_version()
    if installed_version != required_version:
        raise SiriusVersionMismatchError(
            f"Installed SIRIUS version {installed_version!r} does not match "
            f"this config's required_sirius_version {required_version!r}. "
            "Install the required SIRIUS version or update the config."
        )


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
) -> GroundtruthSummary:
    """Run the full happy-path pipeline against an already-open `session`.

    Fetches `config.massspecgym_revision`, chunks it by instrument type, and
    runs every chunk through `run_cache.get_or_create_run` with
    `source_kind='ground_truth_massspecgym'`/`extract_id=None`, setting each
    resulting feature's `true_inchikey`. Only `flush`es `session`, matching
    `run_cache`'s and `metadata`'s caller-controls-the-transaction convention
    -- the caller commits.

    Raises:
        SiriusVersionMismatchError: the installed SIRIUS version doesn't
            match `config.required_sirius_version`.
        MassSpecGymFetchError: MassSpecGym couldn't be fetched.
        MassSpecGymParseError: MassSpecGym's TSV is malformed.
        RunCacheError: a chunk's SIRIUS results couldn't be persisted.
    """
    _require_matching_sirius_version(sirius, config.required_sirius_version)

    tsv_path = fetch_massspecgym_tsv(config.massspecgym_revision)
    spectra = parse_massspecgym_spectra(tsv_path)
    true_inchikey_by_identifier = _true_inchikey_by_identifier(spectra)
    chunk_paths_by_instrument_type = write_sirius_chunks(spectra, work_dir / "chunks")

    # `--force` (JobSubmission.recompute=True) is deferred to the next
    # ticket, so every ground-truth run is a non-recomputing job regardless
    # of what the config's [sirius.analysis_params] table happens to set.
    analysis_params = JobSubmission(
        **{**config.sirius.analysis_params, "recompute": False}
    )

    chunks_processed = 0
    cache_hits = 0
    features_seen = 0
    annotations_seen = 0
    for instrument_type, chunk_paths in chunk_paths_by_instrument_type.items():
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
                instrument_type=instrument_type,
            )
            result = get_or_create_run(session, sirius, request)
            for feature in result.features:
                true_inchikey = true_inchikey_by_identifier.get(
                    feature.external_feature_id
                )
                if true_inchikey is not None:
                    feature.true_inchikey = true_inchikey
            session.flush()

            chunks_processed += 1
            cache_hits += 1 if result.from_cache else 0
            features_seen += len(result.features)
            annotations_seen += len(result.annotations)

    return GroundtruthSummary(
        chunks_processed=chunks_processed,
        cache_hits=cache_hits,
        features_seen=features_seen,
        annotations_seen=annotations_seen,
    )


def _format_summary(summary: GroundtruthSummary) -> str:
    newly_run = summary.chunks_processed - summary.cache_hits
    return (
        f"generate-groundtruth: processed {summary.chunks_processed} chunk(s) "
        f"({summary.cache_hits} cache hit(s), {newly_run} newly run), covering "
        f"{summary.features_seen} feature(s) and {summary.annotations_seen} "
        "annotation(s)."
    )


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
def main(
    config_path: Path | None,
    db_path: str | None,
    massspecgym_revision: str | None,
) -> None:
    """Fetch MassSpecGym, run it through cache-aware SIRIUS, populate ground truth."""
    try:
        config = load_config(
            config_path if config_path is not None else DEFAULT_CONFIG_PATH,
            overrides={"db_path": db_path, "massspecgym_revision": massspecgym_revision},
        )
        engine = create_database(config.db_path)
        try:
            sirius = Sirius(headless=True)
            with (
                tempfile.TemporaryDirectory(prefix="generate-groundtruth-") as tmp,
                Session(engine) as session,
            ):
                summary = generate_groundtruth(
                    session, sirius, config, work_dir=Path(tmp)
                )
                session.commit()
        finally:
            engine.dispose()
    except _CLICK_EXCEPTION_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_format_summary(summary))


if __name__ == "__main__":
    main()

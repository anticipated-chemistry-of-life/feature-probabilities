"""``generate-groundtruth``: MSnLib-vs-SIRIUS ground truth CLI.

Given the shared TOML config, fetches MSnLib, the Ground truth set
(GLOSSARY.md), at the config's ``ground_truth_revision``
(`msnlib.fetch_msnlib_parquet`), attaches each spectrum's measured MS1 scan
from MSnLib's raw mzML archives (`msnlib_ms1`; ADR 0003), chunks it by
(Ionization mode, Instrument type), and runs every chunk through the
cache-aware SIRIUS pipeline with
``source_kind='ground_truth_msnlib'``/``extract_id=None``, setting each
resulting feature's ``true_inchikey`` from MSnLib's known structure.

- Each chunk's SIRIUS run records its Ionization mode and Instrument type,
  and is analysed under that Instrument type's SIRIUS profile
  (`instrument.analysis_params_for_instrument_type`).
- ``--force`` sets the SIRIUS ``JobSubmission.recompute`` field and passes
  ``force=True`` through to `run_cache.get_or_create_run`, bypassing the
  cache lookup for every chunk.
- ``--ionization-mode {positive,negative,all}`` keeps only that Ionization
  mode's spectra before chunking (default ``all``); the other mode's chunks
  are never written or sent to SIRIUS.
- A single chunk's failure (any exception raised while running or
  persisting it) is caught, recorded into the run's summary, and rolled
  back in isolation -- `generate_groundtruth` `commit`s after every chunk
  (success or failure), so an earlier chunk's successful, already-committed
  work survives a later chunk's failure, and a rerun only reprocesses the
  failed/missing chunks (`run_cache`'s cache lookup finds the successful
  ones). `main` exits non-zero and prints every failed chunk's name and
  error in the final summary whenever `GroundtruthSummary.failures` is
  non-empty.
- SIRIUS is shut down after every chunk (success or failure) and the next
  chunk that needs it starts a fresh instance (`Sirius.shutdown`). SIRIUS
  6.5.4 retains ~0.9 GB of heap per chunk that only a process exit frees,
  so one long-lived instance grows by hundreds of GB over the full dataset.
- A chunk whose SIRIUS job makes no progress for
  `sirius.JOB_STALL_TIMEOUT_SECONDS` (SIRIUS 6.5.4's job-pool deadlock)
  has its SIRIUS killed and is recorded as a failed chunk; the run moves on
  with a fresh instance instead of waiting forever.
- ``--smoke-test`` deletes and recreates the config's ``[smoke_test]``
  database, then processes only the first ``spectra_per_ionization_mode``
  spectra of each Ionization mode (after filtering, in parquet order), so
  the whole pipeline can be exercised in minutes. It rejects ``--db`` and
  ``--ionization-mode``.
- Each feature keeps only its ``config.sirius.top_k`` best-ranked structure
  candidates (GLOSSARY.md's Top-k; docs/adr/0001-top-k-structure-candidates.md).

A feature's true structure is looked up by ``external_feature_id`` against
an MSnLib-``identifier``-keyed map: SIRIUS's pre-picked import path carries
each MGF spectrum's ``FEATURE_ID`` (that same ``identifier``) through
unchanged onto ``AlignedFeature.external_feature_id``.
"""

from __future__ import annotations

import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import PySirius
from sqlalchemy.orm import Session

from feature_probabilities.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    for_smoke_test,
    load_config,
)
from feature_probabilities.instrument import analysis_params_for_instrument_type
from feature_probabilities.kde import IONIZATION_MODES
from feature_probabilities.msnlib import (
    FilterCounts,
    MsnlibFetchError,
    MsnlibParseError,
    fetch_msnlib_parquet,
    read_ground_truth_set,
    write_sirius_chunks,
)
from feature_probabilities.msnlib_ms1 import (
    attach_measured_ms1,
    fetch_msnlib_mzml_archives,
)
from feature_probabilities.run_cache import (
    RunCacheError,
    SiriusRunRequest,
    get_or_create_run,
)
from feature_probabilities.schema import (
    SOURCE_KIND_GROUND_TRUTH_MSNLIB,
    create_database,
)
from feature_probabilities.sirius import (
    Sirius,
    SiriusCredentialsError,
    SiriusShutdownError,
    SiriusStartupError,
    SiriusVersionMismatchError,
    SiriusVersionUnavailableError,
    require_matching_sirius_version,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from feature_probabilities.msnlib import GroundTruthSpectrum
    from feature_probabilities.sirius import SiriusInterface


#: Every error this CLI's `main` treats as a clean, actionable failure
#: rather than a raw traceback.
_CLICK_EXCEPTION_ERRORS = (
    ConfigError,
    MsnlibFetchError,
    MsnlibParseError,
    RunCacheError,
    SiriusCredentialsError,
    SiriusShutdownError,
    SiriusStartupError,
    SiriusVersionMismatchError,
    SiriusVersionUnavailableError,
)


@dataclass(frozen=True, slots=True)
class ChunkFailure:
    """One chunk whose SIRIUS run or persistence raised.

    `error` is `str(exc)` rather than the exception object itself, so
    `GroundtruthSummary` (and anything printing it) stays a plain,
    repr-friendly data value.
    """

    chunk_path: Path
    error: str


@dataclass(frozen=True, slots=True)
class GroundtruthSummary:
    """A basic success/failure summary of what one `generate_groundtruth` call did."""

    filter_counts: FilterCounts
    chunks_processed: int
    cache_hits: int
    features_seen: int
    annotations_seen: int
    #: Spectra sent to SIRIUS with MS2 only, since no measured MS1 scan
    #: preceding their MS2 scans holds the precursor.
    spectra_without_ms1: int
    failures: tuple[ChunkFailure, ...] = ()


def _first_per_ionization_mode(
    spectra: Sequence[GroundTruthSpectrum], limit: int
) -> list[GroundTruthSpectrum]:
    """The first `limit` spectra of each Ionization mode, in input order."""
    taken: Counter[str] = Counter()
    first: list[GroundTruthSpectrum] = []
    for spectrum in spectra:
        if taken[spectrum.ionization_mode] < limit:
            taken[spectrum.ionization_mode] += 1
            first.append(spectrum)
    return first


def generate_groundtruth(
    session: Session,
    sirius: SiriusInterface,
    config: Config,
    *,
    work_dir: Path,
    force: bool = False,
    ionization_mode: str = "all",
    spectra_per_ionization_mode: int | None = None,
) -> GroundtruthSummary:
    """Run the full pipeline against an already-open `session`.

    Fetches `config.ground_truth_revision` of MSnLib, attaches each kept
    spectrum's measured MS1 scan, chunks it by (Ionization mode, Instrument
    type) -- only `ionization_mode`'s spectra unless it's `"all"`, and only
    the first `spectra_per_ionization_mode` of each Ionization mode, in
    parquet order, when set -- and runs every chunk through
    `run_cache.get_or_create_run` with
    `source_kind='ground_truth_msnlib'`/`extract_id=None`/`force=force`,
    setting each resulting feature's `true_inchikey`. `force=True` also sets
    the SIRIUS `JobSubmission.recompute` field.

    `commit`s `session` after every chunk -- on success, so an earlier
    chunk's work survives a later chunk's failure; on failure, `rollback`s
    first, so a partially-persisted failed chunk leaves no rows behind,
    then records a `ChunkFailure` and moves on to the next chunk rather
    than aborting the whole run. Either way `sirius` is shut down after the
    chunk, freeing the heap SIRIUS never releases on its own. Only errors
    raised *before* any chunk is processed (SIRIUS version mismatch, MSnLib
    fetch/parse failure, MS1 extraction failure) or by a shutdown propagate
    out of this function.

    Raises:
        SiriusVersionMismatchError: the installed SIRIUS version doesn't
            match `config.required_sirius_version`.
        MsnlibFetchError: MSnLib's parquet or mzML archives couldn't be
            fetched, or a spectrum's raw file is in none of the archives.
        MsnlibParseError: MSnLib's parquet lacks a required column, a kept
            spectrum is malformed or names an unknown instrument, or its raw
            file lacks one of its MS2 scans.
        SiriusShutdownError: SIRIUS didn't exit after a chunk.
    """
    require_matching_sirius_version(sirius, config.required_sirius_version)

    ground_truth_set = read_ground_truth_set(
        fetch_msnlib_parquet(config.ground_truth_revision)
    )
    spectra = [
        spectrum
        for spectrum in ground_truth_set.spectra
        if ionization_mode in ("all", spectrum.ionization_mode)
    ]
    if spectra_per_ionization_mode is not None:
        spectra = _first_per_ionization_mode(spectra, spectra_per_ionization_mode)
    measured = attach_measured_ms1(spectra, fetch_msnlib_mzml_archives())
    true_inchikey_by_identifier = {
        spectrum.identifier: spectrum.true_inchikey for spectrum in measured.spectra
    }
    chunk_groups = write_sirius_chunks(measured.spectra, work_dir / "chunks")

    chunks_processed = 0
    cache_hits = 0
    features_seen = 0
    annotations_seen = 0
    failures: list[ChunkFailure] = []
    for group in chunk_groups:
        analysis_params = analysis_params_for_instrument_type(
            config.sirius.analysis_params, group.instrument_type, recompute=force
        )
        for chunk_path in group.paths:
            request = SiriusRunRequest(
                input_file=chunk_path,
                project_path=work_dir / "projects" / chunk_path.stem,
                source_kind=SOURCE_KIND_GROUND_TRUTH_MSNLIB,
                extract_id=None,
                import_params=None,
                analysis_params=analysis_params,
                sirius_version=config.required_sirius_version,
                pysirius_client_version=PySirius.__version__,
                ionization_mode=group.ionization_mode,
                instrument_type=group.instrument_type,
                top_k=config.sirius.top_k,
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
                failures.append(ChunkFailure(chunk_path=chunk_path, error=str(exc)))
                continue
            finally:
                sirius.shutdown()

            chunks_processed += 1
            cache_hits += 1 if result.from_cache else 0
            features_seen += len(result.features)
            annotations_seen += len(result.annotations)

    return GroundtruthSummary(
        filter_counts=ground_truth_set.filter_counts,
        chunks_processed=chunks_processed,
        cache_hits=cache_hits,
        features_seen=features_seen,
        annotations_seen=annotations_seen,
        spectra_without_ms1=measured.without_ms1,
        failures=tuple(failures),
    )


def _format_summary(summary: GroundtruthSummary) -> str:
    counts = summary.filter_counts
    newly_run = summary.chunks_processed - summary.cache_hits
    lines = [
        (
            f"Ground truth set: {counts.kept} spectrum(s) kept; dropped "
            f"{counts.dropped_not_all_energies} not ALL_ENERGIES, "
            f"{counts.dropped_ambiguous_identity} matched to other library "
            f"compounds, {counts.dropped_undetectable_adduct} with an adduct "
            "SIRIUS cannot detect."
        ),
        (
            f"{summary.spectra_without_ms1} spectrum(s) had no measured MS1 scan "
            "holding their precursor and were sent to SIRIUS with MS2 only."
        ),
        (
            f"generate-groundtruth: processed {summary.chunks_processed} chunk(s) "
            f"({summary.cache_hits} cache hit(s), {newly_run} newly run), covering "
            f"{summary.features_seen} feature(s) and {summary.annotations_seen} "
            "annotation(s)."
        ),
    ]
    if summary.failures:
        lines.append(f"{len(summary.failures)} chunk(s) FAILED:")
        lines.extend(
            f"  - {failure.chunk_path.name}: {failure.error}"
            for failure in summary.failures
        )
    return "\n".join(lines)


def _delete_database(db_path: str) -> None:
    """Delete the DuckDB file at `db_path` and its write-ahead log, if present."""
    Path(db_path).unlink(missing_ok=True)
    Path(f"{db_path}.wal").unlink(missing_ok=True)


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
    "--ground-truth-revision",
    "ground_truth_revision",
    type=str,
    default=None,
    help="Override the config's pinned MSnLib HuggingFace dataset revision.",
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
    "--ionization-mode",
    "ionization_mode",
    type=click.Choice([*IONIZATION_MODES, "all"]),
    default=None,
    help="Only process chunks of this Ionization mode (default: all).",
)
@click.option(
    "--smoke-test",
    "smoke_test",
    is_flag=True,
    default=False,
    help=(
        "Reset the config's [smoke_test] database and populate it from only "
        "the first [smoke_test] spectra_per_ionization_mode spectra of each "
        "Ionization mode."
    ),
)
def main(
    config_path: Path | None,
    db_path: str | None,
    ground_truth_revision: str | None,
    force: bool,
    ionization_mode: str | None,
    smoke_test: bool,
) -> None:
    """Fetch MSnLib, run it through cache-aware SIRIUS, populate ground truth."""
    if smoke_test and db_path is not None:
        raise click.UsageError("--smoke-test cannot be combined with --db.")
    if smoke_test and ionization_mode is not None:
        raise click.UsageError(
            "--smoke-test cannot be combined with --ionization-mode: a smoke "
            "test always covers both Ionization modes."
        )
    try:
        config = load_config(
            config_path if config_path is not None else DEFAULT_CONFIG_PATH,
            overrides={
                "db_path": db_path,
                "ground_truth_revision": ground_truth_revision,
            },
        )
        spectra_per_ionization_mode = None
        if smoke_test:
            config, smoke_test_config = for_smoke_test(config)
            spectra_per_ionization_mode = smoke_test_config.spectra_per_ionization_mode
            _delete_database(config.db_path)
        engine = create_database(config.db_path)
        try:
            sirius = Sirius(headless=True)
            try:
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
                        ionization_mode=ionization_mode or "all",
                        spectra_per_ionization_mode=spectra_per_ionization_mode,
                    )
            finally:
                # Covers runs that end before or without a per-chunk
                # shutdown (no chunks, a fetch/version error, Ctrl-C);
                # a no-op when the last chunk already shut SIRIUS down.
                sirius.shutdown()
        finally:
            engine.dispose()
    except _CLICK_EXCEPTION_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_format_summary(summary))
    if summary.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

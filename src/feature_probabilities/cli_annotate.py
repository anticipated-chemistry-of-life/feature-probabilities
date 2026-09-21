"""``annotate``: the happy-path field-mzML batch CLI.

Issue #23's happy-path slice: given the shared TOML config (#11), a
directory of new mzML files (``--mzml-dir``), and a metadata CSV
(``--metadata-csv``), upserts the batch's ``species``/``extracts`` rows
(#17) and runs every mzML file through the cache-aware SIRIUS pipeline
(#19) with ``source_kind='field_mzml'``, the mzML's peak-picking
(``LcmsSubmissionParameters``) parameters from ``config.sirius.import_params``,
and the just-upserted file's ``extract_id``.

Calibration application, ``--export``, ``--kde-model`` selection,
``--force``, and per-file batch-failure isolation are all deliberately out
of scope here (issue #23's body) -- deferred to later tickets. A single bad
file (a version mismatch, a malformed metadata CSV, or an mzML with no
matching metadata row) aborts the whole run with a clear, non-zero-exit
error rather than silently skipping it or partially committing.

``ionization_mode``/``instrument_type`` are per-batch CLI flags rather than
metadata-CSV columns: per ``CONTEXT.md``'s Extract entry, they're
acquisition-specific detail that lives on the SIRIUS run, not the physical
sample -- the metadata CSV (#17) only ever upserts ``species``/``extracts``
rows.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import click
import PySirius
from PySirius import JobSubmission, LcmsSubmissionParameters
from sqlalchemy.orm import Session

from feature_probabilities.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
)
from feature_probabilities.metadata import (
    MetadataError,
    find_metadata_row_for_mzml,
    load_metadata_csv,
    upsert_metadata_row,
)
from feature_probabilities.run_cache import (
    RunCacheError,
    SiriusRunRequest,
    get_or_create_run,
)
from feature_probabilities.schema import SOURCE_KIND_FIELD_MZML, create_database
from feature_probabilities.sirius import (
    Sirius,
    SiriusCredentialsError,
    SiriusStartupError,
    SiriusVersionMismatchError,
    SiriusVersionUnavailableError,
    require_matching_sirius_version,
)

if TYPE_CHECKING:
    from feature_probabilities.sirius import SiriusInterface

#: Case-insensitive file extension identifying an mzML file in `--mzml-dir`.
_MZML_SUFFIX = ".mzml"


class AnnotateError(Exception):
    """Raised when `--mzml-dir` isn't a usable directory."""


#: Every error this CLI's `main` treats as a clean, actionable failure
#: rather than a raw traceback.
_CLICK_EXCEPTION_ERRORS = (
    AnnotateError,
    ConfigError,
    MetadataError,
    RunCacheError,
    SiriusCredentialsError,
    SiriusStartupError,
    SiriusVersionMismatchError,
    SiriusVersionUnavailableError,
)


@dataclass(frozen=True, slots=True)
class AnnotateSummary:
    """A basic success summary of what one `annotate_batch` call did."""

    files_processed: int
    cache_hits: int
    features_seen: int
    annotations_seen: int


def _mzml_files(mzml_dir: Path) -> list[Path]:
    """Every `--mzml-dir` mzML file, sorted by name for a deterministic order.

    Raises:
        AnnotateError: `mzml_dir` doesn't exist or isn't a directory.
    """
    if not mzml_dir.is_dir():
        raise AnnotateError(f"mzML directory not found: {mzml_dir}")
    return sorted(
        (
            path
            for path in mzml_dir.iterdir()
            if path.is_file() and path.suffix.lower() == _MZML_SUFFIX
        ),
        key=lambda path: path.name,
    )


def annotate_batch(
    session: Session,
    sirius: SiriusInterface,
    config: Config,
    *,
    mzml_dir: Path,
    metadata_csv: Path,
    work_dir: Path,
    ionization_mode: str,
    instrument_type: str,
) -> AnnotateSummary:
    """Upsert `metadata_csv` and run every `mzml_dir` file through cache-aware SIRIUS.

    For each mzML file (processed in sorted-by-name order): looks up its
    metadata row by filename/`sample_code` (`find_metadata_row_for_mzml`),
    upserts the corresponding `species`/`extracts` rows
    (`upsert_metadata_row`), then runs it through `run_cache.get_or_create_run`
    with `source_kind='field_mzml'`, the just-upserted row's `extract_id`,
    and `config.sirius.import_params`/`analysis_params`.

    `commit`s `session` once, after every file succeeds -- a single bad
    file (a missing metadata row, a SIRIUS failure) aborts the whole batch
    with nothing persisted, rather than partially committing; per-file
    batch-failure isolation is deferred (issue #23's body).

    Raises:
        SiriusVersionMismatchError: the installed SIRIUS version doesn't
            match `config.required_sirius_version`.
        AnnotateError: `mzml_dir` doesn't exist or isn't a directory.
        MetadataError: `metadata_csv` is malformed, or an mzML file has no
            matching metadata row.
        RunCacheError: propagated from `get_or_create_run`.
    """
    require_matching_sirius_version(sirius, config.required_sirius_version)

    mzml_paths = _mzml_files(mzml_dir)
    metadata_df = load_metadata_csv(metadata_csv)

    import_params = LcmsSubmissionParameters(**config.sirius.import_params)
    analysis_params = JobSubmission(**config.sirius.analysis_params)

    files_processed = 0
    cache_hits = 0
    features_seen = 0
    annotations_seen = 0
    for mzml_path in mzml_paths:
        row = find_metadata_row_for_mzml(metadata_df, mzml_path)
        extract = upsert_metadata_row(session, row, csv_path=metadata_csv)

        request = SiriusRunRequest(
            input_file=mzml_path,
            project_path=work_dir / "projects" / mzml_path.stem,
            source_kind=SOURCE_KIND_FIELD_MZML,
            extract_id=extract.extract_id,
            import_params=import_params,
            analysis_params=analysis_params,
            sirius_version=config.required_sirius_version,
            pysirius_client_version=PySirius.__version__,
            ionization_mode=ionization_mode,
            instrument_type=instrument_type,
        )
        result = get_or_create_run(session, sirius, request)

        files_processed += 1
        cache_hits += 1 if result.from_cache else 0
        features_seen += len(result.features)
        annotations_seen += len(result.annotations)

    session.commit()

    return AnnotateSummary(
        files_processed=files_processed,
        cache_hits=cache_hits,
        features_seen=features_seen,
        annotations_seen=annotations_seen,
    )


def _format_summary(summary: AnnotateSummary) -> str:
    newly_run = summary.files_processed - summary.cache_hits
    return (
        f"annotate: processed {summary.files_processed} mzML file(s) "
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
    "--mzml-dir",
    "mzml_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory of new mzML files to process.",
)
@click.option(
    "--metadata-csv",
    "metadata_csv",
    type=click.Path(path_type=Path),
    required=True,
    help="CSV of species/extract metadata to upsert, keyed by sample_code.",
)
@click.option(
    "--ionization-mode",
    "ionization_mode",
    type=click.Choice(["positive", "negative"]),
    required=True,
    help="Ionization polarity SIRIUS should record for every run in this batch.",
)
@click.option(
    "--instrument-type",
    "instrument_type",
    type=str,
    required=True,
    help=(
        "Instrument type SIRIUS should record for every run in this batch "
        "(e.g. Orbitrap, QTOF)."
    ),
)
def main(
    config_path: Path | None,
    db_path: str | None,
    mzml_dir: Path,
    metadata_csv: Path,
    ionization_mode: str,
    instrument_type: str,
) -> None:
    """Upsert batch metadata and run new mzML files through cache-aware SIRIUS."""
    try:
        config = load_config(
            config_path if config_path is not None else DEFAULT_CONFIG_PATH,
            overrides={"db_path": db_path},
        )
        engine = create_database(config.db_path)
        try:
            sirius = Sirius(headless=True)
            with (
                tempfile.TemporaryDirectory(prefix="annotate-") as tmp,
                Session(engine) as session,
            ):
                summary = annotate_batch(
                    session,
                    sirius,
                    config,
                    mzml_dir=mzml_dir,
                    metadata_csv=metadata_csv,
                    work_dir=Path(tmp),
                    ionization_mode=ionization_mode,
                    instrument_type=instrument_type,
                )
        finally:
            engine.dispose()
    except _CLICK_EXCEPTION_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_format_summary(summary))


if __name__ == "__main__":
    main()

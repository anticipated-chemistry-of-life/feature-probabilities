"""``annotate``: the field-mzML batch CLI producing the feature-probability table.

Issue #23's happy-path slice (given the shared TOML config (#11), a
directory of new mzML files (``--mzml-dir``), and a metadata CSV
(``--metadata-csv``), upserts the batch's ``species``/``extracts`` rows
(#17) and runs every mzML file through the cache-aware SIRIUS pipeline
(#19) with ``source_kind='field_mzml'``, the mzML's peak-picking
(``LcmsSubmissionParameters``) parameters from ``config.sirius.import_params``,
and the just-upserted file's ``extract_id``) plus issue #25's remaining CLI
surface:

- ``--kde-model <path>`` selects the ``kde_models`` row whose
  ``artifact_path`` equals ``<path>``; omitted, it defaults to the most
  recently fitted row (highest ``kde_model_id``). Every processed file's
  annotations (both freshly run and cache-hit) are scored against it via
  ``calibrate.apply_calibration`` (#24) -- re-pointing ``--kde-model`` at a
  different, already-fitted pickle for an already-processed batch adds a
  second, coexisting set of ``calibration_scores`` rows without any new
  SIRIUS calls (a pure cache hit); re-running with the *same* model
  overwrites those rows in place (`apply_calibration`'s upsert semantics),
  keeping an unchanged rerun idempotent.
- ``--force`` sets the SIRIUS ``JobSubmission.recompute`` field and passes
  ``force=True`` through to `run_cache.get_or_create_run`, bypassing the
  cache lookup for every file.
- A single mzML's SIRIUS-and-calibration failure (any exception raised
  while running, persisting, or scoring it) is caught, recorded into the
  batch's summary, and rolled back in isolation -- `annotate_batch` now
  `commit`s after every file (success or failure) rather than once at the
  end, so an earlier file's successful, already-committed work survives a
  later file's failure. A malformed batch-level input (a missing metadata
  row, a SIRIUS version mismatch) still aborts the whole run immediately,
  before any file is processed. `main` exits non-zero and prints every
  failed file's name and error in the final summary whenever
  `AnnotateSummary.failures` is non-empty.
- ``--export <path>`` writes the batch's feature-probability table -- one
  row per feature-structure-candidate pair from every successfully
  processed file, with its calibration score -- to CSV or Parquet
  (selected by ``<path>``'s suffix): the literal deliverable `CONTEXT.md`'s
  Feature-probability table entry names.

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
import pandas as pd
import PySirius
from PySirius import JobSubmission, LcmsSubmissionParameters
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.calibrate import apply_calibration, load_kdes
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
from feature_probabilities.schema import (
    SOURCE_KIND_FIELD_MZML,
    Annotation,
    CalibrationScore,
    Extract,
    Feature,
    KdeModel,
    Molecule,
    SiriusRun,
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

    from feature_probabilities.sirius import SiriusInterface

#: Case-insensitive file extension identifying an mzML file in `--mzml-dir`.
_MZML_SUFFIX = ".mzml"

#: `--export` output columns, one row per feature-structure-candidate pair.
_EXPORT_COLUMNS = (
    "annotation_id",
    "feature_id",
    "external_feature_id",
    "sample_code",
    "ion_mass",
    "molecule_id",
    "inchikey",
    "smiles",
    "csi_score",
    "rank",
    "adduct",
    "calibration_score",
)

#: `--export` suffixes this CLI knows how to write, mapped to the
#: `pandas.DataFrame` writer method name that produces them.
_EXPORT_WRITERS = {".csv": "to_csv", ".parquet": "to_parquet"}


class AnnotateError(Exception):
    """Raised for a clean, actionable batch-level input problem.

    Covers an unusable `--mzml-dir` and an unsupported `--export` file
    extension.
    """


class NoKdeModelError(Exception):
    """Raised when `--kde-model` (or its default) can't be resolved.

    Either `--kde-model` names a path with no matching `kde_models` row, or
    it was omitted and the table has no rows at all -- `fit-kde` hasn't run
    yet.
    """


#: Every error this CLI's `main` treats as a clean, actionable failure
#: rather than a raw traceback.
_CLICK_EXCEPTION_ERRORS = (
    AnnotateError,
    ConfigError,
    MetadataError,
    NoKdeModelError,
    RunCacheError,
    SiriusCredentialsError,
    SiriusStartupError,
    SiriusVersionMismatchError,
    SiriusVersionUnavailableError,
)


@dataclass(frozen=True, slots=True)
class FileFailure:
    """One mzML file whose SIRIUS run or calibration raised, per issue #25.

    `error` is `str(exc)` rather than the exception object itself, so
    `AnnotateSummary` (and anything printing it) stays a plain,
    repr-friendly data value.
    """

    mzml_path: Path
    error: str


@dataclass(frozen=True, slots=True)
class AnnotateSummary:
    """A basic success/failure summary of what one `annotate_batch` call did."""

    files_processed: int
    cache_hits: int
    features_seen: int
    annotations_seen: int
    kde_model_id: int
    calibration_scores_seen: int
    failures: tuple[FileFailure, ...] = ()


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


def _resolve_kde_model(session: Session, kde_model_path: Path | None) -> KdeModel:
    """The `kde_models` row `--kde-model` selects, or the most recently fitted one.

    "Most recently fitted" is the highest `kde_model_id` -- the same
    monotonic-sequence idiom `run_cache._find_matching_run` uses for "most
    recent", rather than `fitted_at`, which two rows inserted in the same
    test/batch can tie on at DuckDB's `now()` resolution.

    Raises:
        NoKdeModelError: `kde_model_path` matches no `kde_models.artifact_path`,
            or (when omitted) the table has zero rows.
    """
    if kde_model_path is not None:
        kde_model = session.scalars(
            select(KdeModel).where(KdeModel.artifact_path == str(kde_model_path))
        ).one_or_none()
        if kde_model is None:
            raise NoKdeModelError(
                f"No kde_models row found with artifact_path={str(kde_model_path)!r}. "
                "Pass the exact path a fit-kde run printed."
            )
        return kde_model

    kde_model = session.scalars(
        select(KdeModel).order_by(KdeModel.kde_model_id.desc())
    ).first()
    if kde_model is None:
        raise NoKdeModelError(
            "No kde_models row found. Run fit-kde first, or pass --kde-model "
            "explicitly."
        )
    return kde_model


def _export_rows(
    session: Session, run_ids: Sequence[int], kde_model_id: int
) -> pd.DataFrame:
    """Every processed `run_ids` annotation, joined into one export row each.

    Inner-joins `calibration_scores` on `kde_model_id`: every annotation
    `annotate_batch` processes this run was just scored against exactly
    that model (`apply_calibration` covers the whole `result.annotations`
    set, cache hit or not), so no annotation is dropped by the join.
    """
    if not run_ids:
        return pd.DataFrame(columns=_EXPORT_COLUMNS)

    stmt = (
        select(
            Annotation.annotation_id,
            Feature.feature_id,
            Feature.external_feature_id,
            Extract.sample_code,
            Feature.ion_mass,
            Molecule.molecule_id,
            Molecule.inchikey,
            Molecule.smiles,
            Annotation.csi_score,
            Annotation.rank,
            Annotation.adduct,
            CalibrationScore.score,
        )
        .join(Feature, Annotation.feature_id == Feature.feature_id)
        .join(SiriusRun, Feature.run_id == SiriusRun.run_id)
        .join(Extract, SiriusRun.extract_id == Extract.extract_id)
        .join(Molecule, Annotation.molecule_id == Molecule.molecule_id)
        .join(
            CalibrationScore,
            (CalibrationScore.annotation_id == Annotation.annotation_id)
            & (CalibrationScore.kde_model_id == kde_model_id),
        )
        .where(Feature.run_id.in_(run_ids))
        .order_by(Feature.feature_id, Annotation.annotation_id)
    )
    rows = session.execute(stmt).all()
    return pd.DataFrame(rows, columns=_EXPORT_COLUMNS)


def _export_writer_name(export_path: Path) -> str:
    """The `pandas.DataFrame` writer method name for `export_path`'s suffix.

    Raises:
        AnnotateError: `export_path`'s suffix isn't `.csv` or `.parquet`.
    """
    writer_name = _EXPORT_WRITERS.get(export_path.suffix.lower())
    if writer_name is None:
        raise AnnotateError(
            f"Unsupported --export file extension: {export_path.suffix!r} "
            f"(expected one of {sorted(_EXPORT_WRITERS)})"
        )
    return writer_name


def _write_export(df: pd.DataFrame, export_path: Path, writer_name: str) -> None:
    """Write `df` to `export_path` via its already-resolved `writer_name`."""
    export_path.parent.mkdir(parents=True, exist_ok=True)
    getattr(df, writer_name)(export_path, index=False)


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
    kde_model_path: Path | None = None,
    force: bool = False,
    export_path: Path | None = None,
) -> AnnotateSummary:
    """Upsert `metadata_csv` and run every `mzml_dir` file through cache-aware SIRIUS.

    Resolves the `kde_models` row to apply (`_resolve_kde_model`) once,
    up front. For each mzML file (processed in sorted-by-name order): looks
    up its metadata row by filename/`sample_code`
    (`find_metadata_row_for_mzml`), upserts the corresponding
    `species`/`extracts` rows (`upsert_metadata_row`), runs it through
    `run_cache.get_or_create_run` with `source_kind='field_mzml'`,
    `force=force`, the just-upserted row's `extract_id`, and
    `config.sirius.import_params`/`analysis_params` (with
    `JobSubmission.recompute` also set from `force`), then scores every one
    of its annotations (`apply_calibration`) against the resolved model.

    `commit`s `session` after every file -- on success, so an earlier
    file's work survives a later file's failure; on failure, `rollback`s
    first, then records a `FileFailure` and moves on to the next file
    rather than aborting the whole batch. Only errors raised *before* any
    file is processed (SIRIUS version mismatch, an unresolvable
    `--kde-model`, a missing metadata row) propagate out of this function.
    If `export_path` is given, writes the feature-probability table for
    every successfully processed file once the batch finishes.

    Raises:
        SiriusVersionMismatchError: the installed SIRIUS version doesn't
            match `config.required_sirius_version`.
        NoKdeModelError: `kde_model_path` doesn't resolve to a `kde_models`
            row.
        AnnotateError: `mzml_dir` doesn't exist or isn't a directory, or
            `export_path`'s extension is unsupported.
        MetadataError: `metadata_csv` is malformed, or an mzML file has no
            matching metadata row.
    """
    require_matching_sirius_version(sirius, config.required_sirius_version)
    kde_model = _resolve_kde_model(session, kde_model_path)
    kdes = load_kdes(kde_model.artifact_path)
    export_writer_name = (
        _export_writer_name(export_path) if export_path is not None else None
    )

    mzml_paths = _mzml_files(mzml_dir)
    metadata_df = load_metadata_csv(metadata_csv)

    import_params = LcmsSubmissionParameters(**config.sirius.import_params)
    analysis_params = JobSubmission(
        **{**config.sirius.analysis_params, "recompute": force}
    )

    files_processed = 0
    cache_hits = 0
    features_seen = 0
    annotations_seen = 0
    calibration_scores_seen = 0
    processed_run_ids: list[int] = []
    failures: list[FileFailure] = []
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
        try:
            result = get_or_create_run(session, sirius, request, force=force)
            calibration_summary = apply_calibration(
                session,
                kdes,
                kde_model.kde_model_id,
                annotation_ids=[
                    annotation.annotation_id for annotation in result.annotations
                ],
            )
            session.commit()
        except Exception as exc:  # noqa: BLE001 -- one bad file must not abort the batch
            session.rollback()
            failures.append(FileFailure(mzml_path=mzml_path, error=str(exc)))
            continue

        files_processed += 1
        cache_hits += 1 if result.from_cache else 0
        features_seen += len(result.features)
        annotations_seen += len(result.annotations)
        calibration_scores_seen += calibration_summary.annotations_scored
        processed_run_ids.append(result.run.run_id)

    if export_path is not None and export_writer_name is not None:  # narrows for type-checkers
        _write_export(
            _export_rows(session, processed_run_ids, kde_model.kde_model_id),
            export_path,
            export_writer_name,
        )

    return AnnotateSummary(
        files_processed=files_processed,
        cache_hits=cache_hits,
        features_seen=features_seen,
        annotations_seen=annotations_seen,
        kde_model_id=kde_model.kde_model_id,
        calibration_scores_seen=calibration_scores_seen,
        failures=tuple(failures),
    )


def _format_summary(summary: AnnotateSummary) -> str:
    newly_run = summary.files_processed - summary.cache_hits
    lines = [
        (
            f"annotate: processed {summary.files_processed} mzML file(s) "
            f"({summary.cache_hits} cache hit(s), {newly_run} newly run), covering "
            f"{summary.features_seen} feature(s), {summary.annotations_seen} "
            f"annotation(s), and {summary.calibration_scores_seen} calibration "
            f"score(s) from kde_models row {summary.kde_model_id}."
        )
    ]
    if summary.failures:
        lines.append(f"{len(summary.failures)} file(s) FAILED:")
        lines.extend(
            f"  - {failure.mzml_path.name}: {failure.error}"
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
@click.option(
    "--kde-model",
    "kde_model_path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Path of the fitted kde_models pickle to apply (default: the most "
        "recently fitted row)."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help=(
        "Bypass the sirius_runs cache and set JobSubmission.recompute so "
        "every file is recomputed by SIRIUS."
    ),
)
@click.option(
    "--export",
    "export_path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Write the batch's feature-probability table (.csv or .parquet) "
        "for every successfully processed file."
    ),
)
def main(
    config_path: Path | None,
    db_path: str | None,
    mzml_dir: Path,
    metadata_csv: Path,
    ionization_mode: str,
    instrument_type: str,
    kde_model_path: Path | None,
    force: bool,
    export_path: Path | None,
) -> None:
    """Upsert batch metadata, run new mzML files through cache-aware SIRIUS, and calibrate."""
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
                    kde_model_path=kde_model_path,
                    force=force,
                    export_path=export_path,
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

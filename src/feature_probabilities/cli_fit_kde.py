"""``fit-kde``: turns ground truth into a pickled calibration artifact.

Reads every ground-truth `features`/`annotations` row already in the DB
(populated by `generate-groundtruth`, #20/#21 -- this executable never talks
to SIRIUS or MassSpecGym itself), applies the correctness-matching-and-
deduplication transform (`groundtruth.correct_assignment_rows`, #18), fits
one `scipy.stats.gaussian_kde` per stratum (`kde.fit_stratified_kdes`, per
the KDE calibration method decision, issue #7) -- `Orbitrap`, `QTOF`, and a
pooled `__global__` over every instrument type combined -- and persists both
the pickle and its `kde_models` row.

Exits non-zero with a clear message (rather than a raw fitting traceback) if
there is no usable ground-truth data yet, telling the operator to run
`generate-groundtruth` first.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    load_config,
)
from feature_probabilities.groundtruth import (
    GroundTruthAnnotationRow,
    correct_assignment_rows,
)
from feature_probabilities.kde import EmptyStratumError, fit_stratified_kdes
from feature_probabilities.schema import (
    SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
    Annotation,
    Feature,
    KdeModel,
    Molecule,
    SiriusRun,
    create_database,
)

if TYPE_CHECKING:
    from scipy.stats import gaussian_kde


class NoGroundTruthDataError(Exception):
    """Raised when the DB has zero correct-assignment rows to fit a KDE on."""


#: Every error this CLI's `main` treats as a clean, actionable failure
#: rather than a raw traceback.
_CLICK_EXCEPTION_ERRORS = (
    ConfigError,
    EmptyStratumError,
    NoGroundTruthDataError,
)


@dataclass(frozen=True, slots=True)
class FitKdeSummary:
    """A basic summary of what one `fit_kde` call did."""

    ground_truth_rows_seen: int
    correct_assignment_rows_used: int
    strata_fitted: tuple[str, ...]
    artifact_path: Path
    kde_model_id: int


def _ground_truth_rows(session: Session) -> list[GroundTruthAnnotationRow]:
    """Every ground-truth-run feature-structure-candidate pair, joined and flattened.

    Restricted to `sirius_runs.source_kind == 'ground_truth_massspecgym'` rows
    whose `features.true_inchikey` was actually resolved (field-run features
    always have it NULL, and a ground-truth feature MassSpecGym's identifier
    lookup missed is unusable for the correctness match).
    """
    stmt = (
        select(Feature, Annotation, Molecule, SiriusRun)
        .join(SiriusRun, Feature.run_id == SiriusRun.run_id)
        .join(Annotation, Annotation.feature_id == Feature.feature_id)
        .join(Molecule, Annotation.molecule_id == Molecule.molecule_id)
        .where(Feature.true_inchikey.is_not(None))
        .where(SiriusRun.source_kind == SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM)
    )
    return [
        GroundTruthAnnotationRow(
            feature_id=feature.feature_id,
            candidate_inchikey=molecule.inchikey,
            true_inchikey=feature.true_inchikey,
            adduct=annotation.adduct,
            instrument_type=sirius_run.instrument_type,
            ion_mass=feature.ion_mass,
            csi_score=annotation.csi_score,
        )
        for feature, annotation, molecule, sirius_run in session.execute(stmt).all()
    ]


def _pickle_kdes(kdes: dict[str, gaussian_kde], output_path: Path) -> None:
    """Write `kdes` to `output_path` via `pickle`.

    Trust boundary: `pickle` is the only format that round-trips a fitted
    `scipy.stats.gaussian_kde` (its dataset, covariance, and bandwidth
    factor aren't JSON-serializable), and the artifact never crosses a
    trust boundary -- `fit-kde` writes it and `annotate` (same deployment,
    same operator-controlled filesystem) is the only reader, per the
    `kde_models.artifact_path` "pickled calibration model" schema contract
    (issue #4/#7/#22).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(kdes, handle)


def fit_kde(session: Session, *, output_path: Path) -> FitKdeSummary:
    """Fit every stratum's KDE from `session`'s ground truth and persist it.

    Pickles the fitted `{"Orbitrap": kde, "QTOF": kde, "__global__": kde}`
    dict to `output_path` and inserts+commits one `kde_models` row pointing
    at it.

    Raises:
        NoGroundTruthDataError: there are zero correct-assignment rows to
            fit on -- `generate-groundtruth` hasn't produced usable data yet.
        EmptyStratumError: propagated from `kde.fit_stratified_kdes` if a
            trained stratum has correct-assignment rows for the *other*
            instrument type but none for its own.
    """
    ground_truth_rows = _ground_truth_rows(session)
    correct_rows = correct_assignment_rows(ground_truth_rows)
    if not correct_rows:
        raise NoGroundTruthDataError(
            "No correct-assignment ground-truth rows found in the database. "
            "Run generate-groundtruth first to populate ground-truth data."
        )

    kdes = fit_stratified_kdes(correct_rows)
    _pickle_kdes(kdes, output_path)

    kde_model = KdeModel(artifact_path=str(output_path))
    session.add(kde_model)
    session.commit()

    return FitKdeSummary(
        ground_truth_rows_seen=len(ground_truth_rows),
        correct_assignment_rows_used=len(correct_rows),
        strata_fitted=tuple(sorted(kdes)),
        artifact_path=output_path,
        kde_model_id=kde_model.kde_model_id,
    )


def _format_summary(summary: FitKdeSummary) -> str:
    return "\n".join(
        [
            f"Ground-truth rows seen: {summary.ground_truth_rows_seen}",
            f"Correct-assignment rows used: {summary.correct_assignment_rows_used}",
            f"Strata fitted: {', '.join(summary.strata_fitted)}",
            f"Artifact written to: {summary.artifact_path}",
            f"kde_models row: {summary.kde_model_id}",
        ]
    )


def _default_output_path(config: Config) -> Path:
    """`config.kde_output_path` with a UTC timestamp inserted before its suffix.

    Applied only when `--output` isn't passed: every default-path
    `fit-kde` invocation then writes a distinct file, so a later run never
    silently overwrites an earlier `kde_models` row's `artifact_path` out
    from under it. An explicit `--output` is a fully user-controlled,
    deliberately reusable target and is used as-is.
    """
    base = Path(config.kde_output_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    return base.with_name(f"{base.stem}_{timestamp}{base.suffix}")


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
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the config's kde_output_path for the pickled artifact.",
)
def main(
    config_path: Path | None,
    db_path: str | None,
    output_path: Path | None,
) -> None:
    """Fit stratified KDE calibration models from the ground-truth table."""
    try:
        config = load_config(
            config_path if config_path is not None else DEFAULT_CONFIG_PATH,
            overrides={"db_path": db_path},
        )
        resolved_output_path = (
            output_path if output_path is not None else _default_output_path(config)
        )
        engine = create_database(config.db_path)
        try:
            with Session(engine) as session:
                summary = fit_kde(session, output_path=resolved_output_path)
        finally:
            engine.dispose()
    except _CLICK_EXCEPTION_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_format_summary(summary))


if __name__ == "__main__":
    main()

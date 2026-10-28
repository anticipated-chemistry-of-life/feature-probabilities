"""``fit-kde``: turns ground truth into a pickled calibration artifact.

Reads every ground-truth `features`/`annotations` row already in the DB
(populated by `generate-groundtruth`, #20/#21 -- this executable never talks
to SIRIUS or fetches the ground truth set itself), applies the
correctness-matching-and-deduplication transform
(`groundtruth.correct_assignment_rows`, #18) and
`groundtruth.incorrect_assignment_rows` (ADR 0004), fits per Ionization mode
-- `positive` and `negative`, each only on its own rows
(`kde.fit_stratified_kdes`, ADR 0002) -- a correct density on the correct
assignments and an incorrect density on a seeded subsample of the incorrect
ones, and persists both the pickle and its `kde_models` row (recording the
subsample seed and size).

Exits non-zero with a clear message (rather than a raw fitting traceback) if
there is no usable ground-truth data yet, telling the operator to run
`generate-groundtruth` first, or if either Ionization mode has no correct
or no incorrect assignments, naming that mode.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import click
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ConfigError,
    for_smoke_test,
    load_config,
)
from feature_probabilities.groundtruth import (
    GroundTruthAnnotationRow,
    correct_assignment_rows,
    incorrect_assignment_rows,
)
from feature_probabilities.kde import (
    INCORRECT_SUBSAMPLE_SIZE,
    SUBSAMPLE_SEED,
    EmptyStratumError,
    ModeDensities,
    ModeDiagnostic,
    ScoreSummary,
    fit_stratified_kdes,
    incorrect_density_diagnostic,
)
from feature_probabilities.schema import (
    SOURCE_KIND_GROUND_TRUTH_MSNLIB,
    Annotation,
    Feature,
    KdeModel,
    Molecule,
    SiriusRun,
    create_database,
)


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
    diagnostics: dict[str, ModeDiagnostic]


def _ground_truth_rows(session: Session) -> list[GroundTruthAnnotationRow]:
    """Every ground-truth-run feature-structure-candidate pair, joined and flattened.

    Restricted to `sirius_runs.source_kind == 'ground_truth_msnlib'` rows
    whose `features.true_inchikey` was actually resolved, so a field-run
    Feature never contributes to the fit. Each row carries its SIRIUS run's
    Ionization mode, which selects the KDE it is fitted into.
    """
    stmt = (
        select(Feature, Annotation, Molecule, SiriusRun)
        .join(SiriusRun, Feature.run_id == SiriusRun.run_id)
        .join(Annotation, Annotation.feature_id == Feature.feature_id)
        .join(Molecule, Annotation.molecule_id == Molecule.molecule_id)
        .where(Feature.true_inchikey.is_not(None))
        .where(SiriusRun.source_kind == SOURCE_KIND_GROUND_TRUTH_MSNLIB)
    )
    return [
        GroundTruthAnnotationRow(
            feature_id=feature.feature_id,
            candidate_inchikey=molecule.inchikey,
            true_inchikey=feature.true_inchikey,
            adduct=annotation.adduct,
            ionization_mode=sirius_run.ionization_mode,
            ion_mass=feature.ion_mass,
            csi_score=annotation.csi_score,
        )
        for feature, annotation, molecule, sirius_run in session.execute(stmt).all()
    ]


def _pickle_kdes(kdes: dict[str, ModeDensities], output_path: Path) -> None:
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


def fit_kde(
    session: Session,
    *,
    output_path: Path,
    groundtruth_command: str = "generate-groundtruth",
) -> FitKdeSummary:
    """Fit a correct and an incorrect density per Ionization mode and persist them.

    Pickles the fitted `{"positive": ModeDensities, "negative": ModeDensities}` dict to
    `output_path` and inserts+commits one `kde_models` row pointing at it.
    `groundtruth_command` is the command a no-data error tells the operator
    to run.

    Raises:
        NoGroundTruthDataError: there are zero correct-assignment rows to
            fit on -- `generate-groundtruth` hasn't produced usable data yet.
        EmptyStratumError: propagated from `kde.fit_stratified_kdes` if an
            Ionization mode has no correct-assignment rows or no
            incorrect-assignment rows.
    """
    ground_truth_rows = _ground_truth_rows(session)
    correct_rows = correct_assignment_rows(ground_truth_rows)
    if not correct_rows:
        raise NoGroundTruthDataError(
            "No correct-assignment ground-truth rows found in the database. "
            f"Run {groundtruth_command} first to populate ground-truth data."
        )

    incorrect_rows = incorrect_assignment_rows(ground_truth_rows)
    kdes = fit_stratified_kdes(
        correct_rows,
        incorrect_rows,
        seed=SUBSAMPLE_SEED,
        subsample_size=INCORRECT_SUBSAMPLE_SIZE,
    )
    _pickle_kdes(kdes, output_path)

    kde_model = KdeModel(
        artifact_path=str(output_path),
        subsample_seed=SUBSAMPLE_SEED,
        incorrect_subsample_size=INCORRECT_SUBSAMPLE_SIZE,
        positive_incorrect_rows_used=kdes["positive"].incorrect.n,
        negative_incorrect_rows_used=kdes["negative"].incorrect.n,
    )
    session.add(kde_model)
    session.commit()

    return FitKdeSummary(
        ground_truth_rows_seen=len(ground_truth_rows),
        correct_assignment_rows_used=len(correct_rows),
        strata_fitted=tuple(sorted(kdes)),
        artifact_path=output_path,
        kde_model_id=kde_model.kde_model_id,
        diagnostics=incorrect_density_diagnostic(incorrect_rows, kdes),
    )


def _format_score_summary(label: str, summary: ScoreSummary | None) -> str:
    if summary is None:
        return f"  {label} truth: no incorrect rows"
    return (
        f"  {label} truth: {summary.rows} incorrect rows, CSI score median "
        f"{summary.median:.3f}, q10 {summary.q10:.3f}, q90 {summary.q90:.3f}"
    )


def _format_diagnostic(mode: str, diagnostic: ModeDiagnostic) -> list[str]:
    lines = [
        f"Incorrect-density diagnostic ({mode}):",
        _format_score_summary("listed", diagnostic.listed),
        _format_score_summary("unlisted", diagnostic.unlisted),
    ]
    difference = diagnostic.difference
    if difference is None:
        lines.append("  ln g0_listed - ln g0_unlisted: not computed")
    else:
        lines.append(
            f"  ln g0_listed - ln g0_unlisted: mean {difference.mean:.3f}, "
            f"q10 {difference.q10:.3f}, q90 {difference.q90:.3f}"
        )
    return lines


def _format_summary(summary: FitKdeSummary) -> str:
    lines = [
        f"Ground-truth rows seen: {summary.ground_truth_rows_seen}",
        f"Correct-assignment rows used: {summary.correct_assignment_rows_used}",
        f"Strata fitted: {', '.join(summary.strata_fitted)}",
        f"Artifact written to: {summary.artifact_path}",
        f"kde_models row: {summary.kde_model_id}",
    ]
    for mode, diagnostic in summary.diagnostics.items():
        lines.extend(_format_diagnostic(mode, diagnostic))
    return "\n".join(lines)


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
@click.option(
    "--smoke-test",
    "smoke_test",
    is_flag=True,
    default=False,
    help=(
        "Fit from the config's [smoke_test] database and write the pickle "
        "under its [smoke_test] kde_output_path."
    ),
)
def main(
    config_path: Path | None,
    db_path: str | None,
    output_path: Path | None,
    smoke_test: bool,
) -> None:
    """Fit one KDE calibration model per Ionization mode from the ground-truth table."""
    if smoke_test and db_path is not None:
        raise click.UsageError("--smoke-test cannot be combined with --db.")
    try:
        config = load_config(
            config_path if config_path is not None else DEFAULT_CONFIG_PATH,
            overrides={"db_path": db_path},
        )
        groundtruth_command = "generate-groundtruth"
        if smoke_test:
            config, _ = for_smoke_test(config)
            groundtruth_command = "generate-groundtruth --smoke-test"
        resolved_output_path = (
            output_path if output_path is not None else _default_output_path(config)
        )
        engine = create_database(config.db_path)
        try:
            with Session(engine) as session:
                summary = fit_kde(
                    session,
                    output_path=resolved_output_path,
                    groundtruth_command=groundtruth_command,
                )
        finally:
            engine.dispose()
    except _CLICK_EXCEPTION_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(_format_summary(summary))


if __name__ == "__main__":
    main()

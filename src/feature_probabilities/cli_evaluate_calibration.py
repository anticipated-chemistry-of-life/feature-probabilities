"""``evaluate-calibration``: held-out comparison of two ways to weight a Feature's candidates.

Reads the ground-truth rows already in the DB the way `fit-kde` does
(`cli_fit_kde.read_ground_truth_rows`), cross-validates the best-row baseline
against the likelihood ratio (`calibration_eval.evaluate_calibration`, issue
#38) and prints both side by side to stdout. Read-only: the DB is opened
read-only, and no artifact, `kde_models` row or file is written.

Exits non-zero with a clear message if the DB holds no ground truth yet,
telling the operator to run `generate-groundtruth` first.
"""

from __future__ import annotations

from pathlib import Path

import click
import sqlalchemy as sa
from sqlalchemy.orm import Session

from feature_probabilities.calibration_eval import (
    CalibrationReport,
    MethodReport,
    TooFewStructuresError,
    eval_features,
    evaluate_calibration,
)
from feature_probabilities.cli_fit_kde import (
    NoGroundTruthDataError,
    read_ground_truth_rows,
)
from feature_probabilities.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_config,
)
from feature_probabilities.groundtruth import GroundTruthAnnotationRow
from feature_probabilities.kde import EmptyStratumError

#: Every error this CLI's `main` treats as a clean, actionable failure
#: rather than a raw traceback.
_CLICK_EXCEPTION_ERRORS = (
    ConfigError,
    EmptyStratumError,
    NoGroundTruthDataError,
    TooFewStructuresError,
)

_NO_GROUND_TRUTH = (
    "No ground-truth rows found in the database. "
    "Run generate-groundtruth first to populate ground-truth data."
)

_LABEL_WIDTH = 34
_COLUMN_WIDTH = 24


def _read_ground_truth(db_path: str) -> list[GroundTruthAnnotationRow]:
    """The ground-truth rows of the DB at `db_path`, opened read-only.

    Raises:
        NoGroundTruthDataError: the DB file does not exist.
    """
    if not Path(db_path).exists():
        raise NoGroundTruthDataError(_NO_GROUND_TRUTH)
    engine = sa.create_engine(f"duckdb:///{db_path}", connect_args={"read_only": True})
    try:
        with Session(engine) as session:
            return read_ground_truth_rows(session)
    finally:
        engine.dispose()


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _row(label: str, baseline: str, likelihood_ratio: str) -> str:
    return (
        f"{label:<{_LABEL_WIDTH}}{baseline:<{_COLUMN_WIDTH}}{likelihood_ratio}".rstrip()
    )


def _metric_rows(baseline: MethodReport, ratio: MethodReport) -> list[str]:
    def both(label: str, field: str) -> str:
        return _row(label, _fmt(getattr(baseline, field)), _fmt(getattr(ratio, field)))

    return [
        both("top-1 accuracy", "top_1"),
        both("top-5 accuracy", "top_5"),
        both("mean Unknown-molecule posterior", "mean_unknown_posterior"),
        both("median Unknown-molecule posterior", "median_unknown_posterior"),
        both("held-out log-loss", "log_loss"),
        *(
            _row(f"fitted c, fold {fold}", f"{left:.4g}", f"{right:.4g}")
            for fold, (left, right) in enumerate(
                zip(baseline.fitted_c, ratio.fitted_c, strict=True), start=1
            )
        ),
    ]


def _reliability_cell(count: int, mean: float | None, correct: float | None) -> str:
    if count == 0:
        return "0"
    return f"{count} / {_fmt(mean)} / {_fmt(correct)}"


def _reliability_rows(baseline: MethodReport, ratio: MethodReport) -> list[str]:
    return [
        _row(
            f"[{left.lower:.1f}, {left.upper:.1f}"
            + ("]" if left.upper == 1.0 else ")"),
            _reliability_cell(left.count, left.mean_posterior, left.fraction_correct),
            _reliability_cell(
                right.count, right.mean_posterior, right.fraction_correct
            ),
        )
        for left, right in zip(baseline.reliability, ratio.reliability, strict=True)
    ]


def format_report(report: CalibrationReport, *, seed: int) -> str:
    """Both methods side by side."""
    baseline, ratio = report.baseline, report.likelihood_ratio
    return "\n".join(
        [
            (
                f"Held-out evaluation over {report.n_features} Features "
                f"({report.n_listed} with a correct assignment, {report.n_unlisted} "
                f"without), {report.folds} folds, seed {seed}."
            ),
            "",
            _row("", "baseline", "likelihood ratio"),
            *_metric_rows(baseline, ratio),
            "",
            (
                "Reliability over held-out candidate rows "
                "(count / mean posterior / fraction correct):"
            ),
            _row("posterior bin", "baseline", "likelihood ratio"),
            *_reliability_rows(baseline, ratio),
        ]
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
    "--seed",
    type=int,
    default=0,
    show_default=True,
    help="Seed of the fold assignment and the incorrect-row subsample.",
)
def main(config_path: Path | None, db_path: str | None, seed: int) -> None:
    """Compare the best-row baseline with the likelihood ratio on held-out ground truth."""
    try:
        config = load_config(
            config_path if config_path is not None else DEFAULT_CONFIG_PATH,
            overrides={"db_path": db_path},
        )
        features = eval_features(_read_ground_truth(config.db_path))
        if not features:
            raise NoGroundTruthDataError(_NO_GROUND_TRUTH)
        report = evaluate_calibration(features, seed=seed)
    except _CLICK_EXCEPTION_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(format_report(report, seed=seed))


if __name__ == "__main__":
    main()

"""Pure held-out evaluation of how a Feature's candidates are weighted.

Cross-validates two weightings of a Feature's structure candidates on the
Ground truth set (issue #38), without touching DuckDB or SIRIUS:

- the **baseline**: w_j = g1_j / max_k g1_k, the correct density over the
  Feature's best row, which the MRF uses today;
- the **likelihood ratio**: w_j = g1_j / g0_j, the correct density over the
  incorrect density (ADR 0004).

Both are computed in log space with a uniform prior over candidates. A
candidate's posterior is w_j / (sum of w + c) and the Unknown molecule's is
c / (sum of w + c).

Each distinct true InChIKey lands in exactly one of :data:`FOLDS` folds by a
seeded shuffle, so no InChIKey spans training and test. In every fold both
densities are refitted on the training Features by
`kde.fit_stratified_kdes`, the function `fit-kde` fits with, so g0 is
subsampled from training rows only. Features are scored in place: a
Feature's candidates are scored by the densities of its own Ionization
mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import batched
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize_scalar

from feature_probabilities.calibrate import EVALUATION_CHUNK_SIZE
from feature_probabilities.groundtruth import (
    CorrectAssignmentRow,
    GroundTruthAnnotationRow,
    IncorrectAssignmentRow,
    correct_assignment_rows,
    incorrect_assignment_rows,
    is_correct_assignment,
)
from feature_probabilities.kde import ModeDensities, fit_stratified_kdes

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from numpy.typing import NDArray

#: Cross-validation folds.
FOLDS = 5

#: Equal-width posterior bins of the reliability table.
RELIABILITY_BINS = 10


class TooFewMoleculesError(Exception):
    """Raised when there are fewer distinct true InChIKeys than folds."""


@dataclass(frozen=True, slots=True)
class EvalCandidate:
    """One structure candidate of a Feature: its (ion mass, CSI score) point and its truth."""

    ion_mass: float
    csi_score: float
    correct: bool


@dataclass(frozen=True, slots=True)
class EvalFeature:
    """One deduplicated ground truth Feature with at least one candidate.

    A Feature has a correct assignment when any candidate is `correct`;
    otherwise its truth is the Unknown molecule.
    """

    true_inchikey: str
    ionization_mode: str
    candidates: tuple[EvalCandidate, ...]


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One equal-width posterior bin over held-out candidate rows.

    `mean_posterior` and `fraction_correct` are None for an empty bin.
    """

    lower: float
    upper: float
    count: int
    mean_posterior: float | None
    fraction_correct: float | None


@dataclass(frozen=True, slots=True)
class MethodReport:
    """One weighting's pooled held-out results.

    `top_1` and `top_5` are over Features with a correct assignment, ranking
    candidates only. `reliability` covers every held-out candidate row. The
    Unknown-molecule posteriors are over Features without a correct
    assignment, None when there are none. `fitted_c` is each fold's c, in
    fold order. `log_loss` is the mean -ln posterior of the truth over all
    held-out Features.
    """

    top_1: float
    top_5: float
    fitted_c: tuple[float, ...]
    reliability: tuple[ReliabilityBin, ...]
    mean_unknown_posterior: float | None
    median_unknown_posterior: float | None
    log_loss: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Both weightings' held-out results over the same Features and folds."""

    n_features: int
    n_listed: int
    n_unlisted: int
    folds: int
    baseline: MethodReport
    likelihood_ratio: MethodReport


@dataclass(frozen=True, slots=True)
class _Scored:
    """A set of Features' candidates flattened, with both densities evaluated in log space.

    `starts[i]:starts[i + 1]` slices Feature `i`'s candidates out of the flat
    arrays.
    """

    log_correct: NDArray[np.float64]
    log_incorrect: NDArray[np.float64]
    correct: NDArray[np.bool_]
    starts: NDArray[np.intp]


def _assign_folds(features: Sequence[EvalFeature], seed: int) -> dict[str, int]:
    """Fold of every distinct true InChIKey: a seeded shuffle of the sorted keys, dealt round-robin."""
    keys = sorted({feature.true_inchikey for feature in features})
    if len(keys) < FOLDS:
        raise TooFewMoleculesError(
            f"Need at least {FOLDS} distinct true InChIKeys to cross-validate in "
            f"{FOLDS} folds, found {len(keys)}."
        )
    order = np.random.default_rng(seed).permutation(len(keys))
    return {
        keys[key_index]: position % FOLDS for position, key_index in enumerate(order)
    }


def _fit(training: Sequence[EvalFeature], seed: int) -> dict[str, ModeDensities]:
    """Both densities per Ionization mode, fitted on `training`'s candidates only."""
    correct_rows: list[CorrectAssignmentRow] = []
    incorrect_rows: list[IncorrectAssignmentRow] = []
    for feature_id, feature in enumerate(training):
        listed = any(candidate.correct for candidate in feature.candidates)
        for candidate in feature.candidates:
            if candidate.correct:
                correct_rows.append(
                    CorrectAssignmentRow(
                        feature_id=feature_id,
                        true_inchikey=feature.true_inchikey,
                        adduct="",
                        ionization_mode=feature.ionization_mode,
                        ion_mass=candidate.ion_mass,
                        csi_score=candidate.csi_score,
                    )
                )
            else:
                incorrect_rows.append(
                    IncorrectAssignmentRow(
                        feature_id=feature_id,
                        true_inchikey=feature.true_inchikey,
                        adduct="",
                        ionization_mode=feature.ionization_mode,
                        ion_mass=candidate.ion_mass,
                        csi_score=candidate.csi_score,
                        truth_listed=listed,
                    )
                )
    return fit_stratified_kdes(correct_rows, incorrect_rows, seed=seed)


def _score(
    features: Sequence[EvalFeature], densities: dict[str, ModeDensities]
) -> _Scored:
    """Evaluate both densities at every candidate of `features`, chunk by chunk per mode.

    `logpdf` rather than `pdf`: far from the training data a density
    underflows to 0.0, whose log is -inf.
    """
    sizes = [len(feature.candidates) for feature in features]
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.intp)
    total = int(starts[-1])
    ion_mass = np.empty(total)
    csi_score = np.empty(total)
    correct = np.empty(total, dtype=np.bool_)
    positions_by_mode: dict[str, list[int]] = {}
    for i, feature in enumerate(features):
        positions = range(starts[i], starts[i + 1])
        positions_by_mode.setdefault(feature.ionization_mode, []).extend(positions)
        for position, candidate in zip(positions, feature.candidates, strict=True):
            ion_mass[position] = candidate.ion_mass
            csi_score[position] = candidate.csi_score
            correct[position] = candidate.correct

    log_correct = np.empty(total)
    log_incorrect = np.empty(total)
    for ionization_mode, positions in positions_by_mode.items():
        for chunk in batched(positions, EVALUATION_CHUNK_SIZE):
            index = np.array(chunk)
            points = np.vstack([ion_mass[index], csi_score[index]])
            log_correct[index] = densities[ionization_mode].correct.logpdf(points)
            log_incorrect[index] = densities[ionization_mode].incorrect.logpdf(points)
    return _Scored(log_correct, log_incorrect, correct, starts)


def _truth_ranks(scored: _Scored, log_weight: NDArray[np.float64]) -> list[int]:
    """Truth rank (1-based) of every Feature with a correct assignment.

    Candidates rank by weight, best first; equal weights keep their input
    order, so ties break deterministically.
    """
    ranks = []
    for i in range(len(scored.starts) - 1):
        window = slice(scored.starts[i], scored.starts[i + 1])
        correct = scored.correct[window]
        if not correct.any():
            continue
        order = np.argsort(-log_weight[window], kind="stable")
        ranks.append(int(np.flatnonzero(correct[order])[0]) + 1)
    return ranks


def _baseline_log_weight(scored: _Scored) -> NDArray[np.float64]:
    """ln(g1_j / max_k g1_k): each Feature's candidates against its own best row."""
    best = np.maximum.reduceat(scored.log_correct, scored.starts[:-1])
    return scored.log_correct - np.repeat(best, np.diff(scored.starts))


def _likelihood_ratio_log_weight(scored: _Scored) -> NDArray[np.float64]:
    """ln(g1_j / g0_j)."""
    return scored.log_correct - scored.log_incorrect


#: The weightings compared, by report field name.
_LOG_WEIGHTS = {
    "baseline": _baseline_log_weight,
    "likelihood_ratio": _likelihood_ratio_log_weight,
}

#: Search range of ln c. Wide enough that the optimum is interior unless the
#: truth is (almost) never, or (almost) always, the Unknown molecule.
_LN_C_BOUNDS = (-100.0, 100.0)


@dataclass(frozen=True, slots=True)
class _Weights:
    """One weighting over a scored set, reduced to what a posterior needs.

    `log_truth` is ln of the correct candidates' weight, -inf for a Feature
    without a correct assignment, whose truth is the Unknown molecule.
    """

    log_weight: NDArray[np.float64]
    starts: NDArray[np.intp]
    log_sum: NDArray[np.float64]
    log_truth: NDArray[np.float64]

    @property
    def listed(self) -> NDArray[np.bool_]:
        """Per Feature, whether it has a correct assignment."""
        return np.isfinite(self.log_truth)

    def truth_loss(self, ln_c: float) -> NDArray[np.float64]:
        """Per Feature, -ln posterior of the truth at `c = exp(ln_c)`."""
        log_total = np.logaddexp(self.log_sum, ln_c)
        return np.where(self.listed, log_total - self.log_truth, log_total - ln_c)

    def candidate_posteriors(self, ln_c: float) -> NDArray[np.float64]:
        """Per candidate, w_j / (sum of w + c) within its Feature."""
        log_total = np.logaddexp(self.log_sum, ln_c)
        return np.exp(self.log_weight - np.repeat(log_total, np.diff(self.starts)))

    def unknown_posteriors(self, ln_c: float) -> NDArray[np.float64]:
        """Per Feature, c / (sum of w + c)."""
        return np.exp(ln_c - np.logaddexp(self.log_sum, ln_c))


def _weights(scored: _Scored, log_weight: NDArray[np.float64]) -> _Weights:
    firsts = scored.starts[:-1]
    return _Weights(
        log_weight=log_weight,
        starts=scored.starts,
        log_sum=np.logaddexp.reduceat(log_weight, firsts),
        log_truth=np.logaddexp.reduceat(
            np.where(scored.correct, log_weight, -np.inf), firsts
        ),
    )


def _fit_ln_c(weights: _Weights) -> float:
    """The ln c minimising the mean -ln posterior of the truth, by a bounded 1-D search."""
    result = minimize_scalar(
        lambda ln_c: float(np.mean(weights.truth_loss(ln_c))),
        bounds=_LN_C_BOUNDS,
        method="bounded",
    )
    return float(result.x)


@dataclass(slots=True)
class _Pool:
    """One weighting's held-out results, accumulated over folds."""

    ranks: list[int] = field(default_factory=list)
    fitted_c: list[float] = field(default_factory=list)
    posteriors: list[NDArray[np.float64]] = field(default_factory=list)
    correct: list[NDArray[np.bool_]] = field(default_factory=list)
    unknown_posteriors: list[NDArray[np.float64]] = field(default_factory=list)
    losses: list[NDArray[np.float64]] = field(default_factory=list)

    def add_fold(self, scored_test: _Scored, test: _Weights, ln_c: float) -> None:
        self.ranks += _truth_ranks(scored_test, test.log_weight)
        self.fitted_c.append(float(np.exp(ln_c)))
        self.posteriors.append(test.candidate_posteriors(ln_c))
        self.correct.append(scored_test.correct)
        self.unknown_posteriors.append(test.unknown_posteriors(ln_c)[~test.listed])
        self.losses.append(test.truth_loss(ln_c))

    def report(self) -> MethodReport:
        ranks = np.array(self.ranks)
        posteriors = np.concatenate(self.posteriors)
        correct = np.concatenate(self.correct)
        unknown = np.concatenate(self.unknown_posteriors)
        bins = np.minimum(
            (posteriors * RELIABILITY_BINS).astype(np.intp), RELIABILITY_BINS - 1
        )
        counts = np.bincount(bins, minlength=RELIABILITY_BINS)
        posterior_sums = np.bincount(
            bins, weights=posteriors, minlength=RELIABILITY_BINS
        )
        correct_counts = np.bincount(bins, weights=correct, minlength=RELIABILITY_BINS)
        return MethodReport(
            top_1=float(np.mean(ranks <= 1)),
            top_5=float(np.mean(ranks <= 5)),
            fitted_c=tuple(self.fitted_c),
            reliability=tuple(
                ReliabilityBin(
                    lower=b / RELIABILITY_BINS,
                    upper=(b + 1) / RELIABILITY_BINS,
                    count=int(counts[b]),
                    mean_posterior=float(posterior_sums[b] / counts[b])
                    if counts[b]
                    else None,
                    fraction_correct=float(correct_counts[b] / counts[b])
                    if counts[b]
                    else None,
                )
                for b in range(RELIABILITY_BINS)
            ),
            mean_unknown_posterior=float(np.mean(unknown)) if unknown.size else None,
            median_unknown_posterior=float(np.median(unknown))
            if unknown.size
            else None,
            log_loss=float(np.mean(np.concatenate(self.losses))),
        )


def evaluate_calibration(
    features: Sequence[EvalFeature], *, seed: int
) -> CalibrationReport:
    """Cross-validate the baseline and the likelihood ratio on `features`.

    Features without candidates are excluded. The same `features` and `seed`
    give an identical report.

    Raises:
        TooFewMoleculesError: fewer distinct true InChIKeys than folds.
        kde.EmptyStratumError: a fold's training Features leave an Ionization
            mode without correct or without incorrect candidates.
    """
    kept = [feature for feature in features if feature.candidates]
    fold_of = _assign_folds(kept, seed)

    pools = {name: _Pool() for name in _LOG_WEIGHTS}
    for fold in range(FOLDS):
        training = [f for f in kept if fold_of[f.true_inchikey] != fold]
        test = [f for f in kept if fold_of[f.true_inchikey] == fold]
        densities = _fit(training, seed)
        scored_training = _score(training, densities)
        scored_test = _score(test, densities)
        for name, log_weight_of in _LOG_WEIGHTS.items():
            ln_c = _fit_ln_c(_weights(scored_training, log_weight_of(scored_training)))
            pools[name].add_fold(
                scored_test, _weights(scored_test, log_weight_of(scored_test)), ln_c
            )

    n_listed = sum(any(c.correct for c in f.candidates) for f in kept)
    return CalibrationReport(
        n_features=len(kept),
        n_listed=n_listed,
        n_unlisted=len(kept) - n_listed,
        folds=FOLDS,
        baseline=pools["baseline"].report(),
        likelihood_ratio=pools["likelihood_ratio"].report(),
    )


def eval_features(rows: Iterable[GroundTruthAnnotationRow]) -> list[EvalFeature]:
    """The deduplicated Features of `rows`, each with its candidates.

    Keeps the Features `fit-kde` fits on: the ones the (true InChIKey,
    adduct) dedup of `groundtruth.correct_assignment_rows` and
    `groundtruth.incorrect_assignment_rows` keeps. A Feature without
    candidates has no row, so it never appears. Ordered by feature id, and
    each Feature's candidates by InChIKey, whatever `rows`' order.
    """
    rows = list(rows)
    kept_ids = {row.feature_id for row in correct_assignment_rows(rows)} | {
        row.feature_id for row in incorrect_assignment_rows(rows)
    }
    by_feature: dict[int, list[GroundTruthAnnotationRow]] = {}
    for row in sorted(rows, key=lambda r: (r.feature_id, r.candidate_inchikey)):
        if row.feature_id in kept_ids:
            by_feature.setdefault(row.feature_id, []).append(row)
    return [
        EvalFeature(
            true_inchikey=feature_rows[0].true_inchikey,
            ionization_mode=feature_rows[0].ionization_mode,
            candidates=tuple(
                EvalCandidate(row.ion_mass, row.csi_score, is_correct_assignment(row))
                for row in feature_rows
            ),
        )
        for feature_rows in by_feature.values()
    ]

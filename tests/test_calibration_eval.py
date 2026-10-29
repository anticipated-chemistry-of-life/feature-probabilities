"""Tests for the pure held-out calibration evaluation core.

Hand-built Feature sets drawn from known score distributions, run through
`evaluate_calibration` with no DB and no SIRIUS.
"""

from __future__ import annotations

import numpy as np

from feature_probabilities.calibration_eval import (
    EvalCandidate,
    EvalFeature,
    evaluate_calibration,
)


def _feature(
    rng: np.random.Generator,
    *,
    index: int,
    ionization_mode: str,
    correct_scores: list[float],
    incorrect_scores: list[float],
) -> EvalFeature:
    """A Feature at a random ion mass; correct candidates first, then incorrect ones.

    Ion mass is drawn independently of correctness, so only the CSI score
    carries signal.
    """
    ion_mass = float(rng.uniform(200.0, 400.0))
    candidates = [EvalCandidate(ion_mass, score, True) for score in correct_scores]
    candidates += [EvalCandidate(ion_mass, score, False) for score in incorrect_scores]
    return EvalFeature(
        true_inchikey=f"{ionization_mode[0].upper()}{index:013d}",
        ionization_mode=ionization_mode,
        candidates=tuple(candidates),
    )


def _tilted_incorrect_features(n_per_mode: int = 60) -> list[EvalFeature]:
    """Correct scores are broad around 0; incorrect ones sit in a narrow band at its centre.

    The correct density peaks where the incorrect candidates are, so the
    best-row baseline prefers them, while the likelihood ratio does not.
    """
    rng = np.random.default_rng(1)
    features = []
    for mode in ("positive", "negative"):
        for i in range(n_per_mode):
            features.append(
                _feature(
                    rng,
                    index=i,
                    ionization_mode=mode,
                    correct_scores=[float(rng.normal(0.0, 1.0))],
                    incorrect_scores=[float(s) for s in rng.normal(0.0, 0.2, size=3)],
                )
            )
    return features


def test_likelihood_ratio_beats_baseline_top_1_when_incorrect_candidates_crowd_the_correct_peak() -> (
    None
):
    report = evaluate_calibration(_tilted_incorrect_features(), seed=0)

    assert report.likelihood_ratio.top_1 > report.baseline.top_1 + 0.2


def _features_with_unlisted_share(
    unlisted_share: float, n_per_mode: int = 60
) -> list[EvalFeature]:
    """Listed Features as in `_tilted_incorrect_features`; a share of the rest have no correct candidate."""
    rng = np.random.default_rng(2)
    n_unlisted = round(unlisted_share * n_per_mode)
    features = []
    for mode in ("positive", "negative"):
        for i in range(n_per_mode):
            unlisted = i < n_unlisted
            features.append(
                _feature(
                    rng,
                    index=i,
                    ionization_mode=mode,
                    correct_scores=[] if unlisted else [float(rng.normal(0.0, 1.0))],
                    incorrect_scores=[float(s) for s in rng.normal(0.0, 0.2, size=3)],
                )
            )
    return features


def test_fitted_c_rises_as_the_share_of_features_without_a_correct_assignment_grows() -> (
    None
):
    few = evaluate_calibration(_features_with_unlisted_share(0.1), seed=0)
    many = evaluate_calibration(_features_with_unlisted_share(0.5), seed=0)

    for method in ("baseline", "likelihood_ratio"):
        assert np.mean(getattr(many, method).fitted_c) > np.mean(
            getattr(few, method).fitted_c
        )


def test_reliability_bin_counts_sum_to_the_number_of_held_out_candidate_rows() -> None:
    features = _features_with_unlisted_share(0.2)
    n_candidates = sum(len(feature.candidates) for feature in features)

    report = evaluate_calibration(features, seed=0)

    for method in (report.baseline, report.likelihood_ratio):
        assert len(method.reliability) == 10
        assert sum(bin_.count for bin_ in method.reliability) == n_candidates

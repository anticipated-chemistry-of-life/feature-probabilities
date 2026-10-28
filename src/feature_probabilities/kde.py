"""Pure data transform: ground-truth assignment rows -> per-Ionization-mode density pairs.

Turns the deduplicated `CorrectAssignmentRow`s and the
`IncorrectAssignmentRow`s `groundtruth` produces into the fitted
`scipy.stats.gaussian_kde` models `fit-kde` pickles (ADR 0004):

- Stratified by Ionization mode and by nothing else (ADR 0002): each of
  `positive` and `negative` (:data:`IONIZATION_MODES`) gets a
  :class:`ModeDensities`, a correct density (g1) and an incorrect density
  (g0). There is no pooled fallback.
- Each density is a 2D `scipy.stats.gaussian_kde` over (ion mass, CSI:FingerID
  score), using scipy's Scott's-rule default bandwidth with no manual
  rescaling.
- g1 is fitted on every correct row. g0 is fitted on a seeded uniform
  subsample of at most :data:`INCORRECT_SUBSAMPLE_SIZE` incorrect rows per
  mode (all of them when there are fewer): evaluating a KDE costs
  proportional to its training set. The same rows and seed always pick the
  same subsample.

Deliberately does **not** touch DuckDB or SIRIUS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    from collections.abc import Sequence

    from feature_probabilities.groundtruth import (
        CorrectAssignmentRow,
        IncorrectAssignmentRow,
    )

#: The Ionization modes calibration is split by: the exact keys of every
#: `fit-kde` artifact, and the `sirius_runs.ionization_mode` values
#: `annotate` scores against them.
IONIZATION_MODES = ("positive", "negative")


#: Most incorrect rows one mode's incorrect density is fitted on (ADR 0004).
INCORRECT_SUBSAMPLE_SIZE = 20_000

#: Seed of the incorrect-row subsample; stored on the `kde_models` row.
SUBSAMPLE_SEED = 0


class EmptyStratumError(Exception):
    """Raised when an Ionization mode has zero rows of one kind to fit a density on."""


@dataclass(frozen=True, slots=True)
class ModeDensities:
    """One Ionization mode's fitted densities."""

    correct: gaussian_kde
    incorrect: gaussian_kde


def _fit_one(
    rows: Sequence[CorrectAssignmentRow | IncorrectAssignmentRow],
) -> gaussian_kde:
    """Fit a 2D `gaussian_kde` over (ion_mass, csi_score) for `rows`."""
    dataset = np.array([[row.ion_mass, row.csi_score] for row in rows]).T
    return gaussian_kde(dataset)


def _subsample(
    rows: Sequence[IncorrectAssignmentRow], size: int, seed: int
) -> Sequence[IncorrectAssignmentRow]:
    """`rows` itself if at most `size`, else a seeded uniform sample without replacement."""
    if len(rows) <= size:
        return rows
    chosen = np.random.default_rng(seed).choice(len(rows), size=size, replace=False)
    chosen.sort()
    return [rows[i] for i in chosen]


def fit_stratified_kdes(
    correct_rows: Sequence[CorrectAssignmentRow],
    incorrect_rows: Sequence[IncorrectAssignmentRow],
    *,
    seed: int = SUBSAMPLE_SEED,
    subsample_size: int = INCORRECT_SUBSAMPLE_SIZE,
) -> dict[str, ModeDensities]:
    """Fit a correct and an incorrect density per Ionization mode.

    Each density is fitted only on its mode's rows. `incorrect_rows` must be
    in a reproducible order (`groundtruth.incorrect_assignment_rows` sorts
    them), so the same data and `seed` give an identical result.

    Returns a dict with exactly the keys :data:`IONIZATION_MODES`.

    Raises:
        EmptyStratumError: a mode has zero correct rows or zero incorrect
            rows (including when the inputs are empty).
    """
    densities: dict[str, ModeDensities] = {}
    for ionization_mode in IONIZATION_MODES:
        mode_correct = [r for r in correct_rows if r.ionization_mode == ionization_mode]
        if not mode_correct:
            raise EmptyStratumError(
                f"No correct-assignment rows for Ionization mode "
                f"{ionization_mode!r}; cannot fit its KDE."
            )
        mode_incorrect = [
            r for r in incorrect_rows if r.ionization_mode == ionization_mode
        ]
        if not mode_incorrect:
            raise EmptyStratumError(
                f"No incorrect-assignment rows for Ionization mode "
                f"{ionization_mode!r}; incorrect assignments are missing, "
                "so its incorrect density cannot be fitted."
            )
        densities[ionization_mode] = ModeDensities(
            correct=_fit_one(mode_correct),
            incorrect=_fit_one(_subsample(mode_incorrect, subsample_size, seed)),
        )
    return densities

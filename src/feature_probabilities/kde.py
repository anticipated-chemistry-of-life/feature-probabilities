"""Pure data transform: deduplicated correct-assignment rows -> per-Ionization-mode KDEs.

Turns the deduplicated `CorrectAssignmentRow`s `groundtruth.correct_assignment_rows`
produces into the fitted `scipy.stats.gaussian_kde` models `fit-kde` (#22)
pickles, per the KDE calibration method decision (issue #7, amended by #9's
InChIKey/SMILES correction and ADR 0002):

- Stratified by Ionization mode and by nothing else: exactly one KDE each
  for `positive` and `negative` (:data:`IONIZATION_MODES`). Instrument type
  does not split calibration, and there is no pooled fallback KDE: every
  SIRIUS run has an Ionization mode, and both modes always have a KDE.
- Each KDE is a 2D `scipy.stats.gaussian_kde` over (ion mass, CSI:FingerID
  score), using scipy's Scott's-rule default bandwidth with no manual
  rescaling (ion-mass and CSI-score variances are of comparable magnitude in
  the real data, so no rescaling is warranted).

Deliberately does **not** touch DuckDB or SIRIUS, matching the project's
testing decision that this module is exercised with real `scipy` fitting
against fixture rows, not a DB round-trip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import gaussian_kde

if TYPE_CHECKING:
    from collections.abc import Sequence

    from feature_probabilities.groundtruth import CorrectAssignmentRow

#: The Ionization modes calibration is split by: the exact keys of every
#: `fit-kde` artifact, and the `sirius_runs.ionization_mode` values
#: `annotate` scores against them.
IONIZATION_MODES = ("positive", "negative")


class EmptyStratumError(Exception):
    """Raised when an Ionization mode has zero correct-assignment rows to fit on."""


def _fit_one(rows: Sequence[CorrectAssignmentRow]) -> gaussian_kde:
    """Fit a 2D `gaussian_kde` over (ion_mass, csi_score) for `rows`."""
    dataset = np.array([[row.ion_mass, row.csi_score] for row in rows]).T
    return gaussian_kde(dataset)


def fit_stratified_kdes(
    rows: Sequence[CorrectAssignmentRow],
) -> dict[str, gaussian_kde]:
    """Fit one KDE per Ionization mode, each only on that mode's rows.

    Returns a dict with exactly the keys :data:`IONIZATION_MODES`.

    Raises:
        EmptyStratumError: `rows` has zero rows for one of
            :data:`IONIZATION_MODES` (including when `rows` is empty).
    """
    kdes: dict[str, gaussian_kde] = {}
    for ionization_mode in IONIZATION_MODES:
        mode_rows = [row for row in rows if row.ionization_mode == ionization_mode]
        if not mode_rows:
            raise EmptyStratumError(
                f"No correct-assignment rows for Ionization mode "
                f"{ionization_mode!r}; cannot fit its KDE."
            )
        kdes[ionization_mode] = _fit_one(mode_rows)
    return kdes

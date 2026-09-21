"""Pure data transform: deduplicated correct-assignment rows -> per-stratum KDEs.

Turns the deduplicated `CorrectAssignmentRow`s `groundtruth.correct_assignment_rows`
produces into the fitted `scipy.stats.gaussian_kde` models `fit-kde` (#22)
pickles, per the KDE calibration method decision (issue #7, amended by #9's
InChIKey/SMILES correction):

- Stratified by `instrument_type`: one dedicated KDE each for `Orbitrap` and
  `QTOF` (:data:`TRAINED_INSTRUMENT_TYPES`), plus a pooled
  :data:`GLOBAL_STRATUM` KDE fit over every row regardless of instrument
  type -- the fallback `annotate` uses when a feature's instrument type
  matches neither trained stratum.
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

#: Stratum key for the pooled model fit over every instrument type's rows
#: combined -- the fallback `annotate` uses when a feature's instrument type
#: matches neither trained stratum.
GLOBAL_STRATUM = "__global__"

#: `instrument_type` values `fit-kde` trains a dedicated stratum for, per the
#: KDE calibration method decision (issue #7): both instrument types
#: MassSpecGym's ground truth amply covers. Any other `instrument_type`
#: value (unknown/missing metadata, a future third instrument type) still
#: contributes to :data:`GLOBAL_STRATUM` but gets no dedicated stratum.
TRAINED_INSTRUMENT_TYPES = ("Orbitrap", "QTOF")


class EmptyStratumError(Exception):
    """Raised when a stratum this method requires has zero rows to fit on."""


def _fit_one(rows: Sequence[CorrectAssignmentRow]) -> gaussian_kde:
    """Fit a 2D `gaussian_kde` over (ion_mass, csi_score) for `rows`."""
    dataset = np.array([[row.ion_mass, row.csi_score] for row in rows]).T
    return gaussian_kde(dataset)


def fit_stratified_kdes(
    rows: Sequence[CorrectAssignmentRow],
) -> dict[str, gaussian_kde]:
    """Fit one KDE per trained instrument-type stratum plus a pooled fallback.

    Returns a dict with exactly the keys `TRAINED_INSTRUMENT_TYPES` plus
    `GLOBAL_STRATUM`. `GLOBAL_STRATUM` is fit over every row in `rows`
    regardless of `instrument_type`, so it always covers at least the union
    of the trained strata's rows.

    Raises:
        EmptyStratumError: `rows` has zero rows for one of
            `TRAINED_INSTRUMENT_TYPES`, or `rows` itself is empty.
    """
    kdes: dict[str, gaussian_kde] = {}
    for instrument_type in TRAINED_INSTRUMENT_TYPES:
        stratum_rows = [row for row in rows if row.instrument_type == instrument_type]
        if not stratum_rows:
            raise EmptyStratumError(
                f"No correct-assignment rows for instrument type "
                f"{instrument_type!r}; cannot fit its KDE stratum."
            )
        kdes[instrument_type] = _fit_one(stratum_rows)

    kdes[GLOBAL_STRATUM] = _fit_one(rows)
    return kdes

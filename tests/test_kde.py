"""Tests for the pure per-Ionization-mode KDE-fitting transform.

Exercises real `scipy.stats.gaussian_kde` fitting against fixture
`CorrectAssignmentRow`s -- per the project's testing decisions, no fake is
needed since this module touches neither DuckDB nor SIRIUS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scipy.stats import gaussian_kde

from feature_probabilities.groundtruth import CorrectAssignmentRow
from feature_probabilities.kde import EmptyStratumError, fit_stratified_kdes

if TYPE_CHECKING:
    from collections.abc import Callable


def _row(
    feature_id: int,
    *,
    ionization_mode: str,
    ion_mass: float,
    csi_score: float,
    true_inchikey: str,
) -> CorrectAssignmentRow:
    return CorrectAssignmentRow(
        feature_id=feature_id,
        true_inchikey=true_inchikey,
        adduct="[M+H]+" if ionization_mode == "positive" else "[M-H]-",
        ionization_mode=ionization_mode,
        ion_mass=ion_mass,
        csi_score=csi_score,
    )


def _positive_rows() -> list[CorrectAssignmentRow]:
    return [
        _row(
            1,
            ionization_mode="positive",
            ion_mass=301.1,
            csi_score=0.9,
            true_inchikey="AXFAVZQXPFQIEI",
        ),
        _row(
            2,
            ionization_mode="positive",
            ion_mass=250.4,
            csi_score=0.7,
            true_inchikey="BXFAVZQXPFQIEI",
        ),
        _row(
            3,
            ionization_mode="positive",
            ion_mass=410.9,
            csi_score=0.5,
            true_inchikey="CXFAVZQXPFQIEI",
        ),
    ]


def _negative_rows() -> list[CorrectAssignmentRow]:
    return [
        _row(
            4,
            ionization_mode="negative",
            ion_mass=180.2,
            csi_score=0.8,
            true_inchikey="DXFAVZQXPFQIEI",
        ),
        _row(
            5,
            ionization_mode="negative",
            ion_mass=520.6,
            csi_score=0.3,
            true_inchikey="EXFAVZQXPFQIEI",
        ),
        _row(
            6,
            ionization_mode="negative",
            ion_mass=99.9,
            csi_score=0.95,
            true_inchikey="FXFAVZQXPFQIEI",
        ),
    ]


def test_fits_exactly_one_kde_per_ionization_mode() -> None:
    kdes = fit_stratified_kdes(_positive_rows() + _negative_rows())

    assert set(kdes) == {"positive", "negative"}
    for kde in kdes.values():
        assert isinstance(kde, gaussian_kde)


def test_each_kde_is_fit_only_on_its_own_ionization_modes_rows() -> None:
    positive_rows = _positive_rows()
    negative_rows = _negative_rows()

    kdes = fit_stratified_kdes(negative_rows + positive_rows)

    for mode, rows in (("positive", positive_rows), ("negative", negative_rows)):
        fitted_points = {(point[0], point[1]) for point in kdes[mode].dataset.T}
        assert fitted_points == {(row.ion_mass, row.csi_score) for row in rows}


@pytest.mark.parametrize(
    ("present_rows", "missing_mode"),
    [(_positive_rows, "negative"), (_negative_rows, "positive")],
)
def test_a_mode_without_rows_raises_empty_stratum_error_naming_it(
    present_rows: Callable[[], list[CorrectAssignmentRow]], missing_mode: str
) -> None:
    with pytest.raises(EmptyStratumError, match=missing_mode):
        fit_stratified_kdes(present_rows())


def test_empty_input_raises_empty_stratum_error() -> None:
    with pytest.raises(EmptyStratumError):
        fit_stratified_kdes([])

"""Tests for the pure per-Ionization-mode KDE-fitting transform.

Exercises real `scipy.stats.gaussian_kde` fitting against fixture
`CorrectAssignmentRow`s -- per the project's testing decisions, no fake is
needed since this module touches neither DuckDB nor SIRIUS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from scipy.stats import gaussian_kde

from feature_probabilities.groundtruth import (
    CorrectAssignmentRow,
    IncorrectAssignmentRow,
)
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


def _incorrect_row(
    feature_id: int, *, ionization_mode: str, ion_mass: float, csi_score: float
) -> IncorrectAssignmentRow:
    return IncorrectAssignmentRow(
        feature_id=feature_id,
        true_inchikey="AXFAVZQXPFQIEI",
        adduct="[M+H]+" if ionization_mode == "positive" else "[M-H]-",
        ionization_mode=ionization_mode,
        ion_mass=ion_mass,
        csi_score=csi_score,
        truth_listed=True,
    )


def _incorrect_rows(ionization_mode: str, n: int) -> list[IncorrectAssignmentRow]:
    return [
        _incorrect_row(
            i,
            ionization_mode=ionization_mode,
            ion_mass=100.0 + i + (i % 3),
            csi_score=-300.0 - (i % 7) * 5,
        )
        for i in range(n)
    ]


def _all_incorrect_rows() -> list[IncorrectAssignmentRow]:
    return _incorrect_rows("positive", 10) + _incorrect_rows("negative", 10)


def _points(kde: gaussian_kde) -> set[tuple[float, float]]:
    return {(point[0], point[1]) for point in kde.dataset.T}


def test_fits_a_correct_and_an_incorrect_density_per_ionization_mode() -> None:
    densities = fit_stratified_kdes(
        _positive_rows() + _negative_rows(), _all_incorrect_rows()
    )

    assert set(densities) == {"positive", "negative"}
    for mode_densities in densities.values():
        assert isinstance(mode_densities.correct, gaussian_kde)
        assert isinstance(mode_densities.incorrect, gaussian_kde)


def test_each_density_is_fit_only_on_its_own_ionization_modes_rows() -> None:
    positive_rows = _positive_rows()
    negative_rows = _negative_rows()
    incorrect = _all_incorrect_rows()

    densities = fit_stratified_kdes(negative_rows + positive_rows, incorrect)

    for mode, rows in (("positive", positive_rows), ("negative", negative_rows)):
        assert _points(densities[mode].correct) == {
            (row.ion_mass, row.csi_score) for row in rows
        }
        assert _points(densities[mode].incorrect) == {
            (row.ion_mass, row.csi_score)
            for row in incorrect
            if row.ionization_mode == mode
        }


def test_incorrect_density_is_subsampled_to_the_size_but_correct_is_not() -> None:
    incorrect = _incorrect_rows("positive", 50) + _incorrect_rows("negative", 5)

    densities = fit_stratified_kdes(
        _positive_rows() + _negative_rows(), incorrect, subsample_size=10
    )

    assert densities["positive"].incorrect.n == 10
    assert densities["negative"].incorrect.n == 5  # at or below the cap: all rows
    assert densities["positive"].correct.n == 3


def test_same_rows_and_seed_give_the_same_subsample_and_another_seed_differs() -> None:
    correct = _positive_rows() + _negative_rows()
    incorrect = _incorrect_rows("positive", 50) + _incorrect_rows("negative", 5)

    first = fit_stratified_kdes(correct, incorrect, seed=1, subsample_size=10)
    again = fit_stratified_kdes(correct, incorrect, seed=1, subsample_size=10)
    other = fit_stratified_kdes(correct, incorrect, seed=2, subsample_size=10)

    assert _points(first["positive"].incorrect) == _points(again["positive"].incorrect)
    assert _points(first["positive"].incorrect) != _points(other["positive"].incorrect)


@pytest.mark.parametrize(
    ("present_rows", "missing_mode"),
    [(_positive_rows, "negative"), (_negative_rows, "positive")],
)
def test_a_mode_without_correct_rows_raises_empty_stratum_error_naming_it(
    present_rows: Callable[[], list[CorrectAssignmentRow]], missing_mode: str
) -> None:
    with pytest.raises(EmptyStratumError, match=missing_mode):
        fit_stratified_kdes(present_rows(), _all_incorrect_rows())


def test_a_mode_without_incorrect_rows_raises_empty_stratum_error_naming_it() -> None:
    with pytest.raises(EmptyStratumError, match="negative"):
        fit_stratified_kdes(
            _positive_rows() + _negative_rows(), _incorrect_rows("positive", 10)
        )


def test_empty_input_raises_empty_stratum_error() -> None:
    with pytest.raises(EmptyStratumError):
        fit_stratified_kdes([], [])

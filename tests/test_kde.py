"""Tests for the pure stratified KDE-fitting transform.

Exercises real `scipy.stats.gaussian_kde` fitting against fixture
`CorrectAssignmentRow`s -- per the project's testing decisions, no fake is
needed since this module touches neither DuckDB nor SIRIUS.
"""

from __future__ import annotations

import pytest
from scipy.stats import gaussian_kde

from feature_probabilities.groundtruth import CorrectAssignmentRow
from feature_probabilities.kde import (
    GLOBAL_STRATUM,
    TRAINED_INSTRUMENT_TYPES,
    EmptyStratumError,
    fit_stratified_kdes,
)


def _row(
    feature_id: int,
    *,
    instrument_type: str,
    ion_mass: float,
    csi_score: float,
    true_inchikey: str = "AXFAVZQXPFQIEI",
    adduct: str = "[M+H]+",
) -> CorrectAssignmentRow:
    return CorrectAssignmentRow(
        feature_id=feature_id,
        true_inchikey=true_inchikey,
        adduct=adduct,
        instrument_type=instrument_type,
        ion_mass=ion_mass,
        csi_score=csi_score,
    )


def _orbitrap_rows() -> list[CorrectAssignmentRow]:
    return [
        _row(1, instrument_type="Orbitrap", ion_mass=301.1, csi_score=0.9),
        _row(
            2,
            instrument_type="Orbitrap",
            ion_mass=250.4,
            csi_score=0.7,
            true_inchikey="BXFAVZQXPFQIEI",
        ),
        _row(
            3,
            instrument_type="Orbitrap",
            ion_mass=410.9,
            csi_score=0.5,
            true_inchikey="CXFAVZQXPFQIEI",
        ),
    ]


def _qtof_rows() -> list[CorrectAssignmentRow]:
    return [
        _row(
            4,
            instrument_type="QTOF",
            ion_mass=180.2,
            csi_score=0.8,
            true_inchikey="DXFAVZQXPFQIEI",
        ),
        _row(
            5,
            instrument_type="QTOF",
            ion_mass=520.6,
            csi_score=0.3,
            true_inchikey="EXFAVZQXPFQIEI",
        ),
        _row(
            6,
            instrument_type="QTOF",
            ion_mass=99.9,
            csi_score=0.95,
            true_inchikey="FXFAVZQXPFQIEI",
        ),
    ]


def test_fits_exactly_the_trained_strata_plus_the_pooled_global_key() -> None:
    rows = _orbitrap_rows() + _qtof_rows()

    kdes = fit_stratified_kdes(rows)

    assert set(kdes) == {*TRAINED_INSTRUMENT_TYPES, GLOBAL_STRATUM}
    for kde in kdes.values():
        assert isinstance(kde, gaussian_kde)


def test_each_trained_stratum_is_fit_only_on_its_own_instrument_types_rows() -> None:
    orbitrap_rows = _orbitrap_rows()
    qtof_rows = _qtof_rows()

    kdes = fit_stratified_kdes(orbitrap_rows + qtof_rows)

    assert kdes["Orbitrap"].dataset.shape[1] == len(orbitrap_rows)
    assert kdes["QTOF"].dataset.shape[1] == len(qtof_rows)


def test_global_stratum_is_fit_over_the_union_of_both_strata() -> None:
    orbitrap_rows = _orbitrap_rows()
    qtof_rows = _qtof_rows()
    rows = orbitrap_rows + qtof_rows

    kdes = fit_stratified_kdes(rows)

    assert kdes[GLOBAL_STRATUM].dataset.shape[1] == len(rows)
    expected_points = {(row.ion_mass, row.csi_score) for row in rows}
    actual_points = {(point[0], point[1]) for point in kdes[GLOBAL_STRATUM].dataset.T}
    assert actual_points == expected_points


def test_missing_data_for_a_trained_stratum_raises_empty_stratum_error() -> None:
    orbitrap_only = _orbitrap_rows()

    with pytest.raises(EmptyStratumError, match="QTOF"):
        fit_stratified_kdes(orbitrap_only)


def test_empty_input_raises_empty_stratum_error() -> None:
    with pytest.raises(EmptyStratumError):
        fit_stratified_kdes([])

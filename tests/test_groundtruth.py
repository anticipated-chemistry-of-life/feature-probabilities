"""Tests for the pure ground-truth correctness-matching and dedup transform."""

from __future__ import annotations

import random

from feature_probabilities.groundtruth import (
    CorrectAssignmentRow,
    GroundTruthAnnotationRow,
    correct_assignment_rows,
    is_correct_assignment,
)


def _row(
    *,
    feature_id: int,
    candidate_inchikey: str = "ABCDEFGHIJKLMN",
    true_inchikey: str = "ABCDEFGHIJKLMN",
    adduct: str = "[M+H]+",
    instrument_type: str = "Orbitrap",
    ion_mass: float = 300.1,
    csi_score: float = 0.9,
) -> GroundTruthAnnotationRow:
    return GroundTruthAnnotationRow(
        feature_id=feature_id,
        candidate_inchikey=candidate_inchikey,
        true_inchikey=true_inchikey,
        adduct=adduct,
        instrument_type=instrument_type,
        ion_mass=ion_mass,
        csi_score=csi_score,
    )


def test_matching_first_block_inchikeys_are_a_correct_assignment() -> None:
    row = _row(
        feature_id=1,
        candidate_inchikey="ABCDEFGHIJKLMN",
        true_inchikey="ABCDEFGHIJKLMN",
    )

    assert is_correct_assignment(row) is True


def test_differing_inchikeys_are_not_a_correct_assignment() -> None:
    row = _row(
        feature_id=1,
        candidate_inchikey="ABCDEFGHIJKLMN",
        true_inchikey="ZYXWVUTSRQPONM",
    )

    assert is_correct_assignment(row) is False


def test_a_candidate_key_that_is_a_prefix_of_the_true_key_is_not_correct() -> None:
    """No partial/prefix matching -- only plain equality of the two first-block keys."""
    row = _row(
        feature_id=1,
        candidate_inchikey="ABCDEFGHIJKLMN",
        true_inchikey="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    )

    assert is_correct_assignment(row) is False


def test_correct_assignment_rows_excludes_incorrect_rows() -> None:
    rows = [
        _row(
            feature_id=1,
            candidate_inchikey="AAAAAAAAAAAAAA",
            true_inchikey="AAAAAAAAAAAAAA",
        ),
        _row(
            feature_id=2,
            candidate_inchikey="BBBBBBBBBBBBBB",
            true_inchikey="CCCCCCCCCCCCCC",
        ),
    ]

    kept = correct_assignment_rows(rows)

    assert [row.feature_id for row in kept] == [1]


def test_dedup_keeps_the_lowest_feature_id_among_a_shared_group() -> None:
    rows = [
        _row(
            feature_id=30,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
            adduct="[M+H]+",
            instrument_type="Orbitrap",
        ),
        _row(
            feature_id=10,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
            adduct="[M+H]+",
            instrument_type="Orbitrap",
        ),
        _row(
            feature_id=20,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
            adduct="[M+H]+",
            instrument_type="Orbitrap",
        ),
    ]

    kept = correct_assignment_rows(rows)

    assert [row.feature_id for row in kept] == [10]


def test_dedup_key_is_true_inchikey_adduct_and_instrument_type() -> None:
    """A difference in any one of the three key fields keeps both rows distinct."""
    shared = {
        "true_inchikey": "AAAAAAAAAAAAAA",
        "candidate_inchikey": "AAAAAAAAAAAAAA",
    }
    rows = [
        _row(feature_id=1, adduct="[M+H]+", instrument_type="Orbitrap", **shared),
        _row(feature_id=2, adduct="[M+Na]+", instrument_type="Orbitrap", **shared),
        _row(feature_id=3, adduct="[M+H]+", instrument_type="QTOF", **shared),
    ]

    kept = correct_assignment_rows(rows)

    assert {row.feature_id for row in kept} == {1, 2, 3}


def test_dedup_is_independent_of_input_order() -> None:
    rows = [
        _row(
            feature_id=feature_id,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
        )
        for feature_id in (5, 1, 3)
    ]
    shuffled = list(rows)
    random.Random(0).shuffle(shuffled)

    assert correct_assignment_rows(rows) == correct_assignment_rows(shuffled)
    assert [row.feature_id for row in correct_assignment_rows(rows)] == [1]


def test_rerunning_deduplication_on_the_same_input_is_byte_identical() -> None:
    rows = [
        _row(
            feature_id=1,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
        ),
        _row(
            feature_id=2,
            true_inchikey="BBBBBBBBBBBBBB",
            candidate_inchikey="BBBBBBBBBBBBBB",
        ),
        _row(
            feature_id=3,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
        ),
    ]

    first_run = correct_assignment_rows(rows)
    second_run = correct_assignment_rows(rows)

    assert first_run == second_run
    assert [row.feature_id for row in first_run] == [1, 2]


def test_correct_assignment_rows_carries_fields_the_kde_fit_needs() -> None:
    rows = [
        _row(
            feature_id=1,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
            adduct="[M+H]+",
            instrument_type="Orbitrap",
            ion_mass=123.456,
            csi_score=0.75,
        )
    ]

    kept = correct_assignment_rows(rows)

    assert kept == [
        CorrectAssignmentRow(
            feature_id=1,
            true_inchikey="AAAAAAAAAAAAAA",
            adduct="[M+H]+",
            instrument_type="Orbitrap",
            ion_mass=123.456,
            csi_score=0.75,
        )
    ]


def test_groundtruth_module_imports_no_duckdb_or_sirius() -> None:
    import ast
    import inspect

    import feature_probabilities.groundtruth as groundtruth_module

    source = inspect.getsource(groundtruth_module)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert imported_roots.isdisjoint(
        {"duckdb", "sqlalchemy", "pysirius", "PySirius", "scipy"}
    )

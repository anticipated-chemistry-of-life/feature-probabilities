"""Tests for the pure ground-truth correctness-matching and dedup transform."""

from __future__ import annotations

import random

from feature_probabilities.groundtruth import (
    CorrectAssignmentRow,
    GroundTruthAnnotationRow,
    correct_assignment_rows,
    incorrect_assignment_rows,
    is_correct_assignment,
)


def _row(
    *,
    feature_id: int,
    candidate_inchikey: str = "ABCDEFGHIJKLMN",
    true_inchikey: str = "ABCDEFGHIJKLMN",
    adduct: str = "[M+H]+",
    ionization_mode: str = "positive",
    ion_mass: float = 300.1,
    csi_score: float = 0.9,
) -> GroundTruthAnnotationRow:
    return GroundTruthAnnotationRow(
        feature_id=feature_id,
        candidate_inchikey=candidate_inchikey,
        true_inchikey=true_inchikey,
        adduct=adduct,
        ionization_mode=ionization_mode,
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
        ),
        _row(
            feature_id=10,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
            adduct="[M+H]+",
        ),
        _row(
            feature_id=20,
            true_inchikey="AAAAAAAAAAAAAA",
            candidate_inchikey="AAAAAAAAAAAAAA",
            adduct="[M+H]+",
        ),
    ]

    kept = correct_assignment_rows(rows)

    assert [row.feature_id for row in kept] == [10]


def test_dedup_key_is_true_inchikey_and_adduct() -> None:
    """Same true InChIKey and adduct collapse; a different adduct stays distinct."""
    shared = {
        "true_inchikey": "AAAAAAAAAAAAAA",
        "candidate_inchikey": "AAAAAAAAAAAAAA",
    }
    rows = [
        _row(feature_id=1, adduct="[M+H]+", ion_mass=301.0, csi_score=0.9, **shared),
        _row(feature_id=2, adduct="[M+H]+", ion_mass=301.1, csi_score=0.4, **shared),
        _row(feature_id=3, adduct="[M+Na]+", ion_mass=323.0, csi_score=0.8, **shared),
    ]

    kept = correct_assignment_rows(rows)

    assert [row.feature_id for row in kept] == [1, 3]


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
            adduct="[M-H]-",
            ionization_mode="negative",
            ion_mass=123.456,
            csi_score=0.75,
        )
    ]

    kept = correct_assignment_rows(rows)

    assert kept == [
        CorrectAssignmentRow(
            feature_id=1,
            true_inchikey="AAAAAAAAAAAAAA",
            adduct="[M-H]-",
            ionization_mode="negative",
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


def test_incorrect_rows_flag_whether_the_features_truth_is_listed() -> None:
    rows = [
        # Feature 1: truth listed (rank 1), one incorrect candidate.
        _row(
            feature_id=1,
            true_inchikey="TRUEAAAAAAAAAA",
            candidate_inchikey="TRUEAAAAAAAAAA",
        ),
        _row(
            feature_id=1,
            true_inchikey="TRUEAAAAAAAAAA",
            candidate_inchikey="WRONGAAAAAAAAA",
            csi_score=-5.0,
        ),
        # Feature 2: truth ranked below Top-k, so only incorrect candidates.
        _row(
            feature_id=2,
            true_inchikey="TRUEBBBBBBBBBB",
            candidate_inchikey="WRONGBBBBBBBBB",
            csi_score=-7.0,
        ),
    ]

    result = incorrect_assignment_rows(rows)

    assert [(r.feature_id, r.csi_score, r.truth_listed) for r in result] == [
        (1, -5.0, True),
        (2, -7.0, False),
    ]


def test_incorrect_rows_follow_the_dedup_and_prefer_the_listed_feature() -> None:
    same = {"true_inchikey": "TRUEAAAAAAAAAA", "adduct": "[M+H]+"}
    rows = [
        # Feature 1 (lowest id) has no listed truth; feature 2 and 3 do.
        _row(feature_id=1, candidate_inchikey="WRONG1AAAAAAAA", csi_score=-1.0, **same),
        _row(feature_id=2, candidate_inchikey="TRUEAAAAAAAAAA", **same),
        _row(feature_id=2, candidate_inchikey="WRONG2AAAAAAAA", csi_score=-2.0, **same),
        _row(feature_id=3, candidate_inchikey="TRUEAAAAAAAAAA", **same),
        _row(feature_id=3, candidate_inchikey="WRONG3AAAAAAAA", csi_score=-3.0, **same),
        # Same truth under another adduct is its own Feature.
        _row(
            feature_id=4,
            candidate_inchikey="WRONG4AAAAAAAA",
            csi_score=-4.0,
            true_inchikey="TRUEAAAAAAAAAA",
            adduct="[M+Na]+",
        ),
    ]

    result = incorrect_assignment_rows(rows)

    # Feature 2 is the one correct_assignment_rows keeps for the key.
    assert [r.feature_id for r in correct_assignment_rows(rows)] == [2]
    assert [(r.feature_id, r.truth_listed) for r in result] == [(2, True), (4, False)]


def test_incorrect_rows_are_independent_of_input_order() -> None:
    rows = [
        _row(
            feature_id=i,
            true_inchikey=f"TRUE{i:010d}",
            candidate_inchikey=f"WRONG{j}{i:08d}",
            csi_score=-float(j),
        )
        for i in range(1, 6)
        for j in range(3)
    ]
    shuffled = rows[:]
    random.Random(0).shuffle(shuffled)

    assert incorrect_assignment_rows(shuffled) == incorrect_assignment_rows(rows)

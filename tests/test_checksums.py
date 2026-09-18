"""Tests for the pure SIRIUS rerun-avoidance checksum functions."""

from __future__ import annotations

from dataclasses import dataclass

from feature_probabilities.checksums import (
    JSONValue,
    analysis_params_checksum,
    import_params_checksum,
    input_file_checksum,
)


@dataclass
class FakeParams:
    """Minimal ``to_dict``-shaped stand-in for LcmsSubmissionParameters/JobSubmission."""

    payload: dict[str, JSONValue]

    def to_dict(self) -> dict[str, JSONValue]:
        return self.payload


def test_hashing_the_same_bytes_twice_produces_the_same_checksum() -> None:
    data = b"some raw mzml/mgf bytes"

    assert input_file_checksum(data) == input_file_checksum(data)


def test_hashing_different_bytes_produces_a_different_checksum() -> None:
    assert input_file_checksum(b"file one") != input_file_checksum(b"file two")


def test_import_params_checksum_is_independent_of_key_insertion_order() -> None:
    ordered = FakeParams({"allow_ms1_only": False, "profile": "orbitrap"})
    reordered = FakeParams({"profile": "orbitrap", "allow_ms1_only": False})

    assert import_params_checksum(ordered) == import_params_checksum(reordered)


def test_analysis_params_checksum_is_independent_of_key_insertion_order() -> None:
    ordered = FakeParams({"ppm_max": 10.0, "recompute": False})
    reordered = FakeParams({"recompute": False, "ppm_max": 10.0})

    assert analysis_params_checksum(ordered) == analysis_params_checksum(reordered)


def test_import_params_checksum_differs_when_a_single_field_differs() -> None:
    base = FakeParams({"allow_ms1_only": False, "profile": "orbitrap"})
    changed = FakeParams({"allow_ms1_only": True, "profile": "orbitrap"})

    assert import_params_checksum(base) != import_params_checksum(changed)


def test_analysis_params_checksum_differs_when_a_single_field_differs() -> None:
    base = FakeParams({"ppm_max": 10.0, "recompute": False})
    changed = FakeParams({"ppm_max": 15.0, "recompute": False})

    assert analysis_params_checksum(base) != analysis_params_checksum(changed)


def test_import_params_checksum_returns_none_for_the_ground_truth_case() -> None:
    assert import_params_checksum(None) is None


def test_checksums_module_imports_no_duckdb_or_sirius() -> None:
    import ast
    import inspect

    import feature_probabilities.checksums as checksums_module

    source = inspect.getsource(checksums_module)
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

    assert imported_roots.isdisjoint({"duckdb", "sqlalchemy", "pysirius", "PySirius"})

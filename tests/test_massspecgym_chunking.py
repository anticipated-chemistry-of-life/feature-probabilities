"""Tests for MassSpecGym TSV parsing and instrument-type SIRIUS chunking."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pytest

from feature_probabilities.massspecgym import (
    MASSSPECGYM_COLUMNS,
    UNKNOWN_INSTRUMENT_TYPE,
    MassSpecGymParseError,
    chunk_massspecgym_for_sirius,
)


def _row(
    identifier: str,
    *,
    instrument_type: str = "Orbitrap",
    mzs: str = "50.0,100.0,150.0",
    intensities: str = "10.0,20.0,30.0",
    precursor_mz: str = "151.0",
    collision_energy: str = "30",
) -> dict[str, str]:
    return {
        "identifier": identifier,
        "mzs": mzs,
        "intensities": intensities,
        "smiles": "CCO",
        "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
        "formula": "C2H6O",
        "precursor_formula": "C2H7O",
        "parent_mass": "150.0",
        "precursor_mz": precursor_mz,
        "adduct": "[M+H]+",
        "instrument_type": instrument_type,
        "collision_energy": collision_energy,
        "fold": "train",
        "simulation_challenge": "false",
    }


def _write_tsv(
    tmp_path: Path, rows: list[dict[str, str]], columns: tuple[str, ...] = MASSSPECGYM_COLUMNS
) -> Path:
    tsv_path = tmp_path / "MassSpecGym1.5.tsv"
    with tsv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(columns), delimiter="\t", extrasaction="ignore"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return tsv_path


def _mgf_blocks(mgf_path: Path) -> list[str]:
    text = mgf_path.read_text()
    return [block for block in text.split("BEGIN IONS") if block.strip()]


def test_one_chunk_file_written_per_instrument_type_present(tmp_path: Path) -> None:
    tsv_path = _write_tsv(
        tmp_path,
        [
            _row("orbitrap-1", instrument_type="Orbitrap"),
            _row("orbitrap-2", instrument_type="Orbitrap"),
            _row("qtof-1", instrument_type="QTOF"),
        ],
    )
    output_dir = tmp_path / "chunks"

    chunk_paths = chunk_massspecgym_for_sirius(tsv_path, output_dir)

    assert set(chunk_paths) == {"Orbitrap", "QTOF"}
    assert len(chunk_paths["Orbitrap"]) == 1
    assert len(chunk_paths["QTOF"]) == 1
    assert chunk_paths["Orbitrap"][0].exists()
    assert chunk_paths["QTOF"][0].exists()
    assert len(_mgf_blocks(chunk_paths["Orbitrap"][0])) == 2
    assert len(_mgf_blocks(chunk_paths["QTOF"][0])) == 1


def test_group_larger_than_chunk_size_splits_into_multiple_files(tmp_path: Path) -> None:
    rows = [_row(f"orbitrap-{i}") for i in range(5)]
    tsv_path = _write_tsv(tmp_path, rows)
    output_dir = tmp_path / "chunks"

    chunk_paths = chunk_massspecgym_for_sirius(tsv_path, output_dir, chunk_size=2)

    assert len(chunk_paths["Orbitrap"]) == 3
    block_counts = [len(_mgf_blocks(path)) for path in chunk_paths["Orbitrap"]]
    assert block_counts == [2, 2, 1]


def test_every_spectrum_carries_charge_pepmass_and_feature_identifier(tmp_path: Path) -> None:
    tsv_path = _write_tsv(
        tmp_path,
        [_row("msg-42", precursor_mz="151.5")],
    )
    output_dir = tmp_path / "chunks"

    chunk_paths = chunk_massspecgym_for_sirius(tsv_path, output_dir)

    block = _mgf_blocks(chunk_paths["Orbitrap"][0])[0]
    assert re.search(r"^CHARGE=1\+$", block, re.MULTILINE)
    assert re.search(r"^PEPMASS=151\.5$", block, re.MULTILINE)
    assert re.search(r"^FEATURE_ID=msg-42$", block, re.MULTILINE)


def test_formula_and_precursor_formula_are_cleared_before_writing(tmp_path: Path) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1")])
    output_dir = tmp_path / "chunks"

    chunk_paths = chunk_massspecgym_for_sirius(tsv_path, output_dir)

    block = _mgf_blocks(chunk_paths["Orbitrap"][0])[0]
    assert "FORMULA=" not in block
    assert "PRECURSOR_FORMULA=" not in block


def test_present_collision_energy_is_written(tmp_path: Path) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1", collision_energy="35")])
    output_dir = tmp_path / "chunks"

    chunk_paths = chunk_massspecgym_for_sirius(tsv_path, output_dir)

    block = _mgf_blocks(chunk_paths["Orbitrap"][0])[0]
    assert re.search(r"^COLLISION_ENERGY=35$", block, re.MULTILINE)


def test_blank_collision_energy_does_not_raise_and_is_omitted(tmp_path: Path) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1", collision_energy="")])
    output_dir = tmp_path / "chunks"

    chunk_paths = chunk_massspecgym_for_sirius(tsv_path, output_dir)

    block = _mgf_blocks(chunk_paths["Orbitrap"][0])[0]
    assert "COLLISION_ENERGY=" not in block


def test_blank_instrument_type_does_not_raise_and_groups_under_unknown(
    tmp_path: Path,
) -> None:
    tsv_path = _write_tsv(
        tmp_path,
        [
            _row("msg-1", instrument_type=""),
            _row("msg-2", instrument_type="Orbitrap"),
        ],
    )
    output_dir = tmp_path / "chunks"

    chunk_paths = chunk_massspecgym_for_sirius(tsv_path, output_dir)

    assert set(chunk_paths) == {UNKNOWN_INSTRUMENT_TYPE, "Orbitrap"}
    assert len(_mgf_blocks(chunk_paths[UNKNOWN_INSTRUMENT_TYPE][0])) == 1
    block = _mgf_blocks(chunk_paths[UNKNOWN_INSTRUMENT_TYPE][0])[0]
    assert "INSTRUMENT_TYPE=" not in block


def test_missing_identifier_raises_clear_error(tmp_path: Path) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("")])
    output_dir = tmp_path / "chunks"

    with pytest.raises(MassSpecGymParseError, match="identifier"):
        chunk_massspecgym_for_sirius(tsv_path, output_dir)


def test_malformed_mzs_raises_clear_error(tmp_path: Path) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1", mzs="not,numbers")])
    output_dir = tmp_path / "chunks"

    with pytest.raises(MassSpecGymParseError, match="mzs"):
        chunk_massspecgym_for_sirius(tsv_path, output_dir)


def test_mismatched_mzs_and_intensities_lengths_raises_clear_error(tmp_path: Path) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1", mzs="50.0,100.0", intensities="10.0")])
    output_dir = tmp_path / "chunks"

    with pytest.raises(MassSpecGymParseError, match="mzs.*intensities|intensities.*mzs"):
        chunk_massspecgym_for_sirius(tsv_path, output_dir)


def test_missing_required_column_raises_clear_error(tmp_path: Path) -> None:
    columns = tuple(c for c in MASSSPECGYM_COLUMNS if c != "adduct")
    tsv_path = _write_tsv(tmp_path, [_row("msg-1")], columns=columns)
    output_dir = tmp_path / "chunks"

    with pytest.raises(MassSpecGymParseError, match="adduct"):
        chunk_massspecgym_for_sirius(tsv_path, output_dir)


def test_missing_precursor_mz_raises_clear_error(tmp_path: Path) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1", precursor_mz="")])
    output_dir = tmp_path / "chunks"

    with pytest.raises(MassSpecGymParseError, match="precursor_mz"):
        chunk_massspecgym_for_sirius(tsv_path, output_dir)


def test_malformed_precursor_mz_raises_clear_error(tmp_path: Path) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1", precursor_mz="not-a-number")])
    output_dir = tmp_path / "chunks"

    with pytest.raises(MassSpecGymParseError, match="precursor_mz"):
        chunk_massspecgym_for_sirius(tsv_path, output_dir)


def test_blank_inchikey_raises_clear_error(tmp_path: Path) -> None:
    row = _row("msg-1")
    row["inchikey"] = ""
    tsv_path = _write_tsv(tmp_path, [row])
    output_dir = tmp_path / "chunks"

    with pytest.raises(MassSpecGymParseError, match="inchikey"):
        chunk_massspecgym_for_sirius(tsv_path, output_dir)

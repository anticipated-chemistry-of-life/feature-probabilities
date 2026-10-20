"""Tests for the `generate-groundtruth` CLI.

Exercises the CLI end-to-end via `click.testing.CliRunner`, with the MSnLib
parquet fetch monkeypatched to a small local fixture parquet (no live
HuggingFace fetch) and `Sirius` monkeypatched to `FakeSirius` (no real
SIRIUS process).
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pandas as pd
import pytest
from click.testing import CliRunner
from huggingface_hub.errors import RevisionNotFoundError
from PySirius import AlignedFeature, StructureCandidateFormula
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.cli_generate_groundtruth import main
from feature_probabilities.schema import (
    Annotation,
    Feature,
    SiriusRun,
    create_database,
)
from feature_probabilities.sirius import FeatureStructureCandidate
from feature_probabilities.sirius_fake import FakeSirius

if TYPE_CHECKING:
    from collections.abc import Iterator

CONFIG_TOML = """
db_path = "{db_path}"
required_sirius_version = "6.0.0"
ground_truth_revision = "configured-revision"

[sirius]
top_k = 200
"""


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONFIG_TOML.format(db_path=db_path))
    return config_path


@dataclass(slots=True)
class _ContentRecordingFakeSirius(FakeSirius):
    """A `FakeSirius` that also records the MGF text of every imported chunk.

    The CLI deletes its chunk files when it exits, so the only place to see
    what was sent to SIRIUS is at `import_spectra` time.
    """

    imported_mgf: dict[str, str] = field(default_factory=dict, init=False)

    def import_spectra(self, spectra_file: Path, params: object = None) -> None:
        FakeSirius.import_spectra(self, spectra_file, params)
        self.imported_mgf[Path(spectra_file).name] = Path(spectra_file).read_text()

    def imported_feature_ids(self) -> dict[str, list[str]]:
        return {
            name: _mgf_values(text, "FEATURE_ID")
            for name, text in self.imported_mgf.items()
        }


def _mgf_values(mgf_text: str, key: str) -> list[str]:
    """Every `key=` value in `mgf_text`, one per spectrum that carries it."""
    prefix = f"{key}="
    return [
        line.removeprefix(prefix)
        for line in mgf_text.splitlines()
        if line.startswith(prefix)
    ]


_ADDUCT_BY_IONIZATION_MODE = {"positive": "[M+H]+", "negative": "[M-H]-"}
_CHARGE_BY_IONIZATION_MODE = {"positive": 1, "negative": -1}


def _row(
    identifier: str,
    *,
    ionmode: str = "positive",
    inchikey: str = "AAAAAAAAAAAAAA-BBBBBBBBBB-N",
    **overrides: object,
) -> dict[str, object]:
    """One MSnLib parquet row, kept by every filter unless `overrides` says otherwise.

    Carries the columns SIRIUS must never see (`smiles`, `formula`,
    `parent_mass`, `collision_energy`) and MSnLib's own `feature_id`, which
    is not the identifier SIRIUS's features are joined back on.
    """
    row: dict[str, object] = {
        "identifier": identifier,
        "mzs": "150.0,50.0,100.0",
        "intensities": "30.0,10.0,20.0",
        "smiles": "CCO",
        "inchikey": inchikey,
        "formula": "C2H6O",
        "parent_mass": 150.0,
        "precursor_mz": 151.0,
        "adduct": _ADDUCT_BY_IONIZATION_MODE.get(ionmode, "[M+H]+"),
        "instrument_type": "Orbitrap",
        "collision_energy": "20.0,30.0,60.0",
        "charge": _CHARGE_BY_IONIZATION_MODE.get(ionmode, 1),
        "feature_id": f"mzmine feature of {identifier}",
        "ionmode": ionmode,
        "other_matched_compounds": None,
        "spectype": "ALL_ENERGIES",
    }
    row.update(overrides)
    return row


def _write_parquet(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    parquet_path = tmp_path / "msnlib_ms2.parquet"
    pd.DataFrame(rows).to_parquet(parquet_path)
    return parquet_path


def _fake_sirius_for(
    identifier: str,
    inchikey: str = "AAAAAAAAAAAAAA",
    *,
    ionization_mode: str = "positive",
) -> _ContentRecordingFakeSirius:
    """A `FakeSirius` whose default canned result matches `identifier`/`inchikey`.

    Uses `default_features`/`default_results` (not `canned_features`/
    `canned_results`) since the CLI writes its MGF chunks to a fresh temp
    directory each run, whose exact path the test doesn't control.
    """
    adduct = _ADDUCT_BY_IONIZATION_MODE[ionization_mode]
    feature = AlignedFeature(
        aligned_feature_id=identifier,
        external_feature_id=identifier,
        ion_mass=151.0,
        charge=_CHARGE_BY_IONIZATION_MODE[ionization_mode],
        detected_adducts=[adduct],
    )
    candidate = StructureCandidateFormula(
        inchi_key=inchikey,
        smiles="CCO",
        rank=1,
        csi_score=-100.0,
        tanimoto_similarity=0.9,
        mces_dist_to_top_hit=0.0,
        molecular_formula="C2H6O",
        adduct=adduct,
        formula_id="fake-formula-1",
        xlog_p=0.5,
    )
    return _ContentRecordingFakeSirius(
        default_features=[feature],
        default_results=[
            FeatureStructureCandidate(feature=feature, candidate=candidate)
        ],
    )


def _patch_sirius(monkeypatch: pytest.MonkeyPatch, fake: FakeSirius) -> None:
    monkeypatch.setattr(
        "feature_probabilities.cli_generate_groundtruth.Sirius",
        lambda **kwargs: fake,
    )


def _patch_seams(
    monkeypatch: pytest.MonkeyPatch, *, parquet_path: Path, fake: FakeSirius
) -> None:
    monkeypatch.setattr(
        "feature_probabilities.cli_generate_groundtruth.fetch_msnlib_parquet",
        lambda revision: parquet_path,
    )
    _patch_sirius(monkeypatch, fake)


@contextmanager
def _open_db(db_path: Path) -> Iterator[Session]:
    """A real DuckDB `Session` against `db_path`, disposing its engine on exit."""
    engine = create_database(str(db_path))
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


def test_a_positive_spectrum_becomes_a_positive_orbitrap_ground_truth_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = _write_parquet(
        tmp_path, [_row("MSnLibID-pos", inchikey="AAAAAAAAAAAAAA-BBBBBBBBBB-N")]
    )
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)
    db_path = tmp_path / "db" / "database.duckdb"

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, db_path))]
    )

    assert result.exit_code == 0, result.output
    assert fake.imported_feature_ids() == {"orbitrap_positive_0.mgf": ["MSnLibID-pos"]}
    assert [call.formula_id_params.profile.value for call in fake.run_calls] == [
        "ORBITRAP"
    ]
    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert [
            (run.source_kind, run.extract_id, run.ionization_mode, run.instrument_type)
            for run in runs
        ] == [("ground_truth_msnlib", None, "positive", "Orbitrap")]
        features = session.scalars(select(Feature)).all()
        assert [feature.true_inchikey for feature in features] == ["AAAAAAAAAAAAAA"]
        assert len(session.scalars(select(Annotation)).all()) == 1


def test_each_ionization_mode_gets_its_own_chunk_charge_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = _write_parquet(
        tmp_path,
        [
            _row(
                "MSnLibID-neg",
                ionmode="negative",
                inchikey="NNNNNNNNNNNNNN-UHFFFAOYSA-N",
            ),
            _row("MSnLibID-pos", ionmode="positive"),
        ],
    )
    fake = _fake_sirius_for(
        "MSnLibID-neg", "NNNNNNNNNNNNNN", ionization_mode="negative"
    )
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)
    db_path = tmp_path / "db" / "database.duckdb"

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, db_path))]
    )

    assert result.exit_code == 0, result.output
    assert fake.imported_feature_ids() == {
        "orbitrap_negative_0.mgf": ["MSnLibID-neg"],
        "orbitrap_positive_0.mgf": ["MSnLibID-pos"],
    }
    assert {
        name: _mgf_values(text, "CHARGE") for name, text in fake.imported_mgf.items()
    } == {"orbitrap_negative_0.mgf": ["1-"], "orbitrap_positive_0.mgf": ["1+"]}
    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert {
            (Path(run.input_file_path).name, run.ionization_mode, run.instrument_type)
            for run in runs
        } == {
            ("orbitrap_negative_0.mgf", "negative", "Orbitrap"),
            ("orbitrap_positive_0.mgf", "positive", "Orbitrap"),
        }
        negative_run = next(run for run in runs if run.ionization_mode == "negative")
        negative_features = session.scalars(
            select(Feature).where(Feature.run_id == negative_run.run_id)
        ).all()
        assert [feature.true_inchikey for feature in negative_features] == [
            "NNNNNNNNNNNNNN"
        ]


def test_the_mgf_carries_neither_the_answer_nor_the_merged_collision_energies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIRIUS must see only what a measured field spectrum carries.

    Measured against real SIRIUS 6.3.3 on the former ground truth set:
    leaving `SMILES`/`INCHIKEY` in the MGF moved top-1 accuracy from 55% to
    68%, and supplying `ADDUCT` (never known for a field mzML run) changed
    25% of top-1 candidates. An `ALL_ENERGIES` spectrum's collision energy
    is a merged list no single field spectrum carries.
    """
    parquet_path = _write_parquet(tmp_path, [_row("MSnLibID-pos")])
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 0, result.output
    mgf_keys = {
        line.split("=", 1)[0].upper()
        for line in fake.imported_mgf["orbitrap_positive_0.mgf"].splitlines()
        if "=" in line
    }
    assert mgf_keys.isdisjoint(
        {"SMILES", "INCHIKEY", "FORMULA", "PARENT_MASS", "ADDUCT", "COLLISION_ENERGY"}
    )
    assert _mgf_values(fake.imported_mgf["orbitrap_positive_0.mgf"], "FEATURE_ID") == [
        "MSnLibID-pos"
    ]


def test_rows_outside_the_ground_truth_set_never_reach_sirius_and_are_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filters apply in order: spectype, then identity, then adduct.

    A dropped row is never parsed, so a non-`ALL_ENERGIES` row with no
    instrument does not fail the run.
    """
    parquet_path = _write_parquet(
        tmp_path,
        [
            _row("kept"),
            _row("single-scan", spectype="SINGLE_SCAN", instrument_type=""),
            _row("same-energy", spectype="SAME_ENERGY"),
            _row("ambiguous-scan", spectype="SINGLE_SCAN", other_matched_compounds=2),
            _row("ambiguous", other_matched_compounds=1),
            _row("radical-cation", adduct="[M]+"),
            _row("water-loss", adduct="[M-H2O]+"),
            _row("radical-anion", ionmode="negative", adduct="[M]-"),
        ],
    )
    fake = _fake_sirius_for("kept")
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 0, result.output
    assert fake.imported_feature_ids() == {"orbitrap_positive_0.mgf": ["kept"]}
    assert re.search(r"\b3 not ALL_ENERGIES\b", result.output), result.output
    assert re.search(r"\b1 matched to other library compounds\b", result.output)
    assert re.search(r"\b3 with an adduct SIRIUS cannot detect\b", result.output)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mzs": "50.0,not-a-number"},
        {"intensities": ""},
        {"mzs": "50.0,100.0", "intensities": "10.0"},
        {"inchikey": ""},
        {"inchikey": "AAAAAAAAAAAAAA"},
        {"precursor_mz": None},
        {"adduct": " "},
        {"charge": None},
        {"charge": 0},
        {"ionmode": "Positive"},
        {"ionmode": ""},
        {"ionmode": "positive", "charge": -1},
        {"ionmode": "negative", "charge": 1, "adduct": "[M-H]-"},
        {"instrument_type": ""},
    ],
    ids=[
        "malformed-mzs",
        "blank-intensities",
        "mismatched-peaks",
        "blank-inchikey",
        "malformed-inchikey",
        "blank-precursor-mz",
        "blank-adduct",
        "blank-charge",
        "zero-charge",
        "ionmode-not-exact",
        "blank-ionmode",
        "positive-ionmode-negative-charge",
        "negative-ionmode-positive-charge",
        "blank-instrument",
    ],
)
def test_a_malformed_spectrum_fails_the_run_naming_it_before_any_sirius_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, overrides: dict[str, object]
) -> None:
    parquet_path = _write_parquet(
        tmp_path, [_row("MSnLibID-good"), _row("MSnLibID-bad", **overrides)]
    )
    fake = _fake_sirius_for("MSnLibID-good")
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 1, result.output
    assert "MSnLibID-bad" in result.output
    assert "Traceback" not in result.output
    assert fake.run_calls == []


def test_an_unknown_instrument_fails_the_run_naming_the_spectrum_and_the_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = _write_parquet(
        tmp_path,
        [_row("MSnLibID-good"), _row("MSnLibID-astral", instrument_type="Astral")],
    )
    fake = _fake_sirius_for("MSnLibID-good")
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 1, result.output
    assert "MSnLibID-astral" in result.output
    assert "'Astral'" in result.output
    assert fake.run_calls == []


@pytest.mark.parametrize(
    "column",
    [
        "identifier",
        "mzs",
        "intensities",
        "inchikey",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "charge",
        "ionmode",
        "spectype",
        "other_matched_compounds",
    ],
)
def test_a_missing_required_column_fails_the_run_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, column: str
) -> None:
    row = _row("MSnLibID-1")
    del row[column]
    parquet_path = _write_parquet(tmp_path, [row])
    fake = _fake_sirius_for("MSnLibID-1")
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 1, result.output
    assert column in result.output
    assert "Traceback" not in result.output
    assert fake.run_calls == []


def test_a_spectrum_with_no_identifier_fails_the_run_naming_its_parquet_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = _write_parquet(tmp_path, [_row("MSnLibID-good"), _row("")])
    fake = _fake_sirius_for("MSnLibID-good")
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 1, result.output
    assert "row 1" in result.output
    assert "identifier" in result.output
    assert fake.run_calls == []


def _both_modes_parquet(tmp_path: Path) -> Path:
    return _write_parquet(
        tmp_path,
        [
            _row("MSnLibID-pos", ionmode="positive"),
            _row("MSnLibID-neg", ionmode="negative"),
        ],
    )


@pytest.mark.parametrize("ionization_mode", ["positive", "negative"])
def test_the_ionization_mode_flag_processes_only_that_modes_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ionization_mode: str
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)
    db_path = tmp_path / "db.duckdb"

    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(_write_config(tmp_path, db_path)),
            "--ionization-mode",
            ionization_mode,
        ],
    )

    assert result.exit_code == 0, result.output
    assert list(fake.imported_mgf) == [f"orbitrap_{ionization_mode}_0.mgf"]
    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert [run.ionization_mode for run in runs] == [ionization_mode]


def test_the_ionization_mode_flag_all_processes_both_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)

    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(_write_config(tmp_path, tmp_path / "db.duckdb")),
            "--ionization-mode",
            "all",
        ],
    )

    assert result.exit_code == 0, result.output
    assert set(fake.imported_mgf) == {
        "orbitrap_positive_0.mgf",
        "orbitrap_negative_0.mgf",
    }


def test_a_mode_with_more_than_2000_spectra_is_split_into_2000_spectrum_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identifiers = [f"MSnLibID{i:07d}" for i in range(2_001)]
    parquet_path = _write_parquet(tmp_path, [_row(i) for i in identifiers])
    fake = _fake_sirius_for(identifiers[0])
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 0, result.output
    assert fake.imported_feature_ids() == {
        "orbitrap_positive_0.mgf": identifiers[:2_000],
        "orbitrap_positive_1.mgf": identifiers[2_000:],
    }


def test_second_run_with_unchanged_inputs_makes_zero_sirius_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)
    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    first = CliRunner().invoke(main, ["--config", str(config_path)])
    assert first.exit_code == 0, first.output
    assert len(fake.import_spectra_calls) == 2
    assert len(fake.run_calls) == 2

    second = CliRunner().invoke(main, ["--config", str(config_path)])

    assert second.exit_code == 0, second.output
    assert "2 cache hit(s), 0 newly run" in second.output
    assert len(fake.import_spectra_calls) == 2
    assert len(fake.run_calls) == 2
    with _open_db(db_path) as session:
        assert len(session.scalars(select(SiriusRun)).all()) == 2


def test_sirius_version_mismatch_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _ContentRecordingFakeSirius(version="9.9.9")
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code != 0
    assert "9.9.9" in result.output
    assert fake.import_spectra_calls == []
    assert fake.run_calls == []


def test_missing_config_file_fails_fast_with_nonzero_exit(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main, ["--config", str(tmp_path / "does-not-exist.toml")]
    )

    assert result.exit_code != 0
    assert "does-not-exist.toml" in result.output


def test_db_flag_overrides_the_config_files_db_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)
    config_only_db_path = tmp_path / "db" / "unused.duckdb"
    override_db_path = tmp_path / "db" / "overridden.duckdb"

    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(_write_config(tmp_path, config_only_db_path)),
            "--db",
            str(override_db_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert override_db_path.exists()
    assert not config_only_db_path.exists()


def test_force_flag_recomputes_every_chunk_even_when_fully_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(
        monkeypatch,
        parquet_path=_write_parquet(tmp_path, [_row("MSnLibID-pos")]),
        fake=fake,
    )
    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    first = CliRunner().invoke(main, ["--config", str(config_path)])
    assert first.exit_code == 0, first.output
    assert len(fake.run_calls) == 1

    second = CliRunner().invoke(main, ["--config", str(config_path), "--force"])

    assert second.exit_code == 0, second.output
    assert len(fake.import_spectra_calls) == 2
    assert len(fake.run_calls) == 2
    assert fake.run_calls[-1].recompute is True
    with _open_db(db_path) as session:
        assert len(session.scalars(select(SiriusRun)).all()) == 2


def test_one_chunk_failure_is_reported_others_succeed_and_exit_is_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    fake.fail_on_run = {"orbitrap_negative_0.mgf": RuntimeError("sirius blew up")}
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)
    db_path = tmp_path / "db" / "database.duckdb"

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, db_path))]
    )

    assert result.exit_code != 0
    assert "orbitrap_negative_0.mgf" in result.output
    assert "sirius blew up" in result.output
    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert [run.ionization_mode for run in runs] == ["positive"]


def test_rerun_after_partial_failure_only_reprocesses_the_failed_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    fake.fail_on_run = {"orbitrap_negative_0.mgf": RuntimeError("sirius blew up")}
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)
    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    first = CliRunner().invoke(main, ["--config", str(config_path)])
    assert first.exit_code != 0
    assert len(fake.import_spectra_calls) == 2

    fake.fail_on_run = {}
    second = CliRunner().invoke(main, ["--config", str(config_path)])

    assert second.exit_code == 0, second.output
    assert len(fake.import_spectra_calls) == 3
    assert fake.import_spectra_calls[-1][0].name == "orbitrap_negative_0.mgf"
    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert {run.ionization_mode for run in runs} == {"positive", "negative"}


def test_every_chunks_project_is_closed_after_its_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unclosed project stays registered in the SIRIUS instance forever.

    One project per chunk is created under a temporary directory that is
    deleted when the CLI exits; without closing, each run leaves a stale
    entry holding its database open in a long-lived SIRIUS instance.
    """
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 0, result.output
    assert len(fake.create_project_calls) == 2
    assert fake.close_project_calls == 2


def test_every_chunk_runs_on_a_fresh_sirius_instance_even_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIRIUS retains heap per chunk that only a process exit frees.

    The first chunk fails, so a shutdown missing from the failure path would
    leave the second chunk on the first chunk's instance.
    """
    fake = _fake_sirius_for("MSnLibID-pos")
    fake.fail_on_run = {"orbitrap_positive_0.mgf": RuntimeError("sirius blew up")}
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert "orbitrap_positive_0.mgf" in result.output
    assert [path.name for path, _ in fake.create_project_calls] == [
        "orbitrap_positive_0",
        "orbitrap_negative_0",
    ]
    assert fake.instances_started == 2
    assert not fake.running


SMOKE_TEST_TOML = """
[smoke_test]
db_path = "{smoke_db_path}"
kde_output_path = "unused-smoke-model.pkl"
spectra_per_ionization_mode = {spectra_per_ionization_mode}
mzml_dir = "unused-smoke-mzml-dir"
metadata_csv = "unused-smoke-metadata.csv"
export_path = "unused-smoke-export.csv"
"""


def _write_smoke_config(
    tmp_path: Path,
    *,
    db_path: Path,
    smoke_db_path: Path,
    spectra_per_ionization_mode: int = 2,
) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        CONFIG_TOML.format(db_path=db_path)
        + SMOKE_TEST_TOML.format(
            smoke_db_path=smoke_db_path,
            spectra_per_ionization_mode=spectra_per_ionization_mode,
        )
    )
    return config_path


def test_smoke_test_sends_the_first_n_kept_spectra_of_each_mode_to_the_smoke_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet_path = _write_parquet(
        tmp_path,
        [
            _row("pos-scan", spectype="SINGLE_SCAN"),
            _row("pos-1"),
            _row("neg-1", ionmode="negative"),
            _row("neg-ambiguous", ionmode="negative", other_matched_compounds=1),
            _row("pos-2"),
            _row("neg-2", ionmode="negative"),
            _row("pos-3"),
            _row("neg-3", ionmode="negative"),
        ],
    )
    fake = _fake_sirius_for("pos-1")
    _patch_seams(monkeypatch, parquet_path=parquet_path, fake=fake)
    db_path = tmp_path / "db" / "database.duckdb"
    smoke_db_path = tmp_path / "db" / "smoke_test" / "database.duckdb"
    config_path = _write_smoke_config(
        tmp_path, db_path=db_path, smoke_db_path=smoke_db_path
    )

    result = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])

    assert result.exit_code == 0, result.output
    assert fake.imported_feature_ids() == {
        "orbitrap_positive_0.mgf": ["pos-1", "pos-2"],
        "orbitrap_negative_0.mgf": ["neg-1", "neg-2"],
    }
    assert not db_path.exists()
    with _open_db(smoke_db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert {run.ionization_mode for run in runs} == {"positive", "negative"}


def test_smoke_test_starts_from_an_empty_smoke_db_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rerun must exercise SIRIUS and persistence again, not hit the cache."""
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(
        monkeypatch,
        parquet_path=_write_parquet(tmp_path, [_row("MSnLibID-pos")]),
        fake=fake,
    )
    smoke_db_path = tmp_path / "smoke" / "database.duckdb"
    config_path = _write_smoke_config(
        tmp_path, db_path=tmp_path / "database.duckdb", smoke_db_path=smoke_db_path
    )

    first = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])
    assert first.exit_code == 0, first.output
    second = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])

    assert second.exit_code == 0, second.output
    assert len(fake.run_calls) == 2
    with _open_db(smoke_db_path) as session:
        assert len(session.scalars(select(SiriusRun)).all()) == 1


def test_smoke_test_refuses_a_smoke_db_that_is_the_real_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(
        monkeypatch,
        parquet_path=_write_parquet(tmp_path, [_row("MSnLibID-pos")]),
        fake=fake,
    )
    db_path = tmp_path / "database.duckdb"
    real_run = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, db_path))]
    )
    assert real_run.exit_code == 0, real_run.output
    config_path = _write_smoke_config(
        tmp_path, db_path=db_path, smoke_db_path=tmp_path / "." / "database.duckdb"
    )

    result = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])

    assert result.exit_code != 0
    assert "same file" in result.output
    assert len(fake.run_calls) == 1
    with _open_db(db_path) as session:
        assert len(session.scalars(select(SiriusRun)).all()) == 1


@pytest.mark.parametrize(
    "extra_args",
    [["--db", "other.duckdb"], ["--ionization-mode", "all"]],
    ids=["db", "ionization-mode"],
)
def test_smoke_test_rejects_flags_that_would_redirect_or_narrow_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra_args: list[str]
) -> None:
    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_seams(monkeypatch, parquet_path=_both_modes_parquet(tmp_path), fake=fake)
    config_path = _write_smoke_config(
        tmp_path,
        db_path=tmp_path / "database.duckdb",
        smoke_db_path=tmp_path / "smoke.duckdb",
    )

    result = CliRunner().invoke(
        main, ["--config", str(config_path), "--smoke-test", *extra_args]
    )

    assert result.exit_code == 2
    assert extra_args[0] in result.output
    assert fake.run_calls == []


def test_smoke_test_without_a_smoke_test_section_fails_with_a_clear_message(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path, tmp_path / "database.duckdb")

    result = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])

    assert result.exit_code != 0
    assert "[smoke_test]" in result.output


def _patch_hub_download(
    monkeypatch: pytest.MonkeyPatch, download: object, *, fake: FakeSirius
) -> None:
    """Fake only HuggingFace itself, so the CLI's real MSnLib fetch runs."""
    monkeypatch.setattr("feature_probabilities.msnlib.hf_hub_download", download)
    _patch_sirius(monkeypatch, fake)


@pytest.mark.parametrize(
    ("extra_args", "expected_revision"),
    [([], "configured-revision"), (["--ground-truth-revision", "abc123"], "abc123")],
    ids=["configured", "overridden"],
)
def test_msnlib_is_fetched_from_the_hub_at_the_ground_truth_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_args: list[str],
    expected_revision: str,
) -> None:
    parquet_path = _write_parquet(tmp_path, [_row("MSnLibID-pos")])
    requests: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        requests.append(kwargs)
        return str(parquet_path)

    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_hub_download(monkeypatch, download, fake=fake)

    result = CliRunner().invoke(
        main,
        ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb")), *extra_args],
    )

    assert result.exit_code == 0, result.output
    assert requests == [
        {
            "repo_id": "salutmoic/msnlib-ms2-parquet",
            "filename": "msnlib_ms2.parquet",
            "repo_type": "dataset",
            "revision": expected_revision,
        }
    ]
    assert list(fake.imported_mgf) == ["orbitrap_positive_0.mgf"]


@pytest.mark.parametrize(
    "error",
    [
        RevisionNotFoundError(
            "Revision Not Found",
            response=httpx.Response(
                404, request=httpx.Request("GET", "https://huggingface.co/x")
            ),
        ),
        OSError("network unreachable"),
    ],
    ids=["unknown-revision", "network"],
)
def test_a_fetch_failure_fails_the_run_naming_the_dataset_and_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def download(**kwargs: object) -> str:
        raise error

    fake = _fake_sirius_for("MSnLibID-pos")
    _patch_hub_download(monkeypatch, download, fake=fake)

    result = CliRunner().invoke(
        main, ["--config", str(_write_config(tmp_path, tmp_path / "db.duckdb"))]
    )

    assert result.exit_code == 1, result.output
    assert "Traceback" not in result.output
    assert "salutmoic/msnlib-ms2-parquet" in result.output
    assert "configured-revision" in result.output
    assert fake.run_calls == []

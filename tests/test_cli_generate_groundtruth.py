"""Tests for the `generate-groundtruth` happy-path CLI.

Exercises the CLI end-to-end via `click.testing.CliRunner`, with
`fetch_massspecgym_tsv` monkeypatched to a small local fixture TSV (no live
HuggingFace fetch) and `Sirius` monkeypatched to `FakeSirius` (no real
SIRIUS process) -- the two seams issue #9's Testing Decisions call for.
"""

from __future__ import annotations

import csv
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner
from PySirius import AlignedFeature, StructureCandidateFormula
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.cli_generate_groundtruth import main
from feature_probabilities.massspecgym import MASSSPECGYM_COLUMNS
from feature_probabilities.schema import (
    SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
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
massspecgym_revision = "unused-because-fetch-is-monkeypatched"
"""


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONFIG_TOML.format(db_path=db_path))
    return config_path


def _row(
    identifier: str,
    *,
    instrument_type: str = "Orbitrap",
    inchikey: str = "AAAAAAAAAAAAAA",
) -> dict[str, str]:
    return {
        "identifier": identifier,
        "mzs": "50.0,100.0,150.0",
        "intensities": "10.0,20.0,30.0",
        "smiles": "CCO",
        "inchikey": inchikey,
        "formula": "C2H6O",
        "precursor_formula": "C2H7O",
        "parent_mass": "150.0",
        "precursor_mz": "151.0",
        "adduct": "[M+H]+",
        "instrument_type": instrument_type,
        "collision_energy": "30",
        "fold": "train",
        "simulation_challenge": "false",
    }


def _write_tsv(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    tsv_path = tmp_path / "MassSpecGym1.5.tsv"
    with tsv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MASSSPECGYM_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return tsv_path


def _fake_sirius_for(identifier: str, inchikey: str) -> FakeSirius:
    """A `FakeSirius` whose default canned result matches `identifier`/`inchikey`.

    Uses `default_features`/`default_results` (not `canned_features`/
    `canned_results`) since the CLI writes its MGF chunks to a fresh temp
    directory each run, whose exact path the test doesn't control.
    """
    feature = AlignedFeature(
        aligned_feature_id=identifier,
        external_feature_id=identifier,
        ion_mass=151.0,
        charge=1,
        detected_adducts=["[M+H]+"],
    )
    candidate = StructureCandidateFormula(
        inchi_key=inchikey,
        smiles="CCO",
        rank=1,
        csi_score=-100.0,
        tanimoto_similarity=0.9,
        mces_dist_to_top_hit=0.0,
        molecular_formula="C2H6O",
        adduct="[M+H]+",
        formula_id="fake-formula-1",
        xlog_p=0.5,
    )
    return FakeSirius(
        default_features=[feature],
        default_results=[FeatureStructureCandidate(feature=feature, candidate=candidate)],
    )


def _patch_seams(monkeypatch: pytest.MonkeyPatch, *, tsv_path: Path, fake: FakeSirius) -> None:
    monkeypatch.setattr(
        "feature_probabilities.cli_generate_groundtruth.fetch_massspecgym_tsv",
        lambda revision=None: tsv_path,
    )
    monkeypatch.setattr(
        "feature_probabilities.cli_generate_groundtruth.Sirius",
        lambda **kwargs: fake,
    )


@contextmanager
def _open_db(db_path: Path) -> Iterator[Session]:
    """A real DuckDB `Session` against `db_path`, disposing its engine on exit."""
    engine = create_database(str(db_path))
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


def test_full_run_populates_expected_rows_through_the_fake_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1")])
    fake = _fake_sirius_for("msg-1", "AAAAAAAAAAAAAA")
    _patch_seams(monkeypatch, tsv_path=tsv_path, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = CliRunner().invoke(main, ["--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert len(fake.import_spectra_calls) == 1
    assert len(fake.run_calls) == 1

    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 1
        assert runs[0].source_kind == SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM
        assert runs[0].extract_id is None

        features = session.scalars(select(Feature)).all()
        assert len(features) == 1
        assert features[0].true_inchikey == "AAAAAAAAAAAAAA"

        annotations = session.scalars(select(Annotation)).all()
        assert len(annotations) == 1


def test_second_run_with_unchanged_inputs_makes_zero_wrapper_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1")])
    fake = _fake_sirius_for("msg-1", "AAAAAAAAAAAAAA")
    _patch_seams(monkeypatch, tsv_path=tsv_path, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    first = CliRunner().invoke(main, ["--config", str(config_path)])
    assert first.exit_code == 0, first.output
    assert len(fake.import_spectra_calls) == 1
    assert len(fake.run_calls) == 1

    second = CliRunner().invoke(main, ["--config", str(config_path)])

    assert second.exit_code == 0, second.output
    assert len(fake.import_spectra_calls) == 1
    assert len(fake.run_calls) == 1

    with _open_db(db_path) as session:
        assert len(session.scalars(select(SiriusRun)).all()) == 1
        assert len(session.scalars(select(Feature)).all()) == 1


def test_multiple_instrument_types_each_get_their_own_chunk_and_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tsv_path = _write_tsv(
        tmp_path,
        [
            _row("msg-orbitrap", instrument_type="Orbitrap"),
            _row("msg-qtof", instrument_type="QTOF"),
        ],
    )
    fake = _fake_sirius_for("msg-1", "AAAAAAAAAAAAAA")
    _patch_seams(monkeypatch, tsv_path=tsv_path, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = CliRunner().invoke(main, ["--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert len(fake.import_spectra_calls) == 2
    assert len(fake.run_calls) == 2

    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert {run.instrument_type for run in runs} == {"Orbitrap", "QTOF"}


def test_sirius_version_mismatch_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1")])
    fake = FakeSirius(version="9.9.9")
    _patch_seams(monkeypatch, tsv_path=tsv_path, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = CliRunner().invoke(main, ["--config", str(config_path)])

    assert result.exit_code != 0
    assert "9.9.9" in result.output
    assert fake.import_spectra_calls == []
    assert fake.run_calls == []


def test_missing_config_file_fails_fast_with_nonzero_exit(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["--config", str(tmp_path / "does-not-exist.toml")])

    assert result.exit_code != 0
    assert "does-not-exist.toml" in result.output


def test_db_flag_overrides_the_config_files_db_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tsv_path = _write_tsv(tmp_path, [_row("msg-1")])
    fake = _fake_sirius_for("msg-1", "AAAAAAAAAAAAAA")
    _patch_seams(monkeypatch, tsv_path=tsv_path, fake=fake)

    config_only_db_path = tmp_path / "db" / "unused.duckdb"
    config_path = _write_config(tmp_path, config_only_db_path)
    override_db_path = tmp_path / "db" / "overridden.duckdb"

    result = CliRunner().invoke(
        main, ["--config", str(config_path), "--db", str(override_db_path)]
    )

    assert result.exit_code == 0, result.output
    assert override_db_path.exists()
    assert not config_only_db_path.exists()

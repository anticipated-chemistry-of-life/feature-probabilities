"""Tests for the `annotate` happy-path CLI.

Exercises the CLI end-to-end via `click.testing.CliRunner`, with `Sirius`
monkeypatched to `FakeSirius` (no real SIRIUS process) -- the one seam
issue #9's Testing Decisions call for. mzML content is a tiny fixture byte
string (SIRIUS's mzML parsing itself is never exercised, only the
checksum/cache path through it), and the metadata CSV is a small fixture
shaped like a real batch's.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner
from PySirius import AlignedFeature, StructureCandidateFormula
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.cli_annotate import main
from feature_probabilities.schema import (
    SOURCE_KIND_FIELD_MZML,
    Annotation,
    Extract,
    Feature,
    SiriusRun,
    Species,
    create_database,
)
from feature_probabilities.sirius import FeatureStructureCandidate
from feature_probabilities.sirius_fake import FakeSirius

if TYPE_CHECKING:
    from collections.abc import Iterator

CONFIG_TOML = """
db_path = "{db_path}"
required_sirius_version = "6.0.0"
massspecgym_revision = "abc123"
"""

BASE_CSV = """sample_code,taxon_name,ncbi_taxid,family,organ
EX-001,Panthera leo,9689,Felidae,leaf
"""


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(CONFIG_TOML.format(db_path=db_path))
    return config_path


def _write_csv(tmp_path: Path, text: str = BASE_CSV) -> Path:
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(text)
    return csv_path


def _write_mzml_dir(tmp_path: Path, names: list[str]) -> Path:
    mzml_dir = tmp_path / "mzml"
    mzml_dir.mkdir()
    for name in names:
        (mzml_dir / name).write_bytes(f"fake mzml bytes for {name}".encode())
    return mzml_dir


def _fake_sirius_for(inchikey: str = "AAAAAAAAAAAAAA") -> FakeSirius:
    feature = AlignedFeature(
        aligned_feature_id="fake-feature-1",
        external_feature_id="fake-feature-1",
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
        default_results=[
            FeatureStructureCandidate(feature=feature, candidate=candidate)
        ],
    )


def _patch_seams(monkeypatch: pytest.MonkeyPatch, *, fake: FakeSirius) -> None:
    monkeypatch.setattr(
        "feature_probabilities.cli_annotate.Sirius", lambda **kwargs: fake
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


def _invoke(config_path: Path, mzml_dir: Path, csv_path: Path) -> object:
    return CliRunner().invoke(
        main,
        [
            "--config",
            str(config_path),
            "--mzml-dir",
            str(mzml_dir),
            "--metadata-csv",
            str(csv_path),
            "--ionization-mode",
            "positive",
            "--instrument-type",
            "Orbitrap",
        ],
    )


def test_full_run_populates_expected_rows_through_the_fake_wrapper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code == 0, result.output
    assert len(fake.import_spectra_calls) == 1
    assert len(fake.run_calls) == 1

    with _open_db(db_path) as session:
        species = session.scalars(select(Species)).one()
        assert species.taxon_name == "Panthera leo"

        extract = session.scalars(select(Extract)).one()
        assert extract.sample_code == "EX-001"
        assert extract.species_id == species.species_id

        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 1
        assert runs[0].source_kind == SOURCE_KIND_FIELD_MZML
        assert runs[0].extract_id == extract.extract_id
        assert runs[0].import_params_checksum is not None

        assert len(session.scalars(select(Feature)).all()) == 1
        assert len(session.scalars(select(Annotation)).all()) == 1


def test_second_run_with_unchanged_inputs_makes_zero_wrapper_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    first = _invoke(config_path, mzml_dir, csv_path)
    assert first.exit_code == 0, first.output
    assert len(fake.import_spectra_calls) == 1
    assert len(fake.run_calls) == 1

    second = _invoke(config_path, mzml_dir, csv_path)

    assert second.exit_code == 0, second.output
    assert len(fake.import_spectra_calls) == 1
    assert len(fake.run_calls) == 1

    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 1
        assert runs[0].import_params_checksum is not None
        assert len(session.scalars(select(Feature)).all()) == 1
        assert len(session.scalars(select(Extract)).all()) == 1


def test_mzml_file_with_no_matching_metadata_row_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-999.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code != 0
    assert "EX-999.mzML" in result.output
    assert fake.import_spectra_calls == []
    assert fake.run_calls == []


def test_sirius_version_mismatch_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = FakeSirius(version="9.9.9")
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code != 0
    assert "9.9.9" in result.output
    assert fake.import_spectra_calls == []
    assert fake.run_calls == []


def test_missing_config_file_fails_fast_with_nonzero_exit(tmp_path: Path) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)

    result = _invoke(tmp_path / "does-not-exist.toml", mzml_dir, csv_path)

    assert result.exit_code != 0
    assert "does-not-exist.toml" in result.output


def test_missing_mzml_dir_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = _invoke(config_path, tmp_path / "does-not-exist-dir", csv_path)

    assert result.exit_code != 0
    assert "does-not-exist-dir" in result.output


def test_missing_metadata_csv_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = _invoke(config_path, mzml_dir, tmp_path / "does-not-exist.csv")

    assert result.exit_code != 0
    assert "does-not-exist.csv" in result.output


def test_db_flag_overrides_the_config_files_db_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    config_only_db_path = tmp_path / "db" / "unused.duckdb"
    config_path = _write_config(tmp_path, config_only_db_path)
    override_db_path = tmp_path / "db" / "overridden.duckdb"

    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(config_path),
            "--db",
            str(override_db_path),
            "--mzml-dir",
            str(mzml_dir),
            "--metadata-csv",
            str(csv_path),
            "--ionization-mode",
            "positive",
            "--instrument-type",
            "Orbitrap",
        ],
    )

    assert result.exit_code == 0, result.output
    assert override_db_path.exists()
    assert not config_only_db_path.exists()


def test_multiple_mzml_files_each_get_their_own_run_and_extract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML", "EX-002.mzML"])
    csv_path = _write_csv(
        tmp_path,
        "sample_code,taxon_name,ncbi_taxid,family,organ\n"
        "EX-001,Panthera leo,9689,Felidae,leaf\n"
        "EX-002,Panthera tigris,9694,Felidae,root\n",
    )
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code == 0, result.output
    assert len(fake.import_spectra_calls) == 2

    with _open_db(db_path) as session:
        extracts = session.scalars(select(Extract)).all()
        assert {extract.sample_code for extract in extracts} == {"EX-001", "EX-002"}

        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 2
        assert {run.extract_id for run in runs} == {
            extract.extract_id for extract in extracts
        }

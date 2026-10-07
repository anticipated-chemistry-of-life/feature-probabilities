"""Tests for the `annotate` CLI: batch processing, calibration, and export.

Exercises the CLI end-to-end via `click.testing.CliRunner`, with `Sirius`
monkeypatched to `FakeSirius` (no real SIRIUS process) -- the one seam
issue #9's Testing Decisions call for. mzML content is a tiny fixture byte
string (SIRIUS's mzML parsing itself is never exercised, only the
checksum/cache path through it), and the metadata CSV is a small fixture
shaped like a real batch's. `kde_models` rows point at real, small
`scipy.stats.gaussian_kde` pickles (no KDE mocking), per the project's
testing decisions.
"""

from __future__ import annotations

import pickle
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from click.testing import CliRunner
from PySirius import AlignedFeature, DataQuality, StructureCandidateFormula
from scipy.stats import gaussian_kde
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.cli_annotate import main
from feature_probabilities.kde import GLOBAL_STRATUM
from feature_probabilities.schema import (
    SOURCE_KIND_FIELD_MZML,
    Annotation,
    CalibrationScore,
    Extract,
    Feature,
    KdeModel,
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

[sirius]
top_k = 200
"""

BASE_CSV = """sample_id,filename,taxon_scientific_name,ncbi_taxid,family,organ
EX-001,EX-001.mzML,Panthera leo,9689,Felidae,leaf
"""

TWO_EXTRACT_CSV = """sample_id,filename,taxon_scientific_name,ncbi_taxid,family,organ
EX-001,EX-001.mzML,Panthera leo,9689,Felidae,leaf
EX-002,EX-002.mzML,Panthera tigris,9694,Felidae,root
"""

#: (ion_mass, csi_score) points bracketing `_fake_sirius_for`'s default
#: candidate (151.0, -100.0), so every stratum's KDE can score it.
_KDE_POINTS = [(150.0, -100.0), (151.0, -99.0), (149.5, -101.0), (152.0, -98.5)]

#: A distinctly different point cloud, so a KDE fit on it disagrees with
#: `_KDE_POINTS`'s -- lets tests tell "which model actually got applied"
#: apart by the resulting `calibration_scores.score` alone.
_OTHER_KDE_POINTS = [(151.0, -100.0), (151.2, -99.8), (150.8, -100.2), (151.1, -99.9)]


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


def _aligned_feature(
    feature_id: str, quality: DataQuality = DataQuality.GOOD
) -> AlignedFeature:
    return AlignedFeature(
        aligned_feature_id=feature_id,
        external_feature_id=feature_id,
        ion_mass=151.0,
        charge=1,
        detected_adducts=["[M+H]+"],
        quality=quality,
    )


def _candidate(inchikey: str) -> StructureCandidateFormula:
    return StructureCandidateFormula(
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


def _fake_sirius_for(inchikey: str = "AAAAAAAAAAAAAA") -> FakeSirius:
    feature = _aligned_feature("fake-feature-1")
    candidate = _candidate(inchikey)
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


def _kde_dict(points: list[tuple[float, float]]) -> dict[str, gaussian_kde]:
    fitted = gaussian_kde(np.array(points).T)
    return {"Orbitrap": fitted, "QTOF": fitted, GLOBAL_STRATUM: fitted}


def _seed_kde_model(
    db_path: Path,
    artifact_path: Path,
    *,
    points: list[tuple[float, float]] | None = None,
) -> int:
    """Pickle a real, fitted KDE dict to `artifact_path` and insert its `kde_models` row.

    Creates `db_path` if it doesn't exist yet (mirroring `main`'s own
    idempotent `create_database` call). Returns the new row's
    `kde_model_id`.
    """
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("wb") as handle:
        pickle.dump(_kde_dict(points if points is not None else _KDE_POINTS), handle)

    engine = create_database(str(db_path))
    try:
        with Session(engine) as session:
            kde_model = KdeModel(artifact_path=str(artifact_path))
            session.add(kde_model)
            session.commit()
            return kde_model.kde_model_id
    finally:
        engine.dispose()


def _invoke(
    config_path: Path, mzml_dir: Path, csv_path: Path, *extra_args: str
) -> object:
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
            *extra_args,
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
    kde_model_id = _seed_kde_model(db_path, tmp_path / "kde.pkl")

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code == 0, result.output
    assert len(fake.import_spectra_calls) == 1
    assert len(fake.run_calls) == 1

    with _open_db(db_path) as session:
        species = session.scalars(select(Species)).one()
        assert species.taxon_scientific_name == "Panthera leo"

        extract = session.scalars(select(Extract)).one()
        assert extract.sample_id == "EX-001"
        assert extract.species_id == species.species_id

        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 1
        assert runs[0].source_kind == SOURCE_KIND_FIELD_MZML
        assert runs[0].extract_id == extract.extract_id
        assert runs[0].import_params_checksum is not None

        assert len(session.scalars(select(Feature)).all()) == 1
        annotation = session.scalars(select(Annotation)).one()

        score = session.get(CalibrationScore, (annotation.annotation_id, kde_model_id))
        assert score is not None


def test_second_run_with_unchanged_inputs_makes_zero_wrapper_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

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
        # Re-scoring an unchanged, cache-hit batch against the same
        # kde_model_id updates the existing row in place rather than
        # raising a primary-key conflict.
        assert len(session.scalars(select(CalibrationScore)).all()) == 1


def test_only_good_and_decent_features_are_analysed_and_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    features = [
        _aligned_feature(quality.value, quality)
        for quality in (
            DataQuality.GOOD,
            DataQuality.DECENT,
            DataQuality.BAD,
            DataQuality.LOWEST,
        )
    ]
    fake = FakeSirius(
        default_features=features,
        default_results=[
            FeatureStructureCandidate(feature=feature, candidate=_candidate(key))
            for feature, key in zip(
                features,
                [
                    "AAAAAAAAAAAAAA",
                    "BBBBBBBBBBBBBB",
                    "CCCCCCCCCCCCCC",
                    "DDDDDDDDDDDDDD",
                ],
                strict=True,
            )
        ],
    )
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code == 0, result.output
    assert len(fake.run_calls) == 1
    assert sorted(fake.run_calls[0].aligned_feature_ids) == ["DECENT", "GOOD"]

    with _open_db(db_path) as session:
        persisted = session.scalars(select(Feature)).all()
        assert sorted(feature.quality for feature in persisted) == ["DECENT", "GOOD"]
        assert len(session.scalars(select(Annotation)).all()) == 2


def test_mzml_file_with_no_matching_metadata_row_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-999.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

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
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

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
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

    result = _invoke(config_path, mzml_dir, tmp_path / "does-not-exist.csv")

    assert result.exit_code != 0
    assert "does-not-exist.csv" in result.output


def test_missing_kde_model_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    # No kde_models row seeded at all.

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code != 0
    assert "kde_models" in result.output
    assert fake.import_spectra_calls == []


def test_kde_model_flag_with_no_matching_row_fails_fast_with_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")
    bogus_path = str(tmp_path / "does-not-exist.pkl")

    result = _invoke(config_path, mzml_dir, csv_path, "--kde-model", bogus_path)

    assert result.exit_code != 0
    assert bogus_path in result.output
    assert fake.import_spectra_calls == []


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
    _seed_kde_model(override_db_path, tmp_path / "kde.pkl")

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
    csv_path = _write_csv(tmp_path, TWO_EXTRACT_CSV)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code == 0, result.output
    assert len(fake.import_spectra_calls) == 2

    with _open_db(db_path) as session:
        extracts = session.scalars(select(Extract)).all()
        assert {extract.sample_id for extract in extracts} == {"EX-001", "EX-002"}

        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 2
        assert {run.extract_id for run in runs} == {
            extract.extract_id for extract in extracts
        }


def test_omitting_kde_model_defaults_to_most_recently_fitted_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde-v1.pkl", points=_OTHER_KDE_POINTS)
    kde_model_2_id = _seed_kde_model(db_path, tmp_path / "kde-v2.pkl")

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code == 0, result.output
    with _open_db(db_path) as session:
        annotation = session.scalars(select(Annotation)).one()
        scores = session.scalars(select(CalibrationScore)).all()
        assert [row.kde_model_id for row in scores] == [kde_model_2_id]
        assert scores[0].annotation_id == annotation.annotation_id


def test_explicit_kde_model_flag_applies_that_pickle_instead_of_the_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    kde_model_1_path = tmp_path / "kde-v1.pkl"
    kde_model_1_id = _seed_kde_model(
        db_path, kde_model_1_path, points=_OTHER_KDE_POINTS
    )
    _seed_kde_model(db_path, tmp_path / "kde-v2.pkl")

    result = _invoke(
        config_path, mzml_dir, csv_path, "--kde-model", str(kde_model_1_path)
    )

    assert result.exit_code == 0, result.output
    with _open_db(db_path) as session:
        scores = session.scalars(select(CalibrationScore)).all()
        assert [row.kde_model_id for row in scores] == [kde_model_1_id]


def test_repointing_kde_model_on_cached_batch_adds_scores_without_new_sirius_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    kde_model_1_id = _seed_kde_model(db_path, tmp_path / "kde-v1.pkl")

    first = _invoke(config_path, mzml_dir, csv_path)
    assert first.exit_code == 0, first.output
    assert len(fake.run_calls) == 1

    kde_model_2_path = tmp_path / "kde-v2.pkl"
    kde_model_2_id = _seed_kde_model(
        db_path, kde_model_2_path, points=_OTHER_KDE_POINTS
    )

    second = _invoke(
        config_path, mzml_dir, csv_path, "--kde-model", str(kde_model_2_path)
    )

    assert second.exit_code == 0, second.output
    # Purely a cache hit -- re-pointing --kde-model never touches SIRIUS.
    assert len(fake.import_spectra_calls) == 1
    assert len(fake.run_calls) == 1

    with _open_db(db_path) as session:
        annotation = session.scalars(select(Annotation)).one()
        scores = {
            row.kde_model_id: row
            for row in session.scalars(select(CalibrationScore)).all()
        }
        assert set(scores) == {kde_model_1_id, kde_model_2_id}
        assert all(
            row.annotation_id == annotation.annotation_id for row in scores.values()
        )


def test_force_flag_calls_wrapper_for_every_file_even_when_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

    first = _invoke(config_path, mzml_dir, csv_path)
    assert first.exit_code == 0, first.output
    assert len(fake.run_calls) == 1

    second = _invoke(config_path, mzml_dir, csv_path, "--force")

    assert second.exit_code == 0, second.output
    assert len(fake.run_calls) == 2
    assert fake.run_calls[-1].recompute is True

    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 2


def test_one_file_failure_is_reported_others_succeed_and_exit_is_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML", "EX-002.mzML"])
    csv_path = _write_csv(tmp_path, TWO_EXTRACT_CSV)
    fake = _fake_sirius_for()
    fake.fail_on_run = {"EX-002.mzML": RuntimeError("sirius blew up")}
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

    result = _invoke(config_path, mzml_dir, csv_path)

    assert result.exit_code != 0
    assert "EX-002.mzML" in result.output
    assert "sirius blew up" in result.output

    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 1
        extract = session.scalars(
            select(Extract).where(Extract.sample_id == "EX-001")
        ).one()
        assert runs[0].extract_id == extract.extract_id
        assert len(session.scalars(select(CalibrationScore)).all()) == 1


def test_every_file_runs_on_a_fresh_sirius_instance_even_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SIRIUS retains heap per analysed feature that only a process exit frees.

    The first file fails, so a shutdown missing from the failure path would
    leave the second file on the first file's instance.
    """
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML", "EX-002.mzML"])
    csv_path = _write_csv(tmp_path, TWO_EXTRACT_CSV)
    fake = _fake_sirius_for()
    fake.fail_on_run = {"EX-001.mzML": RuntimeError("sirius blew up")}
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")

    result = _invoke(config_path, mzml_dir, csv_path)

    assert "EX-001.mzML" in result.output
    assert [path.name for path, _ in fake.create_project_calls] == [
        "EX-001",
        "EX-002",
    ]
    assert fake.instances_started == 2
    assert not fake.running


def test_export_writes_one_csv_row_per_annotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")
    export_path = tmp_path / "feature_probabilities.csv"

    result = _invoke(config_path, mzml_dir, csv_path, "--export", str(export_path))

    assert result.exit_code == 0, result.output
    assert export_path.exists()

    df = pd.read_csv(export_path)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["sample_id"] == "EX-001"
    assert row["external_feature_id"] == "fake-feature-1"
    assert row["inchikey"] == "AAAAAAAAAAAAAA"
    assert row["csi_score"] == pytest.approx(-100.0)
    assert np.isfinite(row["calibration_score"])


def test_export_writes_parquet_when_given_a_parquet_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")
    export_path = tmp_path / "feature_probabilities.parquet"

    result = _invoke(config_path, mzml_dir, csv_path, "--export", str(export_path))

    assert result.exit_code == 0, result.output
    df = pd.read_parquet(export_path)
    assert len(df) == 1
    assert df.iloc[0]["sample_id"] == "EX-001"


def test_export_rejects_an_unsupported_file_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML"])
    csv_path = _write_csv(tmp_path)
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")
    export_path = tmp_path / "feature_probabilities.json"

    result = _invoke(config_path, mzml_dir, csv_path, "--export", str(export_path))

    assert result.exit_code != 0
    assert ".json" in result.output
    assert not export_path.exists()
    assert fake.import_spectra_calls == []


def test_export_only_includes_successfully_processed_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mzml_dir = _write_mzml_dir(tmp_path, ["EX-001.mzML", "EX-002.mzML"])
    csv_path = _write_csv(tmp_path, TWO_EXTRACT_CSV)
    fake = _fake_sirius_for()
    fake.fail_on_run = {"EX-002.mzML": RuntimeError("sirius blew up")}
    _patch_seams(monkeypatch, fake=fake)

    db_path = tmp_path / "db" / "database.duckdb"
    config_path = _write_config(tmp_path, db_path)
    _seed_kde_model(db_path, tmp_path / "kde.pkl")
    export_path = tmp_path / "feature_probabilities.csv"

    result = _invoke(config_path, mzml_dir, csv_path, "--export", str(export_path))

    assert result.exit_code != 0
    df = pd.read_csv(export_path)
    assert list(df["sample_id"]) == ["EX-001"]


SMOKE_TEST_TOML = """
[smoke_test]
db_path = "{smoke_db_path}"
kde_output_path = "unused-smoke-model.pkl"
spectra_per_instrument_type = 2
mzml_dir = "{mzml_dir}"
metadata_csv = "{metadata_csv}"
ionization_mode = "negative"
instrument_type = "QTOF"
export_path = "{export_path}"
"""


def _write_smoke_setup(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """A config whose `[smoke_test]` points at its own mzML dir, CSV, DB, and export.

    Returns `(config_path, db_path, smoke_db_path, smoke_export_path)`.
    """
    smoke_dir = tmp_path / "smoke"
    smoke_dir.mkdir()
    mzml_dir = _write_mzml_dir(smoke_dir, ["EX-001.mzML"])
    csv_path = _write_csv(smoke_dir)
    db_path = tmp_path / "database.duckdb"
    smoke_db_path = smoke_dir / "database.duckdb"
    smoke_export_path = smoke_dir / "feature_probabilities.csv"
    config_path = _write_config(tmp_path, db_path)
    with config_path.open("a") as handle:
        handle.write(
            SMOKE_TEST_TOML.format(
                smoke_db_path=smoke_db_path,
                mzml_dir=mzml_dir,
                metadata_csv=csv_path,
                export_path=smoke_export_path,
            )
        )
    return config_path, db_path, smoke_db_path, smoke_export_path


def test_smoke_test_runs_the_configured_input_into_the_smoke_db_and_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)
    config_path, db_path, smoke_db_path, smoke_export_path = _write_smoke_setup(
        tmp_path
    )
    _seed_kde_model(smoke_db_path, tmp_path / "smoke-kde.pkl")

    result = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])

    assert result.exit_code == 0, result.output
    assert not db_path.exists()
    assert list(pd.read_csv(smoke_export_path)["sample_id"]) == ["EX-001"]
    with _open_db(smoke_db_path) as session:
        [run] = session.scalars(select(SiriusRun)).all()
        assert (run.ionization_mode, run.instrument_type) == ("negative", "QTOF")


def test_smoke_test_explicit_flags_override_its_configured_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _fake_sirius_for()
    _patch_seams(monkeypatch, fake=fake)
    config_path, _, smoke_db_path, smoke_export_path = _write_smoke_setup(tmp_path)
    _seed_kde_model(smoke_db_path, tmp_path / "smoke-kde.pkl")
    other_mzml_dir = _write_mzml_dir(tmp_path, ["EX-002.mzML"])
    other_csv_path = _write_csv(tmp_path, TWO_EXTRACT_CSV)
    other_export_path = tmp_path / "other.csv"

    result = _invoke(
        config_path,
        other_mzml_dir,
        other_csv_path,
        "--smoke-test",
        "--export",
        str(other_export_path),
    )

    assert result.exit_code == 0, result.output
    assert not smoke_export_path.exists()
    assert list(pd.read_csv(other_export_path)["sample_id"]) == ["EX-002"]
    with _open_db(smoke_db_path) as session:
        [run] = session.scalars(select(SiriusRun)).all()
        assert (run.ionization_mode, run.instrument_type) == ("positive", "Orbitrap")


def test_smoke_test_rejects_the_db_flag(tmp_path: Path) -> None:
    config_path, *_ = _write_smoke_setup(tmp_path)

    result = CliRunner().invoke(
        main,
        ["--config", str(config_path), "--smoke-test", "--db", "other.duckdb"],
    )

    assert result.exit_code == 2
    assert "--db" in result.output


def test_smoke_test_without_a_kde_model_points_at_the_smoke_fit_kde(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_seams(monkeypatch, fake=_fake_sirius_for())
    config_path, *_ = _write_smoke_setup(tmp_path)

    result = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])

    assert result.exit_code != 0
    assert "fit-kde --smoke-test" in result.output


def test_without_smoke_test_a_missing_input_flag_is_reported_before_the_config(
    tmp_path: Path,
) -> None:
    result = CliRunner().invoke(
        main, ["--config", str(tmp_path / "does-not-exist.toml")]
    )

    assert result.exit_code == 2
    assert "--mzml-dir" in result.output

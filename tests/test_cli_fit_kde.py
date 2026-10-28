"""Tests for the `fit-kde` CLI.

Exercises the CLI end-to-end via `click.testing.CliRunner` against a real
temp DuckDB (no fake needed -- `fit-kde` never talks to SIRIUS), with fixture
ground-truth rows inserted directly through the real SQLAlchemy models
rather than depending on `generate-groundtruth`'s actual code having run,
per issue #22's acceptance criteria and issue #9's testing decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner, Result
from scipy.stats import gaussian_kde
from sqlalchemy import insert as sa_insert
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.calibrate import load_kdes
from feature_probabilities.cli_fit_kde import main
from feature_probabilities.schema import (
    SOURCE_KIND_FIELD_MZML,
    SOURCE_KIND_GROUND_TRUTH_MSNLIB,
    Annotation,
    Feature,
    KdeModel,
    Molecule,
    SiriusRun,
    create_database,
)

if TYPE_CHECKING:
    from collections.abc import Callable

CONFIG_TOML = """
db_path = "{db_path}"
required_sirius_version = "6.0.0"
ground_truth_revision = "abc123"
kde_output_path = "{kde_output_path}"

[sirius]
top_k = 200
"""

#: Non-collinear (ion_mass, csi_score) points per Ionization mode --
#: gaussian_kde needs a non-singular 2D covariance, which a perfectly linear
#: point set lacks.
_POINTS_BY_MODE = {
    "positive": [(200.0, 0.9), (210.0, 0.5), (225.0, 0.7)],
    "negative": [(300.0, 0.6), (310.0, 0.3), (330.0, 0.55)],
}

#: Non-collinear incorrect-candidate points per Ionization mode: one incorrect
#: candidate per correct Feature, at that Feature's ion mass.
_INCORRECT_POINTS_BY_MODE = {
    "positive": [(200.0, 0.1), (210.0, -0.4), (225.0, -0.2)],
    "negative": [(300.0, -0.5), (310.0, -0.9), (330.0, -0.6)],
}

#: The adduct SIRIUS detects for a Feature of each Ionization mode.
_ADDUCT_BY_MODE = {"positive": "[M+H]+", "negative": "[M-H]-"}


def _write_config(tmp_path: Path, db_path: Path, kde_output_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        CONFIG_TOML.format(db_path=db_path, kde_output_path=kde_output_path)
    )
    return config_path


def _insert_run(
    session: Session,
    *,
    ionization_mode: str,
    run_suffix: str,
    instrument_type: str = "Orbitrap",
    source_kind: str = SOURCE_KIND_GROUND_TRUTH_MSNLIB,
) -> SiriusRun:
    run = SiriusRun(
        extract_id=None,
        source_kind=source_kind,
        input_file_checksum=f"checksum-{run_suffix}",
        import_params_checksum=None,
        analysis_params_checksum=f"analysis-{run_suffix}",
        sirius_version="6.0.0",
        pysirius_client_version="1.0.0",
        ionization_mode=ionization_mode,
        instrument_type=instrument_type,
        input_file_path=f"/data/{run_suffix}.mgf",
    )
    session.add(run)
    session.flush()
    return run


def _insert_feature(
    session: Session,
    *,
    run: SiriusRun,
    feature_suffix: str,
    true_inchikey: str,
    ion_mass: float,
    adduct: str,
) -> Feature:
    feature = Feature(
        run_id=run.run_id,
        external_feature_id=f"feature-{feature_suffix}",
        ion_mass=ion_mass,
        charge=1 if adduct.endswith("+") else -1,
        true_inchikey=true_inchikey,
    )
    session.add(feature)
    session.flush()
    return feature


def _insert_candidate(
    session: Session,
    *,
    feature: Feature,
    candidate_inchikey: str,
    csi_score: float,
    adduct: str,
    rank: int = 1,
) -> None:
    """Insert one annotation of `feature` for `candidate_inchikey`.

    Reuses the `molecules` row for `candidate_inchikey` if one exists (its
    InChIKey is unique), so several features may share a candidate.
    """
    molecule = session.scalars(
        select(Molecule).where(Molecule.inchikey == candidate_inchikey)
    ).one_or_none()
    if molecule is None:
        molecule = Molecule(
            inchikey=candidate_inchikey, smiles="CCO", molecular_formula="C2H6O"
        )
        session.add(molecule)
        session.flush()
    session.add(
        Annotation(
            feature_id=feature.feature_id,
            molecule_id=molecule.molecule_id,
            rank=rank,
            csi_score=csi_score,
            tanimoto_similarity=1.0,
            mces_dist_to_top_hit=0.0,
            xlogp=1.0,
            adduct=adduct,
            formula_id="formula-1",
        )
    )
    session.flush()


def _insert_correct_assignment(
    session: Session,
    *,
    run: SiriusRun,
    feature_suffix: str,
    inchikey: str,
    ion_mass: float,
    csi_score: float,
    adduct: str,
) -> Feature:
    """Insert one feature + annotation whose candidate matches its true InChIKey."""
    feature = _insert_feature(
        session,
        run=run,
        feature_suffix=feature_suffix,
        true_inchikey=inchikey,
        ion_mass=ion_mass,
        adduct=adduct,
    )
    _insert_candidate(
        session,
        feature=feature,
        candidate_inchikey=inchikey,
        csi_score=csi_score,
        adduct=adduct,
    )
    return feature


def _insert_incorrect_candidate(
    session: Session,
    *,
    feature: Feature,
    candidate_inchikey: str,
    csi_score: float,
    adduct: str,
    rank: int = 2,
) -> None:
    assert candidate_inchikey != feature.true_inchikey
    _insert_candidate(
        session,
        feature=feature,
        candidate_inchikey=candidate_inchikey,
        csi_score=csi_score,
        adduct=adduct,
        rank=rank,
    )


def _inchikey(ionization_mode: str, index: int) -> str:
    """A distinct 14-character first-block InChIKey per mode and index."""
    return f"{ionization_mode[0].upper()}{index:013d}"


def _decoy_inchikey(ionization_mode: str, index: int) -> str:
    """A distinct 14-character first-block InChIKey for an incorrect candidate."""
    return f"{ionization_mode[0].upper()}I{index:012d}"


def _seed_mode(session: Session, ionization_mode: str) -> None:
    """One Orbitrap ground truth run of `ionization_mode`.

    Three Features, each with a correct assignment from `_POINTS_BY_MODE` and
    one incorrect candidate from `_INCORRECT_POINTS_BY_MODE`.
    """
    run = _insert_run(
        session, ionization_mode=ionization_mode, run_suffix=ionization_mode
    )
    for i, (ion_mass, csi_score) in enumerate(_POINTS_BY_MODE[ionization_mode]):
        feature = _insert_correct_assignment(
            session,
            run=run,
            feature_suffix=f"{ionization_mode}-{i}",
            inchikey=_inchikey(ionization_mode, i),
            ion_mass=ion_mass,
            csi_score=csi_score,
            adduct=_ADDUCT_BY_MODE[ionization_mode],
        )
        _insert_incorrect_candidate(
            session,
            feature=feature,
            candidate_inchikey=_decoy_inchikey(ionization_mode, i),
            csi_score=_INCORRECT_POINTS_BY_MODE[ionization_mode][i][1],
            adduct=_ADDUCT_BY_MODE[ionization_mode],
        )


def _seed_both_modes(session: Session) -> None:
    """Three distinct correct assignments each for positive and negative mode."""
    _seed_mode(session, "positive")
    _seed_mode(session, "negative")
    session.commit()


def _fit_from(tmp_path: Path, seed: Callable[[Session], None]) -> tuple[Result, Path]:
    """Seed a fresh DB with `seed`, run `fit-kde --output`, return its result and artifact path."""
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        seed(session)
        session.commit()
    engine.dispose()

    result = CliRunner().invoke(
        main, ["--config", str(config_path), "--output", str(output_path)]
    )
    return result, output_path


def _fitted_points(kde: gaussian_kde) -> set[tuple[float, float]]:
    return {(float(point[0]), float(point[1])) for point in kde.dataset.T}


def test_artifact_has_a_correct_and_an_incorrect_density_per_ionization_mode(
    tmp_path: Path,
) -> None:
    result, output_path = _fit_from(tmp_path, _seed_both_modes)

    assert result.exit_code == 0, result.output
    densities = load_kdes(output_path)
    assert set(densities) == {"positive", "negative"}
    for mode_densities in densities.values():
        assert isinstance(mode_densities.correct, gaussian_kde)
        assert isinstance(mode_densities.incorrect, gaussian_kde)


def test_each_density_is_fit_only_on_its_own_modes_assignments(
    tmp_path: Path,
) -> None:
    result, output_path = _fit_from(tmp_path, _seed_both_modes)

    assert result.exit_code == 0, result.output
    densities = load_kdes(output_path)
    for mode, points in _POINTS_BY_MODE.items():
        assert _fitted_points(densities[mode].correct) == set(points)
        assert _fitted_points(densities[mode].incorrect) == set(
            _INCORRECT_POINTS_BY_MODE[mode]
        )


def test_correct_assignments_dedup_on_true_inchikey_and_adduct_across_instruments(
    tmp_path: Path,
) -> None:
    """A QTOF repeat of an Orbitrap (true InChIKey, adduct) pair collapses into it;
    the same true InChIKey under another adduct is its own row."""

    def seed(session: Session) -> None:
        _seed_mode(session, "positive")  # Orbitrap; inserted first, so kept
        _seed_mode(session, "negative")
        qtof_run = _insert_run(
            session,
            ionization_mode="positive",
            run_suffix="qtof-positive",
            instrument_type="QTOF",
        )
        _insert_correct_assignment(
            session,
            run=qtof_run,
            feature_suffix="qtof-same-adduct",
            inchikey=_inchikey("positive", 0),
            ion_mass=400.0,
            csi_score=0.1,
            adduct="[M+H]+",
        )
        _insert_correct_assignment(
            session,
            run=qtof_run,
            feature_suffix="qtof-other-adduct",
            inchikey=_inchikey("positive", 0),
            ion_mass=420.0,
            csi_score=0.2,
            adduct="[M+Na]+",
        )

    result, output_path = _fit_from(tmp_path, seed)

    assert result.exit_code == 0, result.output
    assert _fitted_points(load_kdes(output_path)["positive"].correct) == {
        *_POINTS_BY_MODE["positive"],
        (420.0, 0.2),
    }


def test_field_run_features_never_contribute_to_the_fit(tmp_path: Path) -> None:
    """Even a field-run Feature carrying a matching true InChIKey is ignored."""

    def seed(session: Session) -> None:
        _seed_both_modes(session)
        field_run = _insert_run(
            session,
            ionization_mode="positive",
            run_suffix="field",
            source_kind=SOURCE_KIND_FIELD_MZML,
        )
        _insert_correct_assignment(
            session,
            run=field_run,
            feature_suffix="field-0",
            inchikey=_inchikey("field", 0),
            ion_mass=999.0,
            csi_score=0.99,
            adduct="[M+H]+",
        )

    result, output_path = _fit_from(tmp_path, seed)

    assert result.exit_code == 0, result.output
    assert _fitted_points(load_kdes(output_path)["positive"].correct) == set(
        _POINTS_BY_MODE["positive"]
    )


@pytest.mark.parametrize(
    ("present_mode", "missing_mode"),
    [("positive", "negative"), ("negative", "positive")],
)
def test_a_mode_without_correct_assignments_exits_nonzero_naming_it(
    tmp_path: Path, present_mode: str, missing_mode: str
) -> None:
    result, output_path = _fit_from(
        tmp_path, lambda session: _seed_mode(session, present_mode)
    )

    assert result.exit_code != 0
    assert repr(missing_mode) in result.output
    assert not output_path.exists()


def test_exactly_one_kde_models_row_is_inserted_pointing_at_the_pickle(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        _seed_both_modes(session)
    engine.dispose()

    result = CliRunner().invoke(
        main, ["--config", str(config_path), "--output", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    engine = create_database(db_path)
    with Session(engine) as session:
        rows = session.scalars(select(KdeModel)).all()
        assert len(rows) == 1
        assert rows[0].artifact_path == str(output_path)
    engine.dispose()


def test_zero_ground_truth_rows_exits_nonzero_and_tells_operator_to_run_groundtruth(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    create_database(db_path).dispose()

    result = CliRunner().invoke(main, ["--config", str(config_path)])

    assert result.exit_code != 0
    assert "generate-groundtruth" in result.output
    assert not output_path.exists()


def test_running_twice_produces_two_kde_models_rows_not_an_overwrite(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        _seed_both_modes(session)
    engine.dispose()

    first = CliRunner().invoke(main, ["--config", str(config_path)])
    second = CliRunner().invoke(main, ["--config", str(config_path)])

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    engine = create_database(db_path)
    with Session(engine) as session:
        rows = session.scalars(select(KdeModel)).all()
        assert len(rows) == 2
        assert rows[0].kde_model_id != rows[1].kde_model_id
        # Each row's artifact must still exist and be distinct from the
        # other's -- the default (no --output) path is timestamped per run
        # so a later fit never silently overwrites an earlier row's pickle.
        assert rows[0].artifact_path != rows[1].artifact_path
        assert Path(rows[0].artifact_path).exists()
        assert Path(rows[1].artifact_path).exists()
    engine.dispose()


def test_db_flag_overrides_the_config_files_db_path(tmp_path: Path) -> None:
    config_only_db_path = tmp_path / "config-db.duckdb"
    override_db_path = tmp_path / "override-db.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, config_only_db_path, output_path)
    engine = create_database(override_db_path)
    with Session(engine) as session:
        _seed_both_modes(session)
    engine.dispose()

    result = CliRunner().invoke(
        main, ["--config", str(config_path), "--db", str(override_db_path)]
    )

    assert result.exit_code == 0, result.output
    assert not config_only_db_path.exists()


def test_output_flag_overrides_the_config_files_kde_output_path(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    config_output_path = tmp_path / "config-model.pkl"
    override_output_path = tmp_path / "override-model.pkl"
    config_path = _write_config(tmp_path, db_path, config_output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        _seed_both_modes(session)
    engine.dispose()

    result = CliRunner().invoke(
        main, ["--config", str(config_path), "--output", str(override_output_path)]
    )

    assert result.exit_code == 0, result.output
    assert override_output_path.exists()
    assert not config_output_path.exists()


def test_missing_config_file_fails_fast_with_nonzero_exit(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main, ["--config", str(tmp_path / "does-not-exist.toml")]
    )

    assert result.exit_code != 0
    assert "does-not-exist.toml" in result.output


SMOKE_TEST_TOML = """
[smoke_test]
db_path = "{smoke_db_path}"
kde_output_path = "{smoke_kde_output_path}"
spectra_per_ionization_mode = 2
mzml_dir = "unused-smoke-mzml-dir"
metadata_csv = "unused-smoke-metadata.csv"
export_path = "unused-smoke-export.csv"
"""


def test_smoke_test_fits_from_the_smoke_db_and_writes_under_the_smoke_kde_path(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    output_dir = tmp_path / "models"
    smoke_db_path = tmp_path / "smoke" / "database.duckdb"
    smoke_output_dir = tmp_path / "smoke-models"
    config_path = _write_config(tmp_path, db_path, output_dir / "kde_model.pkl")
    with config_path.open("a") as handle:
        handle.write(
            SMOKE_TEST_TOML.format(
                smoke_db_path=smoke_db_path,
                smoke_kde_output_path=smoke_output_dir / "kde_model.pkl",
            )
        )
    smoke_db_path.parent.mkdir()
    engine = create_database(smoke_db_path)
    with Session(engine) as session:
        _seed_both_modes(session)
    engine.dispose()

    result = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])

    assert result.exit_code == 0, result.output
    assert not db_path.exists()
    assert not output_dir.exists()
    [pickle_path] = smoke_output_dir.glob("kde_model_*.pkl")
    engine = create_database(smoke_db_path)
    with Session(engine) as session:
        [kde_model] = session.scalars(select(KdeModel)).all()
        assert Path(kde_model.artifact_path) == pickle_path
    engine.dispose()


def test_smoke_test_rejects_the_db_flag(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path, tmp_path / "database.duckdb", tmp_path / "model.pkl"
    )

    result = CliRunner().invoke(
        main,
        ["--config", str(config_path), "--smoke-test", "--db", "other.duckdb"],
    )

    assert result.exit_code == 2
    assert "--db" in result.output


def test_smoke_test_with_an_empty_smoke_db_points_at_the_smoke_groundtruth(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path, tmp_path / "database.duckdb", tmp_path / "model.pkl"
    )
    with config_path.open("a") as handle:
        handle.write(
            SMOKE_TEST_TOML.format(
                smoke_db_path=tmp_path / "smoke.duckdb",
                smoke_kde_output_path=tmp_path / "smoke-model.pkl",
            )
        )

    result = CliRunner().invoke(main, ["--config", str(config_path), "--smoke-test"])

    assert result.exit_code != 0
    assert "generate-groundtruth --smoke-test" in result.output


def test_incorrect_rows_include_features_whose_truth_ranked_below_top_k(
    tmp_path: Path,
) -> None:
    """A Feature with no correct assignment still contributes its incorrect rows."""

    def seed(session: Session) -> None:
        _seed_both_modes(session)
        run = session.scalars(
            select(SiriusRun).where(SiriusRun.ionization_mode == "positive")
        ).one()
        unlisted = _insert_feature(
            session,
            run=run,
            feature_suffix="unlisted",
            true_inchikey=_inchikey("positive", 99),
            ion_mass=240.0,
            adduct="[M+H]+",
        )
        _insert_incorrect_candidate(
            session,
            feature=unlisted,
            candidate_inchikey=_decoy_inchikey("positive", 50),
            csi_score=-0.7,
            adduct="[M+H]+",
        )

    result, output_path = _fit_from(tmp_path, seed)

    assert result.exit_code == 0, result.output
    densities = load_kdes(output_path)
    assert _fitted_points(densities["positive"].incorrect) == {
        *_INCORRECT_POINTS_BY_MODE["positive"],
        (240.0, -0.7),
    }
    assert _fitted_points(densities["positive"].correct) == set(
        _POINTS_BY_MODE["positive"]
    )


def test_incorrect_rows_of_a_deduplicated_feature_never_contribute(
    tmp_path: Path,
) -> None:
    def seed(session: Session) -> None:
        _seed_both_modes(session)
        repeat_run = _insert_run(
            session,
            ionization_mode="positive",
            run_suffix="repeat",
            instrument_type="QTOF",
        )
        repeat = _insert_correct_assignment(
            session,
            run=repeat_run,
            feature_suffix="repeat",
            inchikey=_inchikey("positive", 0),
            ion_mass=400.0,
            csi_score=0.1,
            adduct="[M+H]+",
        )
        _insert_incorrect_candidate(
            session,
            feature=repeat,
            candidate_inchikey=_decoy_inchikey("positive", 60),
            csi_score=-5.0,
            adduct="[M+H]+",
        )

    result, output_path = _fit_from(tmp_path, seed)

    assert result.exit_code == 0, result.output
    assert _fitted_points(load_kdes(output_path)["positive"].incorrect) == set(
        _INCORRECT_POINTS_BY_MODE["positive"]
    )


def test_field_run_incorrect_candidates_never_contribute(tmp_path: Path) -> None:
    def seed(session: Session) -> None:
        _seed_both_modes(session)
        field_run = _insert_run(
            session,
            ionization_mode="positive",
            run_suffix="field",
            source_kind=SOURCE_KIND_FIELD_MZML,
        )
        field_feature = _insert_feature(
            session,
            run=field_run,
            feature_suffix="field-0",
            true_inchikey=_inchikey("field", 0),
            ion_mass=999.0,
            adduct="[M+H]+",
        )
        _insert_incorrect_candidate(
            session,
            feature=field_feature,
            candidate_inchikey=_decoy_inchikey("positive", 70),
            csi_score=-9.0,
            adduct="[M+H]+",
        )

    result, output_path = _fit_from(tmp_path, seed)

    assert result.exit_code == 0, result.output
    assert _fitted_points(load_kdes(output_path)["positive"].incorrect) == set(
        _INCORRECT_POINTS_BY_MODE["positive"]
    )


@pytest.mark.parametrize("missing_mode", ["positive", "negative"])
def test_a_mode_without_incorrect_assignments_exits_nonzero_naming_it(
    tmp_path: Path, missing_mode: str
) -> None:
    present_mode = "negative" if missing_mode == "positive" else "positive"

    def seed(session: Session) -> None:
        _seed_mode(session, present_mode)
        run = _insert_run(
            session, ionization_mode=missing_mode, run_suffix=missing_mode
        )
        for i, (ion_mass, csi_score) in enumerate(_POINTS_BY_MODE[missing_mode]):
            _insert_correct_assignment(
                session,
                run=run,
                feature_suffix=f"{missing_mode}-{i}",
                inchikey=_inchikey(missing_mode, i),
                ion_mass=ion_mass,
                csi_score=csi_score,
                adduct=_ADDUCT_BY_MODE[missing_mode],
            )

    result, output_path = _fit_from(tmp_path, seed)

    assert result.exit_code != 0
    assert repr(missing_mode) in result.output
    assert "incorrect assignments are missing" in result.output
    assert not output_path.exists()


_OVERSIZE = 20_003


def _seed_oversized_positive_incorrect_set(session: Session) -> None:
    """Positive mode gets `_OVERSIZE` incorrect rows on one extra Feature."""
    _seed_both_modes(session)
    run = session.scalars(
        select(SiriusRun).where(SiriusRun.ionization_mode == "positive")
    ).one()
    feature = _insert_feature(
        session,
        run=run,
        feature_suffix="many",
        true_inchikey=_inchikey("positive", 99),
        ion_mass=240.0,
        adduct="[M+H]+",
    )
    first_molecule_id = (
        session.scalars(
            select(Molecule.molecule_id).order_by(Molecule.molecule_id.desc())
        ).first()
        or 0
    ) + 1
    session.execute(
        sa_insert(Molecule),
        [
            {
                "molecule_id": first_molecule_id + i,
                "inchikey": f"ZZ{i:012d}",
                "smiles": "CCO",
                "molecular_formula": "C2H6O",
            }
            for i in range(_OVERSIZE)
        ],
    )
    session.execute(
        sa_insert(Annotation),
        [
            {
                "feature_id": feature.feature_id,
                "molecule_id": first_molecule_id + i,
                "rank": i + 2,
                "csi_score": -1.0 - (i % 997) / 100.0,
                "tanimoto_similarity": 1.0,
                "mces_dist_to_top_hit": 0.0,
                "xlogp": 1.0,
                "adduct": "[M+H]+",
                "formula_id": "formula-1",
            }
            for i in range(_OVERSIZE)
        ],
    )


def test_oversized_incorrect_set_is_subsampled_deterministically_and_recorded(
    tmp_path: Path,
) -> None:
    first, first_path = _fit_from(tmp_path, _seed_oversized_positive_incorrect_set)
    assert first.exit_code == 0, first.output

    db_path = tmp_path / "database.duckdb"
    second_path = tmp_path / "second.pkl"
    second = CliRunner().invoke(
        main,
        [
            "--config",
            str(tmp_path / "config.toml"),
            "--output",
            str(second_path),
        ],
    )
    assert second.exit_code == 0, second.output

    first_densities = load_kdes(first_path)
    second_densities = load_kdes(second_path)
    # g0 holds exactly the cap, from 20,003 + 3 candidates; g1 is never subsampled.
    assert first_densities["positive"].incorrect.n == 20_000
    assert first_densities["positive"].correct.n == 3
    # The same data and seed give an identical artifact.
    for mode in ("positive", "negative"):
        for first_kde, second_kde in (
            (first_densities[mode].correct, second_densities[mode].correct),
            (first_densities[mode].incorrect, second_densities[mode].incorrect),
        ):
            assert (first_kde.dataset == second_kde.dataset).all()
    # A mode within the cap is fitted on every incorrect row.
    assert first_densities["negative"].incorrect.n == 3

    engine = create_database(db_path)
    with Session(engine) as session:
        for row in session.scalars(select(KdeModel)):
            assert row.subsample_seed is not None
            assert row.incorrect_subsample_size == 20_000
            assert row.positive_incorrect_rows_used == 20_000
            assert row.negative_incorrect_rows_used == 3
    engine.dispose()

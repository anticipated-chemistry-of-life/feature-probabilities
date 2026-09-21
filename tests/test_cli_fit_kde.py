"""Tests for the `fit-kde` CLI.

Exercises the CLI end-to-end via `click.testing.CliRunner` against a real
temp DuckDB (no fake needed -- `fit-kde` never talks to SIRIUS), with fixture
ground-truth rows inserted directly through the real SQLAlchemy models
rather than depending on `generate-groundtruth`'s actual code having run,
per issue #22's acceptance criteria and issue #9's testing decisions.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from click.testing import CliRunner
from scipy.stats import gaussian_kde
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.cli_fit_kde import main
from feature_probabilities.kde import GLOBAL_STRATUM, TRAINED_INSTRUMENT_TYPES
from feature_probabilities.schema import (
    SOURCE_KIND_FIELD_MZML,
    SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
    Annotation,
    Feature,
    KdeModel,
    Molecule,
    SiriusRun,
    create_database,
)

CONFIG_TOML = """
db_path = "{db_path}"
required_sirius_version = "6.0.0"
massspecgym_revision = "abc123"
kde_output_path = "{kde_output_path}"
"""


def _write_config(tmp_path: Path, db_path: Path, kde_output_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        CONFIG_TOML.format(db_path=db_path, kde_output_path=kde_output_path)
    )
    return config_path


def _insert_ground_truth_run(
    session: Session, *, instrument_type: str, run_suffix: str
) -> SiriusRun:
    run = SiriusRun(
        extract_id=None,
        source_kind=SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
        input_file_checksum=f"checksum-{run_suffix}",
        import_params_checksum=None,
        analysis_params_checksum=f"analysis-{run_suffix}",
        sirius_version="6.0.0",
        pysirius_client_version="1.0.0",
        ionization_mode="positive",
        instrument_type=instrument_type,
        input_file_path=f"/data/{run_suffix}.mgf",
    )
    session.add(run)
    session.flush()
    return run


def _insert_correct_assignment(
    session: Session,
    *,
    run: SiriusRun,
    feature_suffix: str,
    inchikey: str,
    ion_mass: float,
    csi_score: float,
    adduct: str = "[M+H]+",
) -> None:
    """Insert one feature + molecule + annotation whose candidate matches ground truth."""
    feature = Feature(
        run_id=run.run_id,
        external_feature_id=f"feature-{feature_suffix}",
        ion_mass=ion_mass,
        charge=1,
        true_inchikey=inchikey,
    )
    molecule = Molecule(
        inchikey=inchikey,
        smiles="CCO",
        molecular_formula="C2H6O",
    )
    session.add_all([feature, molecule])
    session.flush()

    session.add(
        Annotation(
            feature_id=feature.feature_id,
            molecule_id=molecule.molecule_id,
            rank=1,
            csi_score=csi_score,
            tanimoto_similarity=1.0,
            mces_dist_to_top_hit=0.0,
            xlogp=1.0,
            adduct=adduct,
            formula_id="formula-1",
        )
    )
    session.flush()


def _seed_two_strata(session: Session) -> None:
    """Three correct-assignment rows each for Orbitrap and QTOF, distinct molecules."""
    orbitrap_run = _insert_ground_truth_run(
        session, instrument_type="Orbitrap", run_suffix="orb"
    )
    qtof_run = _insert_ground_truth_run(
        session, instrument_type="QTOF", run_suffix="qtof"
    )
    orbitrap_inchikeys = ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"]
    qtof_inchikeys = ["DDDDDDDDDDDDDD", "EEEEEEEEEEEEEE", "FFFFFFFFFFFFFF"]
    # Non-collinear (ion_mass, csi_score) pairs -- gaussian_kde needs a
    # non-singular 2D covariance, which a perfectly linear point set lacks.
    orbitrap_points = [(200.0, 0.9), (210.0, 0.5), (225.0, 0.7)]
    qtof_points = [(300.0, 0.6), (310.0, 0.3), (330.0, 0.55)]
    for i, (inchikey, (ion_mass, csi_score)) in enumerate(
        zip(orbitrap_inchikeys, orbitrap_points, strict=True)
    ):
        _insert_correct_assignment(
            session,
            run=orbitrap_run,
            feature_suffix=f"orb-{i}",
            inchikey=inchikey,
            ion_mass=ion_mass,
            csi_score=csi_score,
        )
    for i, (inchikey, (ion_mass, csi_score)) in enumerate(
        zip(qtof_inchikeys, qtof_points, strict=True)
    ):
        _insert_correct_assignment(
            session,
            run=qtof_run,
            feature_suffix=f"qtof-{i}",
            inchikey=inchikey,
            ion_mass=ion_mass,
            csi_score=csi_score,
        )
    session.commit()


def test_full_run_produces_a_pickle_with_exactly_the_three_stratum_keys(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        _seed_two_strata(session)
    engine.dispose()

    result = CliRunner().invoke(
        main, ["--config", str(config_path), "--output", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    assert output_path.exists()
    with output_path.open("rb") as handle:
        kdes = pickle.load(handle)
    assert set(kdes) == {*TRAINED_INSTRUMENT_TYPES, GLOBAL_STRATUM}
    for kde in kdes.values():
        assert isinstance(kde, gaussian_kde)


def test_global_stratum_is_fit_over_the_union_of_both_strata(tmp_path: Path) -> None:
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        _seed_two_strata(session)
    engine.dispose()

    result = CliRunner().invoke(
        main, ["--config", str(config_path), "--output", str(output_path)]
    )

    assert result.exit_code == 0, result.output
    with output_path.open("rb") as handle:
        kdes = pickle.load(handle)
    assert kdes["__global__"].dataset.shape[1] == (
        kdes["Orbitrap"].dataset.shape[1] + kdes["QTOF"].dataset.shape[1]
    )


def test_exactly_one_kde_models_row_is_inserted_pointing_at_the_pickle(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        _seed_two_strata(session)
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


def test_field_run_features_are_excluded_even_with_a_matching_inchikey(
    tmp_path: Path,
) -> None:
    """A field-run feature never has `true_inchikey` set, so it can't fake ground truth."""
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        field_run = SiriusRun(
            extract_id=None,
            source_kind=SOURCE_KIND_FIELD_MZML,
            input_file_checksum="field-checksum",
            import_params_checksum="import-checksum",
            analysis_params_checksum="analysis-checksum",
            sirius_version="6.0.0",
            pysirius_client_version="1.0.0",
            ionization_mode="positive",
            instrument_type="Orbitrap",
            input_file_path="/data/field.mzML",
        )
        session.add(field_run)
        session.flush()
        _insert_correct_assignment(
            session,
            run=field_run,
            feature_suffix="field-0",
            inchikey="AAAAAAAAAAAAAA",
            ion_mass=200.0,
            csi_score=0.5,
        )
        # A field-run feature's true_inchikey is always NULL, per the schema.
        feature = session.scalars(select(Feature)).one()
        feature.true_inchikey = None
        session.commit()
    engine.dispose()

    result = CliRunner().invoke(main, ["--config", str(config_path)])

    assert result.exit_code != 0
    assert "generate-groundtruth" in result.output


def test_running_twice_produces_two_kde_models_rows_not_an_overwrite(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        _seed_two_strata(session)
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
        _seed_two_strata(session)
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
        _seed_two_strata(session)
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


@pytest.mark.parametrize("missing_stratum", ["Orbitrap", "QTOF"])
def test_missing_one_stratums_data_exits_nonzero_with_a_clear_message(
    tmp_path: Path, missing_stratum: str
) -> None:
    db_path = tmp_path / "database.duckdb"
    output_path = tmp_path / "model.pkl"
    config_path = _write_config(tmp_path, db_path, output_path)
    engine = create_database(db_path)
    with Session(engine) as session:
        present_instrument_type = (
            "QTOF" if missing_stratum == "Orbitrap" else "Orbitrap"
        )
        run = _insert_ground_truth_run(
            session, instrument_type=present_instrument_type, run_suffix="only"
        )
        present_points = [(200.0, 0.9), (210.0, 0.5), (225.0, 0.7)]
        present_inchikeys = ["AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB", "CCCCCCCCCCCCCC"]
        for i, (inchikey, (ion_mass, csi_score)) in enumerate(
            zip(present_inchikeys, present_points, strict=True)
        ):
            _insert_correct_assignment(
                session,
                run=run,
                feature_suffix=f"only-{i}",
                inchikey=inchikey,
                ion_mass=ion_mass,
                csi_score=csi_score,
            )
        session.commit()
    engine.dispose()

    result = CliRunner().invoke(main, ["--config", str(config_path)])

    assert result.exit_code != 0
    assert missing_stratum in result.output

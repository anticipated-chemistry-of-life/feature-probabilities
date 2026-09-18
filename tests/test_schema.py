"""Tests for the single-source-of-truth DuckDB schema.

Exercises the real SQLAlchemy models against a real DuckDB engine (in-memory
or a temp file), per the project's testing decisions: DuckDB is embedded and
fast enough that mocking it would only hide bugs in the actual schema.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from feature_probabilities.schema import (
    SOURCE_KIND_FIELD_MZML,
    SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
    Annotation,
    Base,
    CalibrationScore,
    Extract,
    Feature,
    KdeModel,
    Molecule,
    SiriusRun,
    Species,
    create_database,
)

EXPECTED_TABLE_NAMES = {
    "species",
    "extracts",
    "sirius_runs",
    "features",
    "molecules",
    "annotations",
    "kde_models",
    "calibration_scores",
}


def _insert_full_graph(session: Session) -> dict[str, int]:
    """Insert one valid row per table, in FK-respecting order, and flush.

    Returns the assigned primary keys so callers can build on top of a known-
    good baseline (e.g. to test a *second* conflicting insert).
    """
    species = Species(taxon_name="Panthera leo", ncbi_taxid=9689, family="Felidae")
    session.add(species)
    session.flush()

    extract = Extract(
        sample_code="EX-001",
        species_id=species.species_id,
        organ="leaf",
        extra_metadata={"collector": "field team A"},
    )
    session.add(extract)
    session.flush()

    run = SiriusRun(
        extract_id=extract.extract_id,
        source_kind=SOURCE_KIND_FIELD_MZML,
        input_file_checksum="a" * 64,
        import_params_checksum="b" * 64,
        analysis_params_checksum="c" * 64,
        sirius_version="6.0.0",
        pysirius_client_version="1.0.0",
        ionization_mode="positive",
        instrument_type="Orbitrap",
        input_file_path="/data/sample.mzML",
    )
    session.add(run)
    session.flush()

    feature = Feature(
        run_id=run.run_id,
        external_feature_id="feature-1",
        ion_mass=301.1234,
        charge=1,
        rt_start_seconds=10.0,
        rt_end_seconds=12.0,
        rt_apex_seconds=11.0,
        quality="GOOD",
        true_inchikey=None,
    )
    molecule = Molecule(
        inchikey="AXFAVZQXPFQIEI",
        smiles="CCO",
        molecular_formula="C2H6O",
    )
    session.add_all([feature, molecule])
    session.flush()

    annotation = Annotation(
        feature_id=feature.feature_id,
        molecule_id=molecule.molecule_id,
        rank=1,
        csi_score=0.9,
        tanimoto_similarity=0.8,
        mces_dist_to_top_hit=0.0,
        xlogp=1.2,
        adduct="[M+H]+",
        formula_id="formula-1",
    )
    session.add(annotation)
    session.flush()

    kde_model = KdeModel(artifact_path="/artifacts/kde-v1.pkl")
    session.add(kde_model)
    session.flush()

    calibration_score = CalibrationScore(
        annotation_id=annotation.annotation_id,
        kde_model_id=kde_model.kde_model_id,
        score=0.42,
    )
    session.add(calibration_score)
    session.flush()

    return {
        "species_id": species.species_id,
        "extract_id": extract.extract_id,
        "run_id": run.run_id,
        "feature_id": feature.feature_id,
        "molecule_id": molecule.molecule_id,
        "annotation_id": annotation.annotation_id,
        "kde_model_id": kde_model.kde_model_id,
    }


def test_create_database_creates_a_file_containing_all_eight_tables(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "fresh" / "database.duckdb"

    engine = create_database(db_path)

    assert db_path.exists()
    assert set(sa.inspect(engine).get_table_names()) == EXPECTED_TABLE_NAMES
    engine.dispose()


def test_create_database_is_idempotent_and_preserves_existing_data(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "database.duckdb"
    engine = create_database(db_path)
    with Session(engine) as session:
        session.add(Species(taxon_name="Panthera leo"))
        session.commit()
    engine.dispose()

    reopened = create_database(db_path)

    with Session(reopened) as session:
        assert session.scalar(sa.select(sa.func.count()).select_from(Species)) == 1
    reopened.dispose()


def test_inserting_one_valid_row_per_table_succeeds() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        ids = _insert_full_graph(session)
        session.commit()

        assert session.get(Species, ids["species_id"]) is not None
        assert session.get(Extract, ids["extract_id"]) is not None
        assert session.get(SiriusRun, ids["run_id"]) is not None
        assert session.get(Feature, ids["feature_id"]) is not None
        assert session.get(Molecule, ids["molecule_id"]) is not None
        assert session.get(Annotation, ids["annotation_id"]) is not None
        assert session.get(KdeModel, ids["kde_model_id"]) is not None
        calibration_score = session.get(
            CalibrationScore, (ids["annotation_id"], ids["kde_model_id"])
        )
        assert calibration_score is not None
        assert calibration_score.score == pytest.approx(0.42)


def test_duplicate_taxon_name_is_rejected() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        session.add(Species(taxon_name="Panthera leo"))
        session.commit()

        session.add(Species(taxon_name="Panthera leo"))
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_duplicate_sample_code_is_rejected() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        species = Species(taxon_name="Panthera leo")
        session.add(species)
        session.flush()
        session.add(Extract(sample_code="EX-001", species_id=species.species_id))
        session.commit()

        session.add(Extract(sample_code="EX-001", species_id=species.species_id))
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_duplicate_inchikey_is_rejected() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        session.add(
            Molecule(inchikey="AXFAVZQXPFQIEI", smiles="CCO", molecular_formula="C2H6O")
        )
        session.commit()

        session.add(
            Molecule(inchikey="AXFAVZQXPFQIEI", smiles="CCN", molecular_formula="C2H7N")
        )
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_duplicate_feature_molecule_annotation_pair_is_rejected() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        ids = _insert_full_graph(session)
        session.commit()

        session.add(
            Annotation(
                feature_id=ids["feature_id"],
                molecule_id=ids["molecule_id"],
                rank=2,
                csi_score=0.5,
                tanimoto_similarity=0.5,
                mces_dist_to_top_hit=1.0,
                xlogp=0.5,
                adduct="[M+Na]+",
                formula_id="formula-2",
            )
        )
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_sirius_run_extract_id_accepts_null_for_ground_truth_runs() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        run = SiriusRun(
            extract_id=None,
            source_kind=SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
            input_file_checksum="a" * 64,
            import_params_checksum=None,
            analysis_params_checksum="c" * 64,
            sirius_version="6.0.0",
            pysirius_client_version="1.0.0",
            ionization_mode="positive",
            instrument_type="Orbitrap",
            input_file_path="/data/chunk-0001.mgf",
        )
        session.add(run)
        session.commit()

        stored = session.get(SiriusRun, run.run_id)
        assert stored is not None
        assert stored.extract_id is None
        assert stored.import_params_checksum is None


def test_sirius_run_extract_id_accepts_a_valid_fk_for_field_runs() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        species = Species(taxon_name="Panthera leo")
        session.add(species)
        session.flush()
        extract = Extract(sample_code="EX-001", species_id=species.species_id)
        session.add(extract)
        session.flush()

        run = SiriusRun(
            extract_id=extract.extract_id,
            source_kind=SOURCE_KIND_FIELD_MZML,
            input_file_checksum="a" * 64,
            import_params_checksum="b" * 64,
            analysis_params_checksum="c" * 64,
            sirius_version="6.0.0",
            pysirius_client_version="1.0.0",
            ionization_mode="positive",
            instrument_type="Orbitrap",
            input_file_path="/data/sample.mzML",
        )
        session.add(run)
        session.commit()

        stored = session.get(SiriusRun, run.run_id)
        assert stored is not None
        assert stored.extract_id == extract.extract_id


def test_sirius_run_rejects_an_unknown_extract_id() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        run = SiriusRun(
            extract_id=999_999,
            source_kind=SOURCE_KIND_FIELD_MZML,
            input_file_checksum="a" * 64,
            import_params_checksum="b" * 64,
            analysis_params_checksum="c" * 64,
            sirius_version="6.0.0",
            pysirius_client_version="1.0.0",
            ionization_mode="positive",
            instrument_type="Orbitrap",
            input_file_path="/data/sample.mzML",
        )
        session.add(run)
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_sirius_run_rejects_an_unrecognized_source_kind() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        run = SiriusRun(
            extract_id=None,
            source_kind="not_a_real_source_kind",
            input_file_checksum="a" * 64,
            import_params_checksum=None,
            analysis_params_checksum="c" * 64,
            sirius_version="6.0.0",
            pysirius_client_version="1.0.0",
            ionization_mode="positive",
            instrument_type="Orbitrap",
            input_file_path="/data/chunk-0001.mgf",
        )
        session.add(run)
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_two_calibration_scores_for_the_same_annotation_with_different_kde_models_both_succeed() -> (
    None
):
    engine = create_database(":memory:")
    with Session(engine) as session:
        ids = _insert_full_graph(session)
        session.commit()

        second_kde_model = KdeModel(artifact_path="/artifacts/kde-v2.pkl")
        session.add(second_kde_model)
        session.flush()
        session.add(
            CalibrationScore(
                annotation_id=ids["annotation_id"],
                kde_model_id=second_kde_model.kde_model_id,
                score=0.77,
            )
        )
        session.commit()

        scores = session.scalars(
            sa.select(CalibrationScore).where(
                CalibrationScore.annotation_id == ids["annotation_id"]
            )
        ).all()
        assert {s.kde_model_id for s in scores} == {
            ids["kde_model_id"],
            second_kde_model.kde_model_id,
        }
        assert {s.score for s in scores} == {0.42, 0.77}


def test_duplicate_calibration_score_for_same_annotation_and_kde_model_is_rejected() -> (
    None
):
    engine = create_database(":memory:")
    with Session(engine) as session:
        ids = _insert_full_graph(session)
        session.commit()

        session.add(
            CalibrationScore(
                annotation_id=ids["annotation_id"],
                kde_model_id=ids["kde_model_id"],
                score=0.99,
            )
        )
        with pytest.raises(sa.exc.IntegrityError):
            session.commit()


def test_schema_metadata_lists_exactly_the_eight_expected_tables() -> None:
    assert set(Base.metadata.tables) == EXPECTED_TABLE_NAMES

"""Tests for calibration application: scoring annotations against a fitted KDE pickle.

Exercises issue #24's acceptance criteria directly: stratum selection by
instrument type, the `__global__` fallback for an unmatched instrument
type, and `calibration_scores` versioning across `kde_model_id`s -- all
against a real temp DuckDB and real `scipy.stats.gaussian_kde` fits, per the
project's testing decisions (no DB or KDE mocking).
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import gaussian_kde
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.calibrate import apply_calibration, load_kdes
from feature_probabilities.kde import GLOBAL_STRATUM
from feature_probabilities.schema import (
    Annotation,
    CalibrationScore,
    Feature,
    KdeModel,
    Molecule,
    SiriusRun,
    create_database,
)


def _kde(points: list[tuple[float, float]]) -> gaussian_kde:
    return gaussian_kde(np.array(points).T)


def _kdes() -> dict[str, gaussian_kde]:
    orbitrap_points = [(300.0, 0.9), (301.0, 0.85), (299.5, 0.88), (300.5, 0.92)]
    qtof_points = [(500.0, 0.5), (501.0, 0.55), (499.0, 0.52), (500.5, 0.48)]
    return {
        "Orbitrap": _kde(orbitrap_points),
        "QTOF": _kde(qtof_points),
        GLOBAL_STRATUM: _kde(orbitrap_points + qtof_points),
    }


def _insert_annotation(
    session: Session,
    *,
    instrument_type: str,
    ion_mass: float,
    csi_score: float,
    inchikey: str,
    run_suffix: str,
) -> int:
    """Insert a minimal `sirius_runs`/`features`/`molecules`/`annotations` row set.

    Returns the new annotation's `annotation_id`.
    """
    run = SiriusRun(
        extract_id=None,
        source_kind="field_mzml",
        input_file_checksum=f"checksum-{run_suffix}",
        import_params_checksum=f"import-{run_suffix}",
        analysis_params_checksum=f"analysis-{run_suffix}",
        sirius_version="6.0.0",
        pysirius_client_version="1.0.0",
        ionization_mode="positive",
        instrument_type=instrument_type,
        input_file_path=f"/data/{run_suffix}.mzML",
    )
    session.add(run)
    session.flush()

    feature = Feature(
        run_id=run.run_id,
        external_feature_id=f"feature-{run_suffix}",
        ion_mass=ion_mass,
        charge=1,
    )
    molecule = Molecule(inchikey=inchikey, smiles="CCO", molecular_formula="C2H6O")
    session.add_all([feature, molecule])
    session.flush()

    annotation = Annotation(
        feature_id=feature.feature_id,
        molecule_id=molecule.molecule_id,
        rank=1,
        csi_score=csi_score,
        tanimoto_similarity=0.8,
        mces_dist_to_top_hit=0.0,
        xlogp=1.2,
        adduct="[M+H]+",
        formula_id=f"formula-{run_suffix}",
    )
    session.add(annotation)
    session.flush()
    return annotation.annotation_id


def test_each_annotation_is_scored_against_its_matching_trained_stratum() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes = _kdes()
        orbitrap_id = _insert_annotation(
            session,
            instrument_type="Orbitrap",
            ion_mass=300.2,
            csi_score=0.9,
            inchikey="AAAAAAAAAAAAAA",
            run_suffix="orb",
        )
        qtof_id = _insert_annotation(
            session,
            instrument_type="QTOF",
            ion_mass=500.2,
            csi_score=0.51,
            inchikey="BBBBBBBBBBBBBB",
            run_suffix="qtof",
        )
        kde_model = KdeModel(artifact_path="/artifacts/kde-v1.pkl")
        session.add(kde_model)
        session.flush()

        summary = apply_calibration(
            session,
            kdes,
            kde_model.kde_model_id,
            annotation_ids=[orbitrap_id, qtof_id],
        )

        assert summary.annotations_scored == 2
        assert summary.kde_model_id == kde_model.kde_model_id

        orbitrap_score = session.get(
            CalibrationScore, (orbitrap_id, kde_model.kde_model_id)
        )
        qtof_score = session.get(CalibrationScore, (qtof_id, kde_model.kde_model_id))
        assert orbitrap_score is not None
        assert qtof_score is not None
        assert orbitrap_score.score == pytest.approx(
            float(kdes["Orbitrap"].pdf((300.2, 0.9))[0])
        )
        assert qtof_score.score == pytest.approx(
            float(kdes["QTOF"].pdf((500.2, 0.51))[0])
        )
        # Sanity: the two strata's KDEs actually disagree here, so a
        # mismatched-stratum bug (e.g. always using __global__) would fail
        # the exact-match assertions above.
        assert orbitrap_score.score != pytest.approx(
            float(kdes["QTOF"].pdf((300.2, 0.9))[0])
        )


def test_unmatched_instrument_type_falls_back_to_global_without_erroring() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes = _kdes()
        annotation_id = _insert_annotation(
            session,
            instrument_type="Waters",
            ion_mass=350.0,
            csi_score=0.7,
            inchikey="CCCCCCCCCCCCCC",
            run_suffix="waters",
        )
        kde_model = KdeModel(artifact_path="/artifacts/kde-v1.pkl")
        session.add(kde_model)
        session.flush()

        summary = apply_calibration(
            session, kdes, kde_model.kde_model_id, annotation_ids=[annotation_id]
        )

        assert summary.annotations_scored == 1
        stored = session.get(CalibrationScore, (annotation_id, kde_model.kde_model_id))
        assert stored is not None
        assert stored.score == pytest.approx(
            float(kdes[GLOBAL_STRATUM].pdf((350.0, 0.7))[0])
        )


def test_second_kde_model_id_inserts_a_second_row_rather_than_overwriting() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes_v1 = _kdes()
        annotation_id = _insert_annotation(
            session,
            instrument_type="Orbitrap",
            ion_mass=300.2,
            csi_score=0.9,
            inchikey="DDDDDDDDDDDDDD",
            run_suffix="orb",
        )
        kde_model_1 = KdeModel(artifact_path="/artifacts/kde-v1.pkl")
        session.add(kde_model_1)
        session.flush()
        apply_calibration(
            session, kdes_v1, kde_model_1.kde_model_id, annotation_ids=[annotation_id]
        )

        # A distinctly different refit, so its pdf value at the same point
        # differs from kdes_v1's -- proving the first row survives untouched.
        kdes_v2 = {
            "Orbitrap": _kde([(310.0, 0.6), (311.0, 0.62), (309.5, 0.58)]),
            "QTOF": _kde([(500.0, 0.5), (501.0, 0.55), (499.0, 0.52)]),
            GLOBAL_STRATUM: _kde(
                [(310.0, 0.6), (311.0, 0.62), (309.5, 0.58), (500.0, 0.5)]
            ),
        }
        kde_model_2 = KdeModel(artifact_path="/artifacts/kde-v2.pkl")
        session.add(kde_model_2)
        session.flush()
        apply_calibration(
            session, kdes_v2, kde_model_2.kde_model_id, annotation_ids=[annotation_id]
        )

        rows = session.scalars(
            select(CalibrationScore).where(
                CalibrationScore.annotation_id == annotation_id
            )
        ).all()
        assert {row.kde_model_id for row in rows} == {
            kde_model_1.kde_model_id,
            kde_model_2.kde_model_id,
        }

        score_v1 = next(
            row.score for row in rows if row.kde_model_id == kde_model_1.kde_model_id
        )
        score_v2 = next(
            row.score for row in rows if row.kde_model_id == kde_model_2.kde_model_id
        )
        assert score_v1 == pytest.approx(
            float(kdes_v1["Orbitrap"].pdf((300.2, 0.9))[0])
        )
        assert score_v2 == pytest.approx(
            float(kdes_v2["Orbitrap"].pdf((300.2, 0.9))[0])
        )
        assert score_v1 != pytest.approx(score_v2)


def test_only_requested_annotation_ids_are_scored_other_rows_untouched() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes = _kdes()
        included_id = _insert_annotation(
            session,
            instrument_type="Orbitrap",
            ion_mass=300.2,
            csi_score=0.9,
            inchikey="EEEEEEEEEEEEEE",
            run_suffix="included",
        )
        _insert_annotation(
            session,
            instrument_type="QTOF",
            ion_mass=500.2,
            csi_score=0.51,
            inchikey="FFFFFFFFFFFFFF",
            run_suffix="excluded",
        )
        kde_model = KdeModel(artifact_path="/artifacts/kde-v1.pkl")
        session.add(kde_model)
        session.flush()

        summary = apply_calibration(
            session, kdes, kde_model.kde_model_id, annotation_ids=[included_id]
        )

        assert summary.annotations_scored == 1
        rows = session.scalars(select(CalibrationScore)).all()
        assert [row.annotation_id for row in rows] == [included_id]


def test_load_kdes_round_trips_a_pickled_artifact(tmp_path: Path) -> None:
    kdes = _kdes()
    artifact_path = tmp_path / "kde.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(kdes, handle)

    loaded = load_kdes(artifact_path)

    assert set(loaded) == set(kdes)
    for stratum, kde in kdes.items():
        assert isinstance(loaded[stratum], gaussian_kde)
        assert loaded[stratum].pdf((300.0, 0.9)) == pytest.approx(kde.pdf((300.0, 0.9)))

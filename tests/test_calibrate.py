"""Tests for calibration application: scoring annotations against a fitted KDE pickle.

Exercises KDE selection by the SIRIUS run's Ionization mode (ADR 0002) and
`calibration_scores` versioning across `kde_model_id`s -- all against a real
temp DuckDB and real `scipy.stats.gaussian_kde` fits, per the project's
testing decisions (no DB or KDE mocking).
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
from feature_probabilities.kde import ModeDensities
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
    positive_points = [(300.0, 0.9), (301.0, 0.85), (299.5, 0.88), (300.5, 0.92)]
    negative_points = [(500.0, 0.5), (501.0, 0.55), (499.0, 0.52), (500.5, 0.48)]
    return {"positive": _kde(positive_points), "negative": _kde(negative_points)}


def _insert_annotation(
    session: Session,
    *,
    ionization_mode: str,
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
        ionization_mode=ionization_mode,
        instrument_type="Orbitrap",
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


def test_each_annotation_is_scored_by_its_runs_ionization_mode_kde() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes = _kdes()
        positive_id = _insert_annotation(
            session,
            ionization_mode="positive",
            ion_mass=300.2,
            csi_score=0.9,
            inchikey="AAAAAAAAAAAAAA",
            run_suffix="positive",
        )
        negative_id = _insert_annotation(
            session,
            ionization_mode="negative",
            ion_mass=500.2,
            csi_score=0.51,
            inchikey="BBBBBBBBBBBBBB",
            run_suffix="negative",
        )
        kde_model = KdeModel(
            artifact_path="/artifacts/kde-v1.pkl",
            subsample_seed=0,
            incorrect_subsample_size=20_000,
            positive_incorrect_rows_used=1,
            negative_incorrect_rows_used=1,
        )
        session.add(kde_model)
        session.flush()

        summary = apply_calibration(
            session,
            kdes,
            kde_model.kde_model_id,
            annotation_ids=[positive_id, negative_id],
        )

        assert summary.annotations_scored == 2
        assert summary.kde_model_id == kde_model.kde_model_id

        positive_score = session.get(
            CalibrationScore, (positive_id, kde_model.kde_model_id)
        )
        negative_score = session.get(
            CalibrationScore, (negative_id, kde_model.kde_model_id)
        )
        assert positive_score is not None
        assert negative_score is not None
        assert positive_score.score == pytest.approx(
            float(kdes["positive"].pdf((300.2, 0.9))[0])
        )
        assert negative_score.score == pytest.approx(
            float(kdes["negative"].pdf((500.2, 0.51))[0])
        )
        # Sanity: the two modes' KDEs actually disagree here, so a
        # mismatched-mode bug would fail the exact-match assertions above.
        assert positive_score.score != pytest.approx(
            float(kdes["negative"].pdf((300.2, 0.9))[0])
        )


def test_mixed_mode_batch_larger_than_a_chunk_matches_per_row_evaluation() -> None:
    """Issue #34: evaluating in chunks must not change any stored value.

    The two Ionization modes alternate in annotation order. With chunks of 3,
    the 7 positive annotations fill two chunks and leave a third holding one,
    while the 6 negative ones fill exactly two: a chunk boundary falls inside
    each mode's annotations, and the batch ends on a boundary for one mode
    but not the other.
    """
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes = _kdes()
        # Spread each annotation around its mode's cluster so every expected
        # value differs: a result stored against the wrong annotation fails.
        batch = [
            (
                "positive" if index % 2 == 0 else "negative",
                (300.0 if index % 2 == 0 else 500.0) + 0.1 * index - 0.6,
                (0.9 if index % 2 == 0 else 0.5) + 0.005 * index - 0.03,
            )
            for index in range(13)
        ]
        annotation_ids = [
            _insert_annotation(
                session,
                ionization_mode=ionization_mode,
                ion_mass=ion_mass,
                csi_score=csi_score,
                inchikey=f"{index:014d}",
                run_suffix=f"batch-{index}",
            )
            for index, (ionization_mode, ion_mass, csi_score) in enumerate(batch)
        ]
        kde_model = KdeModel(
            artifact_path="/artifacts/kde-v1.pkl",
            subsample_seed=0,
            incorrect_subsample_size=20_000,
            positive_incorrect_rows_used=1,
            negative_incorrect_rows_used=1,
        )
        session.add(kde_model)
        session.flush()

        summary = apply_calibration(
            session,
            kdes,
            kde_model.kde_model_id,
            annotation_ids=annotation_ids,
            chunk_size=3,
        )

        assert summary.annotations_scored == len(batch)
        stored = {
            row.annotation_id: row.score
            for row in session.scalars(select(CalibrationScore))
        }
        assert set(stored) == set(annotation_ids)
        for annotation_id, (ionization_mode, ion_mass, csi_score) in zip(
            annotation_ids, batch, strict=True
        ):
            per_row_score = float(kdes[ionization_mode].pdf((ion_mass, csi_score))[0])
            assert stored[annotation_id] == pytest.approx(
                per_row_score, rel=1e-9, abs=0
            )


def test_second_kde_model_id_inserts_a_second_row_rather_than_overwriting() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes_v1 = _kdes()
        annotation_id = _insert_annotation(
            session,
            ionization_mode="positive",
            ion_mass=300.2,
            csi_score=0.9,
            inchikey="DDDDDDDDDDDDDD",
            run_suffix="orb",
        )
        kde_model_1 = KdeModel(
            artifact_path="/artifacts/kde-v1.pkl",
            subsample_seed=0,
            incorrect_subsample_size=20_000,
            positive_incorrect_rows_used=1,
            negative_incorrect_rows_used=1,
        )
        session.add(kde_model_1)
        session.flush()
        apply_calibration(
            session, kdes_v1, kde_model_1.kde_model_id, annotation_ids=[annotation_id]
        )

        # A distinctly different refit, so its pdf value at the same point
        # differs from kdes_v1's -- proving the first row survives untouched.
        kdes_v2 = {
            "positive": _kde([(310.0, 0.6), (311.0, 0.62), (309.5, 0.58)]),
            "negative": _kde([(500.0, 0.5), (501.0, 0.55), (499.0, 0.52)]),
        }
        kde_model_2 = KdeModel(
            artifact_path="/artifacts/kde-v2.pkl",
            subsample_seed=0,
            incorrect_subsample_size=20_000,
            positive_incorrect_rows_used=1,
            negative_incorrect_rows_used=1,
        )
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
            float(kdes_v1["positive"].pdf((300.2, 0.9))[0])
        )
        assert score_v2 == pytest.approx(
            float(kdes_v2["positive"].pdf((300.2, 0.9))[0])
        )
        assert score_v1 != pytest.approx(score_v2)


def test_same_kde_model_id_updates_the_existing_row_in_place() -> None:
    """Issue #25: annotate must be able to reprocess an unchanged, cache-hit
    batch against the same default kde_model_id without a primary-key
    conflict -- so a second `apply_calibration` call for the same
    `(annotation_id, kde_model_id)` pair overwrites the stored score rather
    than raising `IntegrityError`.
    """
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes_v1 = _kdes()
        annotation_id = _insert_annotation(
            session,
            ionization_mode="positive",
            ion_mass=300.2,
            csi_score=0.9,
            inchikey="GGGGGGGGGGGGGG",
            run_suffix="orb",
        )
        kde_model = KdeModel(
            artifact_path="/artifacts/kde-v1.pkl",
            subsample_seed=0,
            incorrect_subsample_size=20_000,
            positive_incorrect_rows_used=1,
            negative_incorrect_rows_used=1,
        )
        session.add(kde_model)
        session.flush()
        apply_calibration(
            session, kdes_v1, kde_model.kde_model_id, annotation_ids=[annotation_id]
        )

        # A distinctly different refit under the *same* kde_model_id, so a
        # changed pdf value at the same point proves the row was
        # overwritten in place rather than left stale or duplicated.
        kdes_v2 = {
            "positive": _kde([(310.0, 0.6), (311.0, 0.62), (309.5, 0.58)]),
            "negative": _kde([(500.0, 0.5), (501.0, 0.55), (499.0, 0.52)]),
        }

        summary = apply_calibration(
            session, kdes_v2, kde_model.kde_model_id, annotation_ids=[annotation_id]
        )

        assert summary.annotations_scored == 1
        rows = session.scalars(
            select(CalibrationScore).where(
                CalibrationScore.annotation_id == annotation_id
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].score == pytest.approx(
            float(kdes_v2["positive"].pdf((300.2, 0.9))[0])
        )
        assert rows[0].score != pytest.approx(
            float(kdes_v1["positive"].pdf((300.2, 0.9))[0])
        )


def test_only_requested_annotation_ids_are_scored_other_rows_untouched() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
        kdes = _kdes()
        included_id = _insert_annotation(
            session,
            ionization_mode="positive",
            ion_mass=300.2,
            csi_score=0.9,
            inchikey="EEEEEEEEEEEEEE",
            run_suffix="included",
        )
        _insert_annotation(
            session,
            ionization_mode="negative",
            ion_mass=500.2,
            csi_score=0.51,
            inchikey="FFFFFFFFFFFFFF",
            run_suffix="excluded",
        )
        kde_model = KdeModel(
            artifact_path="/artifacts/kde-v1.pkl",
            subsample_seed=0,
            incorrect_subsample_size=20_000,
            positive_incorrect_rows_used=1,
            negative_incorrect_rows_used=1,
        )
        session.add(kde_model)
        session.flush()

        summary = apply_calibration(
            session, kdes, kde_model.kde_model_id, annotation_ids=[included_id]
        )

        assert summary.annotations_scored == 1
        rows = session.scalars(select(CalibrationScore)).all()
        assert [row.annotation_id for row in rows] == [included_id]


def test_load_kdes_round_trips_a_pickled_two_density_artifact(tmp_path: Path) -> None:
    kdes = _kdes()
    artifact = {
        mode: ModeDensities(correct=kde, incorrect=kde) for mode, kde in kdes.items()
    }
    artifact_path = tmp_path / "kde.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(artifact, handle)

    loaded = load_kdes(artifact_path)

    assert set(loaded) == set(kdes)
    for mode_densities in loaded.values():
        assert isinstance(mode_densities.correct, gaussian_kde)
        assert isinstance(mode_densities.incorrect, gaussian_kde)

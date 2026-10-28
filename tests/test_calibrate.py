"""Tests for calibration application: densities evaluated in chunks.

Exercises that a chunked evaluation over a mixed-mode batch stores exactly the
per-row `logpdf` of its Ionization mode's densities, and that only the
requested annotations are calibrated -- against a real temp DuckDB and real
`scipy.stats.gaussian_kde` fits, per the project's testing decisions (no DB or
KDE mocking). Mode selection, versioning across `kde_model_id`s and the
exports are covered at the `annotate` CLI seam (`test_cli_annotate.py`).
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
    CalibrationDensities,
    Feature,
    KdeModel,
    Molecule,
    SiriusRun,
    create_database,
)


def _kde(points: list[tuple[float, float]]) -> gaussian_kde:
    return gaussian_kde(np.array(points).T)


def _densities() -> dict[str, ModeDensities]:
    positive_points = [(300.0, 0.9), (301.0, 0.85), (299.5, 0.88), (300.5, 0.92)]
    negative_points = [(500.0, 0.5), (501.0, 0.55), (499.0, 0.52), (500.5, 0.48)]
    other_points = [(310.0, 0.6), (311.0, 0.62), (309.5, 0.58), (310.5, 0.64)]
    incorrect = _kde(other_points)
    return {
        "positive": ModeDensities(correct=_kde(positive_points), incorrect=incorrect),
        "negative": ModeDensities(correct=_kde(negative_points), incorrect=incorrect),
    }


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
        densities = _densities()
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
            densities,
            kde_model.kde_model_id,
            annotation_ids=annotation_ids,
            chunk_size=3,
        )

        assert summary.annotations_calibrated == len(batch)
        stored = {
            row.annotation_id: (row.log_correct_density, row.log_incorrect_density)
            for row in session.scalars(select(CalibrationDensities))
        }
        assert set(stored) == set(annotation_ids)
        for annotation_id, (ionization_mode, ion_mass, csi_score) in zip(
            annotation_ids, batch, strict=True
        ):
            mode_densities = densities[ionization_mode]
            point = (ion_mass, csi_score)
            assert stored[annotation_id] == pytest.approx(
                (
                    float(mode_densities.correct.logpdf(point)[0]),
                    float(mode_densities.incorrect.logpdf(point)[0]),
                ),
                rel=1e-9,
                abs=0,
            )


def test_only_requested_annotation_ids_are_scored_other_rows_untouched() -> None:
    engine = create_database(":memory:")
    with Session(engine) as session:
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
            session, _densities(), kde_model.kde_model_id, annotation_ids=[included_id]
        )

        assert summary.annotations_calibrated == 1
        rows = session.scalars(select(CalibrationDensities)).all()
        assert [row.annotation_id for row in rows] == [included_id]


def test_load_kdes_round_trips_a_pickled_two_density_artifact(tmp_path: Path) -> None:
    densities = _densities()
    artifact_path = tmp_path / "kde.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump(densities, handle)

    loaded = load_kdes(artifact_path)

    assert set(loaded) == set(densities)
    for mode_densities in loaded.values():
        assert isinstance(mode_densities.correct, gaussian_kde)
        assert isinstance(mode_densities.incorrect, gaussian_kde)

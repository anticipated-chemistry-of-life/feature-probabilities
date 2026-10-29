"""Tests for the `evaluate-calibration` CLI.

A thin seam test against a real temp DuckDB seeded through the real
SQLAlchemy models; the evaluation maths is covered at the core seam in
`test_calibration_eval.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from sqlalchemy.orm import Session

from feature_probabilities.cli_evaluate_calibration import main
from feature_probabilities.schema import (
    SOURCE_KIND_GROUND_TRUTH_MSNLIB,
    Annotation,
    Feature,
    Molecule,
    SiriusRun,
    create_database,
)

_ADDUCT_BY_MODE = {"positive": "[M+H]+", "negative": "[M-H]-"}


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
db_path = "{db_path}"
required_sirius_version = "6.0.0"
ground_truth_revision = "abc123"
kde_output_path = "{tmp_path / "model.pkl"}"

[sirius]
top_k = 200
"""
    )
    return config_path


def _seed_ground_truth(session: Session) -> None:
    """Per mode, 15 Features of one correct and two incorrect candidates, 3 with no correct one."""
    rng = np.random.default_rng(0)
    for mode, adduct in _ADDUCT_BY_MODE.items():
        run = SiriusRun(
            extract_id=None,
            source_kind=SOURCE_KIND_GROUND_TRUTH_MSNLIB,
            input_file_checksum=f"checksum-{mode}",
            import_params_checksum=None,
            analysis_params_checksum=f"analysis-{mode}",
            sirius_version="6.0.0",
            pysirius_client_version="1.0.0",
            ionization_mode=mode,
            instrument_type="Orbitrap",
            input_file_path=f"/data/{mode}.mgf",
        )
        session.add(run)
        session.flush()
        for i in range(15):
            true_inchikey = f"{mode[0].upper()}{i:013d}"
            feature = Feature(
                run_id=run.run_id,
                external_feature_id=f"feature-{mode}-{i}",
                ion_mass=float(rng.uniform(200.0, 400.0)),
                charge=1 if mode == "positive" else -1,
                true_inchikey=true_inchikey,
            )
            session.add(feature)
            session.flush()
            candidates = [(f"{mode[0].upper()}X{j}{i:011d}", 0.2) for j in range(2)]
            if i >= 3:
                candidates.append((true_inchikey, 0.8))
            for rank, (inchikey, score) in enumerate(candidates, start=1):
                molecule = Molecule(
                    inchikey=inchikey, smiles="CCO", molecular_formula="C2H6O"
                )
                session.add(molecule)
                session.flush()
                session.add(
                    Annotation(
                        feature_id=feature.feature_id,
                        molecule_id=molecule.molecule_id,
                        rank=rank,
                        csi_score=float(rng.normal(score, 0.3)),
                        tanimoto_similarity=1.0,
                        mces_dist_to_top_hit=0.0,
                        xlogp=1.0,
                        adduct=adduct,
                        formula_id="formula-1",
                    )
                )
    session.commit()


def _seeded_db(tmp_path: Path) -> tuple[Path, Path]:
    db_path = tmp_path / "database.duckdb"
    engine = create_database(db_path)
    with Session(engine) as session:
        _seed_ground_truth(session)
    engine.dispose()
    return _write_config(tmp_path, db_path), db_path


def test_prints_both_methods_and_leaves_the_database_and_directory_untouched(
    tmp_path: Path,
) -> None:
    config_path, db_path = _seeded_db(tmp_path)
    db_bytes = db_path.read_bytes()
    files = sorted(p.name for p in tmp_path.iterdir())

    result = CliRunner().invoke(main, ["--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "baseline" in result.output
    assert "likelihood ratio" in result.output
    assert "top-1 accuracy" in result.output
    assert db_path.read_bytes() == db_bytes
    assert sorted(p.name for p in tmp_path.iterdir()) == files


@pytest.mark.parametrize("create_empty_database", [True, False])
def test_without_ground_truth_exits_nonzero_and_points_at_generate_groundtruth(
    tmp_path: Path, create_empty_database: bool
) -> None:
    db_path = tmp_path / "database.duckdb"
    if create_empty_database:
        create_database(db_path).dispose()
    config_path = _write_config(tmp_path, db_path)
    files = sorted(p.name for p in tmp_path.iterdir())

    result = CliRunner().invoke(main, ["--config", str(config_path)])

    assert result.exit_code != 0
    assert "generate-groundtruth" in result.output
    assert sorted(p.name for p in tmp_path.iterdir()) == files

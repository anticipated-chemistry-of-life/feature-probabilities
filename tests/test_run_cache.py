"""Tests for the SIRIUS run cache lookup and persistence.

Exercises `FakeSirius` (the one seam, per issue #9's Testing Decisions) plus
a real (in-memory) DuckDB engine -- no further mocking, per the project's
testing decisions.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySirius import (
    AlignedFeature,
    JobSubmission,
    LcmsSubmissionParameters,
    StructureCandidateFormula,
)
from rdkit import Chem
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.run_cache import (
    RunCacheError,
    SiriusRunRequest,
    get_or_create_run,
)
from feature_probabilities.schema import (
    SOURCE_KIND_FIELD_MZML,
    Feature,
    Molecule,
    SiriusRun,
    create_database,
)
from feature_probabilities.sirius import FeatureStructureCandidate
from feature_probabilities.sirius_fake import FakeSirius


def _write_input_file(
    tmp_path: Path, name: str = "input.mzml", content: bytes = b"spectra bytes"
) -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def _request(
    tmp_path: Path,
    input_file: Path,
    *,
    analysis_params: JobSubmission | None = None,
    import_params: LcmsSubmissionParameters | None = None,
    sirius_version: str = "6.0.0",
) -> SiriusRunRequest:
    return SiriusRunRequest(
        input_file=input_file,
        project_path=tmp_path / "project.sirius",
        source_kind=SOURCE_KIND_FIELD_MZML,
        extract_id=None,
        import_params=import_params,
        analysis_params=analysis_params
        if analysis_params is not None
        else JobSubmission(recompute=False),
        sirius_version=sirius_version,
        pysirius_client_version="1.0.0",
        ionization_mode="positive",
        instrument_type="Orbitrap",
    )


def _feature(feature_id: str, ion_mass: float) -> AlignedFeature:
    return AlignedFeature(
        aligned_feature_id=feature_id,
        external_feature_id=feature_id,
        ion_mass=ion_mass,
        charge=1,
        detected_adducts=["[M+H]+"],
    )


def _candidate(
    inchi_key: str, smiles: str = "CCO", **overrides: object
) -> StructureCandidateFormula:
    fields = {
        "inchi_key": inchi_key,
        "smiles": smiles,
        "rank": 1,
        "csi_score": -100.0,
        "tanimoto_similarity": 0.9,
        "mces_dist_to_top_hit": 0.0,
        "molecular_formula": "C2H6O",
        "adduct": "[M+H]+",
        "formula_id": "formula-1",
        "xlog_p": 0.5,
    }
    fields.update(overrides)
    return StructureCandidateFormula(**fields)


def _fake_for(input_file: Path) -> FakeSirius:
    """A `FakeSirius` whose canned candidate carries every NOT NULL annotation field."""
    feature = _feature("feat-1", ion_mass=301.1234)
    return FakeSirius(
        canned_features={input_file: [feature]},
        canned_results={
            input_file: [
                FeatureStructureCandidate(
                    feature=feature, candidate=_candidate("AXFAVZQXPFQIEI")
                )
            ]
        },
    )


def test_first_time_processing_calls_the_wrapper_and_persists_expected_rows(
    tmp_path: Path,
) -> None:
    input_file = _write_input_file(tmp_path)
    fake = _fake_for(input_file)
    engine = create_database(":memory:")
    with Session(engine) as session:
        result = get_or_create_run(session, fake, _request(tmp_path, input_file))
        session.commit()

        assert result.from_cache is False
        assert len(fake.import_spectra_calls) == 1
        assert len(fake.run_calls) == 1
        assert len(result.features) == 1
        assert len(result.annotations) == 1

        assert session.scalars(select(SiriusRun)).all() != []
        molecule = session.scalars(select(Molecule)).one()
        assert molecule.inchikey == "AXFAVZQXPFQIEI"


def test_second_identical_call_makes_zero_wrapper_calls_and_returns_same_run(
    tmp_path: Path,
) -> None:
    input_file = _write_input_file(tmp_path)
    fake = _fake_for(input_file)
    engine = create_database(":memory:")
    with Session(engine) as session:
        request = _request(tmp_path, input_file)
        first = get_or_create_run(session, fake, request)
        session.commit()

        second = get_or_create_run(session, fake, request)
        session.commit()

        assert len(fake.import_spectra_calls) == 1
        assert len(fake.run_calls) == 1
        assert second.from_cache is True
        assert second.run.run_id == first.run.run_id
        assert [f.feature_id for f in second.features] == [
            f.feature_id for f in first.features
        ]


def test_changing_input_file_bytes_forces_a_new_coexisting_run(tmp_path: Path) -> None:
    input_file = _write_input_file(tmp_path)
    fake = _fake_for(input_file)
    engine = create_database(":memory:")
    with Session(engine) as session:
        request = _request(tmp_path, input_file)
        first = get_or_create_run(session, fake, request)
        session.commit()

        input_file.write_bytes(b"different bytes entirely")
        second = get_or_create_run(session, fake, request)
        session.commit()

        assert len(fake.import_spectra_calls) == 2
        assert second.from_cache is False
        assert second.run.run_id != first.run.run_id
        assert len(session.scalars(select(SiriusRun)).all()) == 2


def test_changing_import_params_checksum_forces_a_new_coexisting_run(
    tmp_path: Path,
) -> None:
    input_file = _write_input_file(tmp_path)
    fake = _fake_for(input_file)
    engine = create_database(":memory:")
    with Session(engine) as session:
        request_a = _request(
            tmp_path, input_file, import_params=LcmsSubmissionParameters(min_snr=3)
        )
        first = get_or_create_run(session, fake, request_a)
        session.commit()

        request_b = _request(
            tmp_path, input_file, import_params=LcmsSubmissionParameters(min_snr=10)
        )
        second = get_or_create_run(session, fake, request_b)
        session.commit()

        assert len(fake.import_spectra_calls) == 2
        assert second.from_cache is False
        assert second.run.run_id != first.run.run_id


def test_changing_analysis_params_checksum_forces_a_new_coexisting_run(
    tmp_path: Path,
) -> None:
    input_file = _write_input_file(tmp_path)
    fake = _fake_for(input_file)
    engine = create_database(":memory:")
    with Session(engine) as session:
        request_a = _request(
            tmp_path,
            input_file,
            analysis_params=JobSubmission(
                formula_id_params={"enabled": True, "number_of_candidates": 10}
            ),
        )
        first = get_or_create_run(session, fake, request_a)
        session.commit()

        request_b = _request(
            tmp_path,
            input_file,
            analysis_params=JobSubmission(
                formula_id_params={"enabled": True, "number_of_candidates": 25}
            ),
        )
        second = get_or_create_run(session, fake, request_b)
        session.commit()

        assert len(fake.run_calls) == 2
        assert second.from_cache is False
        assert second.run.run_id != first.run.run_id


def test_a_forced_reruns_row_is_still_found_by_a_later_unforced_run(
    tmp_path: Path,
) -> None:
    """`recompute` must not fork the cache.

    It is set from `--force` and controls whether SIRIUS redoes work, not
    what it computes, so a forced run's row has to stay reachable -- else
    one `--force` makes every later invocation recompute from scratch and
    append a duplicate set of feature/annotation rows.
    """
    input_file = _write_input_file(tmp_path)
    fake = _fake_for(input_file)
    engine = create_database(":memory:")
    with Session(engine) as session:
        forced = _request(
            tmp_path, input_file, analysis_params=JobSubmission(recompute=True)
        )
        first = get_or_create_run(session, fake, forced, force=True)
        session.commit()

        unforced = _request(
            tmp_path, input_file, analysis_params=JobSubmission(recompute=False)
        )
        second = get_or_create_run(session, fake, unforced)
        session.commit()

        assert second.from_cache is True
        assert second.run.run_id == first.run.run_id
        assert len(fake.run_calls) == 1
        assert len(session.scalars(select(Feature)).all()) == len(first.features)


def test_changing_sirius_version_forces_a_new_coexisting_run(tmp_path: Path) -> None:
    input_file = _write_input_file(tmp_path)
    fake = _fake_for(input_file)
    engine = create_database(":memory:")
    with Session(engine) as session:
        request_a = _request(tmp_path, input_file, sirius_version="6.0.0")
        first = get_or_create_run(session, fake, request_a)
        session.commit()

        request_b = _request(tmp_path, input_file, sirius_version="6.1.0")
        second = get_or_create_run(session, fake, request_b)
        session.commit()

        assert len(fake.import_spectra_calls) == 2
        assert second.from_cache is False
        assert second.run.run_id != first.run.run_id


def test_force_true_calls_the_wrapper_even_with_a_matching_row(tmp_path: Path) -> None:
    input_file = _write_input_file(tmp_path)
    fake = _fake_for(input_file)
    engine = create_database(":memory:")
    with Session(engine) as session:
        request = _request(tmp_path, input_file)
        first = get_or_create_run(session, fake, request)
        session.commit()

        second = get_or_create_run(session, fake, request, force=True)
        session.commit()

        assert len(fake.import_spectra_calls) == 2
        assert len(fake.run_calls) == 2
        assert second.from_cache is False
        assert second.run.run_id != first.run.run_id


def test_molecules_are_deduplicated_globally_by_inchikey(tmp_path: Path) -> None:
    input_file = _write_input_file(tmp_path)
    feature_a = _feature("feat-a", ion_mass=100.0)
    feature_b = _feature("feat-b", ion_mass=200.0)
    fake = FakeSirius(
        canned_features={input_file: [feature_a, feature_b]},
        canned_results={
            input_file: [
                FeatureStructureCandidate(
                    feature=feature_a, candidate=_candidate("SAMEINCHIKEY01")
                ),
                FeatureStructureCandidate(
                    feature=feature_b, candidate=_candidate("SAMEINCHIKEY01")
                ),
            ]
        },
    )
    engine = create_database(":memory:")
    with Session(engine) as session:
        result = get_or_create_run(session, fake, _request(tmp_path, input_file))
        session.commit()

        assert len(result.annotations) == 2
        molecules = session.scalars(select(Molecule)).all()
        assert len(molecules) == 1
        assert molecules[0].inchikey == "SAMEINCHIKEY01"


def test_molecules_are_deduplicated_globally_across_separate_runs(
    tmp_path: Path,
) -> None:
    input_file_a = _write_input_file(tmp_path, name="a.mzml", content=b"aaa")
    input_file_b = _write_input_file(tmp_path, name="b.mzml", content=b"bbb")
    feature_a = _feature("feat-a", ion_mass=100.0)
    feature_b = _feature("feat-b", ion_mass=200.0)
    fake = FakeSirius(
        canned_features={input_file_a: [feature_a], input_file_b: [feature_b]},
        canned_results={
            input_file_a: [
                FeatureStructureCandidate(
                    feature=feature_a, candidate=_candidate("SAMEINCHIKEY01")
                )
            ],
            input_file_b: [
                FeatureStructureCandidate(
                    feature=feature_b, candidate=_candidate("SAMEINCHIKEY01")
                )
            ],
        },
    )
    engine = create_database(":memory:")
    with Session(engine) as session:
        get_or_create_run(session, fake, _request(tmp_path, input_file_a))
        session.commit()
        get_or_create_run(session, fake, _request(tmp_path, input_file_b))
        session.commit()

        molecules = session.scalars(select(Molecule)).all()
        assert len(molecules) == 1


def test_newly_inserted_molecule_smiles_is_rdkit_canonicalized_and_stereo_stripped(
    tmp_path: Path,
) -> None:
    input_file = _write_input_file(tmp_path)
    feature = _feature("feat-1", ion_mass=100.0)
    raw_stereo_smiles = "C[C@H](N)C(=O)O"
    fake = FakeSirius(
        canned_features={input_file: [feature]},
        canned_results={
            input_file: [
                FeatureStructureCandidate(
                    feature=feature,
                    candidate=_candidate("ALANINEKEY0001", smiles=raw_stereo_smiles),
                )
            ]
        },
    )
    engine = create_database(":memory:")
    with Session(engine) as session:
        get_or_create_run(session, fake, _request(tmp_path, input_file))
        session.commit()

        molecule = session.scalars(select(Molecule)).one()
        assert molecule.smiles != raw_stereo_smiles
        assert "@" not in molecule.smiles

        expected = Chem.MolFromSmiles(raw_stereo_smiles)
        Chem.RemoveStereochemistry(expected)
        assert molecule.smiles == Chem.MolToSmiles(expected, canonical=True)


def test_unparseable_smiles_raises_a_clear_error(tmp_path: Path) -> None:
    input_file = _write_input_file(tmp_path)
    feature = _feature("feat-1", ion_mass=100.0)
    fake = FakeSirius(
        canned_features={input_file: [feature]},
        canned_results={
            input_file: [
                FeatureStructureCandidate(
                    feature=feature,
                    candidate=_candidate("BADSMILESKEY01", smiles="not-a-smiles((("),
                )
            ]
        },
    )
    engine = create_database(":memory:")
    with Session(engine) as session, pytest.raises(RunCacheError):
        get_or_create_run(session, fake, _request(tmp_path, input_file))

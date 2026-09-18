"""Tests for `FakeSirius`, the seam every SIRIUS-touching ticket's tests build on."""

from __future__ import annotations

from pathlib import Path

import pytest
from PySirius import (
    AlignedFeature,
    JobSubmission,
    LcmsSubmissionParameters,
    StructureCandidateFormula,
)

from feature_probabilities.sirius import (
    FeatureStructureCandidate,
    NoActiveProjectError,
    SiriusInterface,
)
from feature_probabilities.sirius_fake import FakeSirius


def test_fake_sirius_structurally_satisfies_sirius_interface() -> None:
    fake = FakeSirius()

    assert isinstance(fake, SiriusInterface)


def test_pre_picked_import_passes_no_lcms_submission_parameters() -> None:
    fake = FakeSirius()
    fake.create_project(Path("project"))

    fake.import_spectra(Path("chunk.mgf"))

    spectra_file, params = fake.import_spectra_calls[0]
    assert spectra_file == Path("chunk.mgf")
    assert params is None


def test_mzml_import_passes_lcms_submission_parameters() -> None:
    fake = FakeSirius()
    fake.create_project(Path("project"))
    lcms_params = LcmsSubmissionParameters(min_snr=2.0)

    fake.import_spectra(Path("sample.mzml"), params=lcms_params)

    spectra_file, params = fake.import_spectra_calls[0]
    assert spectra_file == Path("sample.mzml")
    assert params is lcms_params


def test_pre_picked_and_mzml_imports_are_recorded_as_distinct_calls() -> None:
    fake = FakeSirius()
    fake.create_project(Path("project"))

    fake.import_spectra(Path("chunk.mgf"))
    fake.import_spectra(Path("sample.mzml"), params=LcmsSubmissionParameters())

    assert len(fake.import_spectra_calls) == 2
    assert fake.import_spectra_calls[0][1] is None
    assert isinstance(fake.import_spectra_calls[1][1], LcmsSubmissionParameters)


def test_run_submits_a_job_submission_shaped_object() -> None:
    fake = FakeSirius()
    fake.create_project(Path("project"))
    job_submission = JobSubmission(recompute=True)

    fake.run(job_submission)

    assert len(fake.run_calls) == 1
    assert isinstance(fake.run_calls[0], JobSubmission)
    assert fake.run_calls[0].recompute is True


def test_structure_candidates_carry_both_csi_score_and_ion_mass_on_one_row() -> None:
    fake = FakeSirius()
    fake.create_project(Path("project"))
    fake.import_spectra(Path("chunk.mgf"))
    fake.run(JobSubmission())

    rows = fake.get_structure_candidates()

    assert len(rows) >= 1
    row = rows[0]
    assert isinstance(row, FeatureStructureCandidate)
    assert row.candidate.csi_score is not None
    assert row.ion_mass is not None
    assert row.ion_mass == row.feature.ion_mass


def test_structure_candidates_use_the_canned_result_configured_for_the_imported_file() -> (
    None
):
    feature_a = _feature("feature-a", ion_mass=100.0)
    feature_b = _feature("feature-b", ion_mass=200.0)
    fake = FakeSirius(
        canned_results={
            Path("a.mgf"): [
                FeatureStructureCandidate(
                    feature=feature_a, candidate=_candidate("AAAAAAAAAAAAAA")
                )
            ],
            Path("b.mgf"): [
                FeatureStructureCandidate(
                    feature=feature_b, candidate=_candidate("BBBBBBBBBBBBBB")
                )
            ],
        }
    )
    fake.create_project(Path("project"))

    fake.import_spectra(Path("a.mgf"))
    fake.run(JobSubmission())
    rows_a = fake.get_structure_candidates()

    fake.import_spectra(Path("b.mgf"))
    fake.run(JobSubmission())
    rows_b = fake.get_structure_candidates()

    assert [row.candidate.inchi_key for row in rows_a] == ["AAAAAAAAAAAAAA"]
    assert [row.candidate.inchi_key for row in rows_b] == ["BBBBBBBBBBBBBB"]


def test_get_features_returns_default_features_for_an_unconfigured_file() -> None:
    fake = FakeSirius()
    fake.create_project(Path("project"))
    fake.import_spectra(Path("chunk.mgf"))

    features = fake.get_features()

    assert [feature.aligned_feature_id for feature in features] == ["fake-feature-1"]


def test_get_features_reaches_a_feature_with_zero_structure_candidates() -> None:
    candidate_free_feature = _feature("candidate-free", ion_mass=150.0)
    fake = FakeSirius(
        canned_features={Path("a.mgf"): [candidate_free_feature]},
        canned_results={Path("a.mgf"): []},
    )
    fake.create_project(Path("project"))
    fake.import_spectra(Path("a.mgf"))

    features = fake.get_features()
    rows = fake.get_structure_candidates()

    assert [feature.aligned_feature_id for feature in features] == ["candidate-free"]
    assert rows == []


def test_get_version_returns_the_configured_version_string() -> None:
    fake = FakeSirius(version="6.6.6")

    assert fake.get_version() == "6.6.6"


def test_get_version_defaults_without_requiring_a_project() -> None:
    fake = FakeSirius()

    assert fake.get_version()


@pytest.mark.parametrize(
    "call",
    [
        lambda fake: fake.import_spectra(Path("chunk.mgf")),
        lambda fake: fake.run(JobSubmission()),
        lambda fake: fake.get_features(),
        lambda fake: fake.get_structure_candidates(),
    ],
)
def test_project_scoped_calls_before_create_project_raise(call) -> None:
    fake = FakeSirius()

    with pytest.raises(NoActiveProjectError):
        call(fake)


def test_zero_calls_recorded_before_any_interaction() -> None:
    fake = FakeSirius()

    assert fake.create_project_calls == []
    assert fake.import_spectra_calls == []
    assert fake.run_calls == []


def _feature(feature_id: str, ion_mass: float) -> AlignedFeature:
    return AlignedFeature(
        aligned_feature_id=feature_id,
        external_feature_id=feature_id,
        ion_mass=ion_mass,
        charge=1,
        detected_adducts=["[M+H]+"],
    )


def _candidate(inchi_key: str) -> StructureCandidateFormula:
    return StructureCandidateFormula(inchi_key=inchi_key, csi_score=-100.0, rank=1)

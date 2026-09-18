"""Standalone `Sirius` wrapper: core orchestration against the raw PySirius client.

Covers exactly the SIRIUS interaction itself -- attach-or-start plus login,
project lifecycle, importing spectra (both the pre-picked and mzML/peak-
picking paths), submitting an analysis job, and reading back structure
candidates joined with their feature's ion mass. It deliberately does **not**
know about rerun-avoidance caching or DuckDB persistence (`sirius_runs`
cache lookups and `features`/`annotations`/`molecules` row persistence): that
ties this wrapper together with `checksums.py` and `schema.py` in a later
ticket (#19). This module never imports either.

`SiriusInterface` is the structural contract (a `typing.Protocol`) both this
module's real `Sirius` class and `sirius_fake.FakeSirius` satisfy -- the one
seam every later SIRIUS-touching ticket's tests are built on, since SIRIUS
requires a licensed, locally-installed desktop application and cannot run in
CI. See CONTEXT.md's Feature, Structure candidate, and Annotation entries for
the domain vocabulary `FeatureStructureCandidate` encodes, and issue #6's
research findings for why `get_structure_candidates` needs a manual join
(`StructureCandidateFormula` doesn't carry ion mass; it lives only on
`AlignedFeature`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from PySirius import (
    AccountCredentials,
    AlignedFeature,
    JobSubmission,
    LcmsSubmissionParameters,
    ProjectInfo,
    PySiriusAPI,
    SiriusSDK,
    StructureCandidateFormula,
)


class NoActiveProjectError(RuntimeError):
    """Raised when a project-scoped method is called before `create_project`."""


#: Shared by `Sirius._require_project_id` and `FakeSirius._require_project` so
#: the two `SiriusInterface` implementations raise byte-for-byte identical
#: `NoActiveProjectError` messages.
NO_ACTIVE_PROJECT_MESSAGE = "create_project must be called before this operation."


class SiriusCredentialsError(RuntimeError):
    """Raised when `SIRIUS_USER`/`SIRIUS_PW` aren't set for the real wrapper to log in."""


class SiriusStartupError(RuntimeError):
    """Raised when SIRIUS can't be attached to or started."""


class SiriusVersionUnavailableError(RuntimeError):
    """Raised when a running SIRIUS instance's `get_info()` reports no version string."""


@dataclass(frozen=True, slots=True)
class FeatureStructureCandidate:
    """One feature-structure-candidate pair: a candidate joined with its feature.

    PySirius's `StructureCandidateFormula` does not carry ion mass natively --
    `get_structure_candidates` joins it in from `AlignedFeature` by feature id
    (CONTEXT.md's Annotation entry). Deliberately not named `Annotation` to
    avoid confusion with `schema.Annotation`, the persisted DB row a later
    ticket (#19) derives from rows of this type -- this is the wrapper's
    in-memory return type, not a DB row.
    """

    feature: AlignedFeature
    candidate: StructureCandidateFormula

    @property
    def ion_mass(self) -> float | None:
        """The feature's precursor ion mass, joined in from `feature`."""
        return self.feature.ion_mass


@runtime_checkable
class SiriusInterface(Protocol):
    """Structural contract both the real `Sirius` wrapper and `FakeSirius` satisfy.

    Attaching to SIRIUS and logging in happen as a side effect of
    construction, not through a method on this Protocol -- a `FakeSirius`
    construction makes no SIRIUS connection at all, so there is nothing to
    protocol-ize there. Everything past construction is captured here, so
    swapping one implementation for the other requires no caller-side code
    changes.
    """

    def get_version(self) -> str:
        """The running SIRIUS application's version string (via `get_info()`).

        Callers compare this against a required version to fail fast on a
        mismatch; that comparison itself belongs to later CLI tickets.
        """
        ...

    def create_project(
        self, project_path: Path, project_name: str | None = None
    ) -> None:
        """Create a SIRIUS project-space at `project_path` and make it current.

        `project_name` defaults to `project_path.stem`. Every other method on
        this Protocol operates on whichever project was created most recently.
        """
        ...

    def import_spectra(
        self, spectra_file: Path, params: LcmsSubmissionParameters | None = None
    ) -> None:
        """Import `spectra_file` into the current project and wait for completion.

        `params is None` takes the pre-picked import path (MGF/MS/CEF/MSP --
        already feature-level, no peak-picking parameters constructed or
        passed). A supplied `LcmsSubmissionParameters` takes the mzML/mzXML
        peak-picking import path instead.
        """
        ...

    def get_features(self) -> list[AlignedFeature]:
        """Every aligned feature for the current project, independent of whether it matched any structure candidate.

        Exposed as its own seam member per issue #9's Testing Decisions (the
        Sirius wrapper's interface explicitly lists `get_aligned_features`
        alongside `get_structure_candidates`), so a candidate-less feature is
        still reachable -- e.g. for persisting the `features` table's rows
        independently of `annotations` (issue #19).
        """
        ...

    def run(self, job_submission: JobSubmission) -> None:
        """Submit `job_submission` for analysis against the current project and wait for it."""
        ...

    def get_structure_candidates(self) -> list[FeatureStructureCandidate]:
        """Every (feature, structure candidate) pair for the current project.

        Joins each feature from `get_features` with its own
        `get_structure_candidates` result, since `StructureCandidateFormula`
        does not carry ion mass on its own -- a feature with zero structure
        candidates contributes no rows here (see `get_features` for those).
        """
        ...


def _credentials_from_env() -> AccountCredentials:
    username = os.getenv("SIRIUS_USER")
    password = os.getenv("SIRIUS_PW")
    if not username or not password:
        raise SiriusCredentialsError(
            "SIRIUS_USER and SIRIUS_PW environment variables must both be set "
            "(e.g. in a .env file) to log into a SIRIUS account."
        )
    return AccountCredentials(username=username, password=password)


class Sirius:
    """Orchestrates a real, locally-running SIRIUS desktop application via PySirius.

    Attaches to (or starts) SIRIUS and logs in immediately on construction,
    mirroring `metabolite_annotator`'s `Sirius` wrapper. Implements
    `SiriusInterface`; use `sirius_fake.FakeSirius` instead of this class in
    every automated test, since SIRIUS requires a licensed, locally-installed
    desktop app unavailable in CI.
    """

    def __init__(self, *, headless: bool = True) -> None:
        sdk = SiriusSDK()
        api = sdk.attach_or_start_sirius(headless=headless)
        if api is None:
            raise SiriusStartupError("Failed to attach to or start a SIRIUS instance.")
        self._sdk = sdk
        self._api: PySiriusAPI = api
        self._api.account().login(True, _credentials_from_env())
        self._project_info: ProjectInfo | None = None

    def get_version(self) -> str:
        info = self._api.infos().get_info()
        if not info.sirius_version:
            raise SiriusVersionUnavailableError(
                "SIRIUS did not report a version via get_info()."
            )
        return info.sirius_version

    def create_project(
        self, project_path: Path, project_name: str | None = None
    ) -> None:
        name = project_name or project_path.stem
        self._project_info = self._api.projects().create_project(
            project_id=name, path_to_project=str(project_path.resolve())
        )

    def import_spectra(
        self, spectra_file: Path, params: LcmsSubmissionParameters | None = None
    ) -> None:
        project_id = self._require_project_id()
        input_files = [str(spectra_file.resolve())]
        if params is None:
            job = self._api.projects().import_preprocessed_data_as_job(
                project_id, input_files=input_files
            )
        else:
            job = self._api.projects().import_ms_run_data_as_job(
                project_id, input_files=input_files, parameters=params
            )
        self._api.wait_for_job_completion(self._project_info, job)

    def run(self, job_submission: JobSubmission) -> None:
        project_id = self._require_project_id()
        job = self._api.jobs().start_job(project_id, job_submission)
        self._api.wait_for_job_completion(self._project_info, job)

    def get_features(self) -> list[AlignedFeature]:
        project_id = self._require_project_id()
        return self._api.features().get_aligned_features(project_id)

    def get_structure_candidates(self) -> list[FeatureStructureCandidate]:
        project_id = self._require_project_id()
        features_api = self._api.features()
        rows: list[FeatureStructureCandidate] = []
        for feature in self.get_features():
            candidates = features_api.get_structure_candidates(
                project_id, feature.aligned_feature_id
            )
            rows.extend(
                FeatureStructureCandidate(feature=feature, candidate=candidate)
                for candidate in candidates
            )
        return rows

    def _require_project_id(self) -> str:
        if self._project_info is None:
            raise NoActiveProjectError(NO_ACTIVE_PROJECT_MESSAGE)
        return self._project_info.project_id

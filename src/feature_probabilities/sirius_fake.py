"""`FakeSirius`: the one seam every SIRIUS-touching ticket's tests are built on.

SIRIUS requires a licensed, locally-installed desktop application and cannot
run in CI, so no automated test talks to the real `sirius.Sirius`. Instead,
`FakeSirius` implements `sirius.SiriusInterface` structurally (see
`test_sirius_fake.py`'s Protocol conformance check) and returns canned,
`AlignedFeature`/`StructureCandidateFormula`-shaped PySirius model instances --
the very same pydantic models the real client returns -- instead of talking to
a real SIRIUS process.

`get_features` and `get_structure_candidates` results are each keyed by the
spectra file path passed to `import_spectra` (`canned_features`/
`canned_results`), mirroring how a real SIRIUS project's results actually
depend on what was imported into it: a caller processing several
chunks/mzML files against one long-lived `FakeSirius` instance (as
`fp-generate-groundtruth`/`fp-annotate` do -- one project per input file)
can configure a distinct canned result set per input file. Any file with no
specific entry falls back to `default_features`/`default_results`, so a
test that doesn't care about the exact data still gets a plausible,
non-empty result. The two are configured independently, matching real
SIRIUS: a feature can exist (`get_features`) with zero structure candidates
(absent from `get_structure_candidates`).

Every call that would talk to a real SIRIUS process is recorded
(`create_project_calls`, `import_spectra_calls`, `run_calls`), so a test can
assert a cache hit made zero such calls without inspecting any other
internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySirius import (
    AlignedFeature,
    JobSubmission,
    LcmsSubmissionParameters,
    StructureCandidateFormula,
)

from .sirius import (
    NO_ACTIVE_PROJECT_MESSAGE,
    FeatureStructureCandidate,
    NoActiveProjectError,
)

#: Version string `FakeSirius` reports by default; override via the
#: `version` field to exercise a version-mismatch fail-fast path.
DEFAULT_VERSION = "6.0.0"

_DEFAULT_FEATURE = AlignedFeature(
    aligned_feature_id="fake-feature-1",
    external_feature_id="fake-feature-1",
    ion_mass=301.1234,
    charge=1,
    detected_adducts=["[M+H]+"],
)

#: A default canned result set: one feature with one structure candidate,
#: used for any spectra file not given a specific entry in `canned_results`.
DEFAULT_RESULTS: tuple[FeatureStructureCandidate, ...] = (
    FeatureStructureCandidate(
        feature=_DEFAULT_FEATURE,
        candidate=StructureCandidateFormula(
            inchi_key="AXFAVZQXPFQIEI",
            smiles="CCO",
            rank=1,
            csi_score=-120.5,
            tanimoto_similarity=0.91,
            mces_dist_to_top_hit=0.0,
            molecular_formula="C2H6O",
            adduct="[M+H]+",
            formula_id="fake-formula-1",
        ),
    ),
)


@dataclass(slots=True)
class FakeSirius:
    """In-memory, canned-data test double satisfying `sirius.SiriusInterface`."""

    version: str = DEFAULT_VERSION
    canned_features: dict[Path, list[AlignedFeature]] = field(default_factory=dict)
    default_features: list[AlignedFeature] = field(
        default_factory=lambda: [_DEFAULT_FEATURE]
    )
    canned_results: dict[Path, list[FeatureStructureCandidate]] = field(
        default_factory=dict
    )
    default_results: list[FeatureStructureCandidate] = field(
        default_factory=lambda: list(DEFAULT_RESULTS)
    )

    #: Every `(project_path, project_name)` passed to `create_project`, in order.
    create_project_calls: list[tuple[Path, str | None]] = field(
        default_factory=list, init=False
    )
    #: Every `(spectra_file, params)` passed to `import_spectra`, in order.
    #: `params is None` marks a pre-picked-path call.
    import_spectra_calls: list[tuple[Path, LcmsSubmissionParameters | None]] = field(
        default_factory=list, init=False
    )
    #: Every `JobSubmission` passed to `run`, in order.
    run_calls: list[JobSubmission] = field(default_factory=list, init=False)

    _current_spectra_file: Path | None = field(default=None, init=False, repr=False)
    _has_project: bool = field(default=False, init=False, repr=False)

    def get_version(self) -> str:
        return self.version

    def create_project(
        self, project_path: Path, project_name: str | None = None
    ) -> None:
        self.create_project_calls.append((Path(project_path), project_name))
        self._has_project = True
        self._current_spectra_file = None

    def import_spectra(
        self, spectra_file: Path, params: LcmsSubmissionParameters | None = None
    ) -> None:
        self._require_project()
        spectra_path = Path(spectra_file)
        self.import_spectra_calls.append((spectra_path, params))
        self._current_spectra_file = spectra_path

    def run(self, job_submission: JobSubmission) -> None:
        self._require_project()
        self.run_calls.append(job_submission)

    def get_features(self) -> list[AlignedFeature]:
        self._require_project()
        if self._current_spectra_file is not None:
            canned = self.canned_features.get(self._current_spectra_file)
            if canned is not None:
                return list(canned)
        return list(self.default_features)

    def get_structure_candidates(self) -> list[FeatureStructureCandidate]:
        self._require_project()
        if self._current_spectra_file is not None:
            canned = self.canned_results.get(self._current_spectra_file)
            if canned is not None:
                return list(canned)
        return list(self.default_results)

    def _require_project(self) -> None:
        if not self._has_project:
            raise NoActiveProjectError(NO_ACTIVE_PROJECT_MESSAGE)

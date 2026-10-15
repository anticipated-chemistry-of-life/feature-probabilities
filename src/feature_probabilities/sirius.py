"""Standalone `Sirius` wrapper: core orchestration against the raw PySirius client.

Covers exactly the SIRIUS interaction itself -- attach-or-start plus login,
shutdown (and lazy restart after it), project lifecycle, importing spectra
(both the pre-picked and mzML/peak-picking paths), submitting an analysis
job (with a stall watchdog that kills a deadlocked SIRIUS), and reading back
each feature's Top-k structure candidates joined with its ion mass. It
deliberately does **not**
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

import contextlib
import os
import signal
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from dotenv import load_dotenv
from PySirius import (
    AccountCredentials,
    AlignedFeature,
    JobProgress,
    JobState,
    JobSubmission,
    LcmsSubmissionParameters,
    ProjectInfo,
    PySiriusAPI,
    SiriusSDK,
    StructureCandidateFormula,
    UseHeuristic,
)

if TYPE_CHECKING:
    from feature_probabilities.config import TOMLTable


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


class SiriusShutdownError(RuntimeError):
    """Raised when a SIRIUS process is still alive after being told to shut down."""


class SiriusJobFailedError(RuntimeError):
    """Raised when SIRIUS reports an analysis job as FAILED or CANCELED."""


class SiriusJobStalledError(RuntimeError):
    """Raised when an analysis job's progress stops advancing.

    By the time this propagates, the stalled SIRIUS process has been
    killed; the next call that needs SIRIUS starts a fresh instance.
    """


class SiriusVersionUnavailableError(RuntimeError):
    """Raised when a running SIRIUS instance's `get_info()` reports no version string."""


class SiriusVersionMismatchError(Exception):
    """Raised when the installed SIRIUS version doesn't match a config's required one.

    Shared by every SIRIUS-touching CLI (`generate-groundtruth`, `annotate`)
    via `require_matching_sirius_version` below, so their fail-fast version
    checks raise byte-for-byte identical errors.
    """


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
    construction, and again on the first call after `shutdown`, not through
    a method on this Protocol -- a `FakeSirius` construction makes no SIRIUS
    connection at all, so there is nothing to protocol-ize there. Everything
    past construction is captured here, so swapping one implementation for
    the other requires no caller-side code changes.
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

        `project_name` defaults to `project_path.stem`. Any missing parent
        directory of `project_path` is created first -- SIRIUS' Nitrite store
        refuses to open a project file whose containing directory doesn't
        exist. Every other method on this Protocol operates on whichever
        project was created most recently.
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
        """Submit `job_submission` for analysis against the current project and wait for it.

        Raises:
            SiriusJobFailedError: SIRIUS reports the job FAILED or CANCELED.
            SiriusJobStalledError: the job's progress stood still for the
                stall timeout; SIRIUS was killed and the current project
                dropped, as after `shutdown`.
        """
        ...

    def get_structure_candidates(
        self, features: Sequence[AlignedFeature], *, top_k: int
    ) -> list[FeatureStructureCandidate]:
        """Each of `features`' Top-k (feature, structure candidate) pairs in the current project.

        Per feature: at most `top_k` structure candidates, its best by
        CSI:FingerID score -- ranks 1..`top_k` -- in rank order (CONTEXT.md's
        Top-k, ADR 0001). Joins each of `features` (as returned by
        `get_features`) with its structure candidates, since
        `StructureCandidateFormula` does not carry ion mass on its own -- a
        feature with zero structure candidates contributes no rows here (see
        `get_features` for those). Callers pass only the features the
        analysis job ran on: any other feature has no results to read.
        """
        ...

    def close_project(self) -> None:
        """Close the current project, releasing it from the SIRIUS instance.

        A project stays registered in the running SIRIUS instance -- holding
        its Nitrite database open -- until it is closed, even after the
        directory it lives in is deleted. A batch that creates one project
        per chunk and never closes any accumulates them for the lifetime of
        the instance, so every caller closes what it created. Calling this
        without a current project is a no-op, so it is safe in a `finally`.
        """
        ...

    def shutdown(self) -> None:
        """Stop the SIRIUS process and wait for it to exit, releasing its memory.

        SIRIUS 6.5.4 leaks roughly 0.9 GB of heap per 1,000-spectrum chunk
        (per-job `ParameterConfig` layers stay registered as event listeners
        on its global default configuration), and only a process exit frees
        it -- closing the project does not. Any current project is dropped.
        The next call that needs SIRIUS attaches to or starts a fresh
        instance, so a batch can shut down between inputs without rebuilding
        its wrapper. Calling this while no instance is running is a no-op, so
        it is safe in a `finally`.
        """
        ...


def require_matching_sirius_version(
    sirius: SiriusInterface, required_version: str
) -> None:
    """Fail fast if the running SIRIUS instance isn't `required_version`.

    Raises:
        SiriusVersionMismatchError: the installed and required versions differ.
    """
    installed_version = sirius.get_version()
    if installed_version != required_version:
        raise SiriusVersionMismatchError(
            f"Installed SIRIUS version {installed_version!r} does not match "
            f"this config's required_sirius_version {required_version!r}. "
            "Install the required SIRIUS version or update the config."
        )


def job_submission_from_config(
    analysis_params: TOMLTable, *, recompute: bool
) -> JobSubmission:
    """`[sirius.analysis_params]` as a `JobSubmission` SIRIUS reads correctly.

    Equivalent to `JobSubmission(**analysis_params, recompute=recompute)`,
    except `formula_id_params.use_heuristic` is sent as `configMap` string
    entries (merged over any `config_map` already configured) instead of
    the `useHeuristic` object. SIRIUS 6.5.4's REST server deserializes
    requests with Jackson 3, which leaves the `final` fields of SIRIUS's
    `UseHeuristic` class at its no-arg constructor's `0`: the object form
    reaches the SIRIUS command as `--UseHeuristic.useHeuristicAboveMz=0
    --UseHeuristic.useOnlyHeuristicAboveMz=0` whatever values it carries.
    `configMap` strings bypass that deserialization.
    """
    params = dict(analysis_params)
    formula_id_params = params.get("formula_id_params")
    if isinstance(formula_id_params, dict) and (
        formula_id_params.get("use_heuristic") is not None
    ):
        formula_id_params = dict(formula_id_params)
        # Validated through PySirius' model so types and key spellings are
        # checked exactly as they were when sent as the object form.
        use_heuristic = UseHeuristic.model_validate(
            formula_id_params.pop("use_heuristic")
        )
        params["formula_id_params"] = formula_id_params
        thresholds = {
            "UseHeuristic.useHeuristicAboveMz": use_heuristic.use_heuristic_above_mz,
            "UseHeuristic.useOnlyHeuristicAboveMz": (
                use_heuristic.use_only_heuristic_above_mz
            ),
        }
        existing_config_map = params.get("config_map") or {}
        if isinstance(existing_config_map, dict):
            params["config_map"] = {
                **existing_config_map,
                **{
                    key: str(value)
                    for key, value in thresholds.items()
                    if value is not None
                },
            }
    return JobSubmission(**{**params, "recompute": recompute})


def _credentials_from_env() -> AccountCredentials:
    username = os.getenv("SIRIUS_USER")
    password = os.getenv("SIRIUS_PW")
    if not username or not password:
        raise SiriusCredentialsError(
            "SIRIUS_USER and SIRIUS_PW environment variables must both be set "
            "(e.g. in a .env file) to log into a SIRIUS account."
        )
    return AccountCredentials(username=username, password=password)


#: Sort for SIRIUS's paged structure-candidates endpoint that puts a
#: feature's rank-1 candidate first. Spelled out rather than relying on the
#: unsorted default, and deliberately not `rank,asc`: SIRIUS 6.5.4 inverts
#: that one, returning the *worst* candidates first (ADR 0001).
_BEST_CSI_SCORE_FIRST = "csiScore,desc"


#: How long an analysis job's progress may stand still before `Sirius.run`
#: declares SIRIUS deadlocked and kills it. SIRIUS 6.5.4 can park every
#: worker of its job pool for good: `FingerblastSearchJJob` waits on
#: ElGordo lipid-tagging subjobs queued on that same, by then exhausted,
#: pool. A healthy job advances every few seconds -- 2,000-spectrum
#: chunks finish in 11-15 minutes overall -- so 20 minutes of standstill
#: is far outside normal behavior.
JOB_STALL_TIMEOUT_SECONDS = 20 * 60.0

#: How often `_await_job` polls SIRIUS for the job's progress.
_JOB_POLL_INTERVAL_SECONDS = 1.0


class Sirius:
    """Orchestrates a real, locally-running SIRIUS desktop application via PySirius.

    Attaches to (or starts) SIRIUS on construction and ensures it is logged
    in, mirroring `metabolite_annotator`'s `Sirius` wrapper. After
    `shutdown`, the next call that needs SIRIUS repeats that attach-or-start
    and login. Implements `SiriusInterface`; use `sirius_fake.FakeSirius`
    instead of this class in every automated test, since SIRIUS requires a
    licensed, locally-installed desktop app unavailable in CI.
    """

    def __init__(
        self,
        *,
        headless: bool = True,
        stall_timeout_seconds: float = JOB_STALL_TIMEOUT_SECONDS,
    ) -> None:
        load_dotenv()
        self._headless = headless
        self._stall_timeout_seconds = stall_timeout_seconds
        self._sdk = SiriusSDK()
        self._project_info: ProjectInfo | None = None
        self._running_api: PySiriusAPI | None = self._connect()

    @property
    def _api(self) -> PySiriusAPI:
        if self._running_api is None:
            self._running_api = self._connect()
        return self._running_api

    def _connect(self) -> PySiriusAPI:
        api = self._sdk.attach_or_start_sirius(headless=self._headless)
        if api is None:
            raise SiriusStartupError("Failed to attach to or start a SIRIUS instance.")
        account = api.account()
        # An attached instance is usually already logged in (SIRIUS keeps a
        # refresh token), and every login is a live OAuth round-trip through
        # SIRIUS' auth service -- one that has been observed to fail the whole
        # run with `500 ACTION_ERROR_TIMEOUT: Action execution timed out`.
        # Connecting is therefore idempotent with respect to login rather
        # than re-authenticating a session that already works.
        if not account.is_logged_in():
            account.login(True, _credentials_from_env())
        return api

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
        resolved = project_path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        self._project_info = self._api.projects().create_project(
            project_id=name, path_to_project=str(resolved)
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
        job_id = self._api.jobs().start_job(project_id, job_submission).id
        if job_id is None:
            raise SiriusJobFailedError("SIRIUS started the analysis job without an id.")
        try:
            _await_job(
                lambda: self._job_progress(project_id, job_id),
                stall_timeout_seconds=self._stall_timeout_seconds,
            )
        except SiriusJobStalledError:
            # A deadlocked job pool cannot be unblocked over REST, and a
            # graceful shutdown would have to wait on the parked workers.
            self._kill()
            raise

    def _job_progress(self, project_id: str, job_id: str) -> JobProgress:
        progress = self._api.jobs().get_job(project_id, job_id).progress
        if progress is None:
            raise SiriusJobFailedError(
                f"SIRIUS reported analysis job {job_id} without its progress."
            )
        return progress

    def get_features(self) -> list[AlignedFeature]:
        project_id = self._require_project_id()
        return self._api.features().get_aligned_features(project_id)

    def get_structure_candidates(
        self, features: Sequence[AlignedFeature], *, top_k: int
    ) -> list[FeatureStructureCandidate]:
        project_id = self._require_project_id()
        features_api = self._api.features()
        rows: list[FeatureStructureCandidate] = []
        for feature in features:
            # Only the first page of `top_k` is ever read: SIRIUS still
            # stores every candidate, but nothing past rank k is transferred.
            page = features_api.get_structure_candidates_page(
                project_id,
                feature.aligned_feature_id,
                page=0,
                size=top_k,
                sort=[_BEST_CSI_SCORE_FIRST],
            )
            rows.extend(
                FeatureStructureCandidate(feature=feature, candidate=candidate)
                for candidate in page.content or []
            )
        return rows

    def close_project(self) -> None:
        if self._project_info is None:
            return
        self._api.projects().close_project(self._project_info.project_id)
        self._project_info = None

    def shutdown(self) -> None:
        if self._running_api is None:
            return
        # Captured before `shutdown_sirius`, which forgets both on success.
        # The `Popen` handle is needed to reap a process this SDK started: a
        # zombie would still pass a pid liveness check.
        process: subprocess.Popen[bytes] | None = SiriusSDK.process
        pid: int | None = SiriusSDK.process_id
        self._sdk.shutdown_sirius()
        self._running_api = None
        self._project_info = None
        # `shutdown_sirius` leaves the dead instance's port and API client
        # behind; the next attach must search for a live instance afresh.
        self._sdk.reset_sdk_class()
        _wait_for_exit(process, pid)

    def _kill(self) -> None:
        """SIGKILL the SIRIUS process and forget it, as `shutdown` does.

        The pid/port files it leaves in `~/.sirius*` are harmless: the next
        attach finds no live API behind them, deletes them and starts anew.

        Raises:
            SiriusShutdownError: the process id is unknown, or the process
                is still alive after `_SHUTDOWN_TIMEOUT_SECONDS`.
        """
        process: subprocess.Popen[bytes] | None = SiriusSDK.process
        pid: int | None = SiriusSDK.process_id
        if process is not None:
            process.kill()
        elif pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        else:
            raise SiriusShutdownError(
                "Cannot kill the stalled SIRIUS instance: its process id is unknown."
            )
        self._running_api = None
        self._project_info = None
        self._sdk.reset_sdk_class()
        _wait_for_exit(process, pid)

    def _require_project_id(self) -> str:
        if self._project_info is None:
            raise NoActiveProjectError(NO_ACTIVE_PROJECT_MESSAGE)
        return self._project_info.project_id


def _await_job(
    fetch_progress: Callable[[], JobProgress],
    *,
    stall_timeout_seconds: float,
    poll_interval_seconds: float = _JOB_POLL_INTERVAL_SECONDS,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Poll `fetch_progress` until the job is DONE.

    Any change of state, current or maximum progress resets the stall
    clock; only a job whose progress stays identical for
    `stall_timeout_seconds` counts as stalled.

    Raises:
        SiriusJobFailedError: the job ended FAILED or CANCELED.
        SiriusJobStalledError: the job's progress didn't change for
            `stall_timeout_seconds`.
    """
    last_seen: tuple[JobState | None, int | None, int | None] | None = None
    last_change = monotonic()
    while True:
        progress = fetch_progress()
        if progress.state is JobState.DONE:
            return
        if progress.state in (JobState.FAILED, JobState.CANCELED):
            raise SiriusJobFailedError(
                f"SIRIUS analysis job ended {progress.state.value}: "
                f"{progress.error_message or 'no error message'}"
            )
        seen = (progress.state, progress.current_progress, progress.max_progress)
        now = monotonic()
        if seen != last_seen:
            last_seen, last_change = seen, now
        elif now - last_change >= stall_timeout_seconds:
            raise SiriusJobStalledError(
                f"SIRIUS analysis job made no progress for "
                f"{now - last_change:.0f}s (stuck at {progress.current_progress}/"
                f"{progress.max_progress}); presumed deadlocked, SIRIUS killed."
            )
        sleep(poll_interval_seconds)


#: How long `Sirius.shutdown` waits for the JVM to exit. `shutdown_sirius`
#: returns as soon as SIRIUS acknowledges the request, while the old process
#: may still be running -- starting the next instance before it is gone
#: would briefly hold both heaps and could attach to the dying one.
_SHUTDOWN_TIMEOUT_SECONDS = 120.0


def _wait_for_exit(process: subprocess.Popen[bytes] | None, pid: int | None) -> None:
    """Block until the shut-down SIRIUS process is gone.

    Raises:
        SiriusShutdownError: it is still alive after
            `_SHUTDOWN_TIMEOUT_SECONDS`.
    """
    if process is not None:
        try:
            process.wait(timeout=_SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            raise SiriusShutdownError(
                f"SIRIUS process {process.pid} did not exit within "
                f"{_SHUTDOWN_TIMEOUT_SECONDS:.0f}s of being shut down."
            ) from exc
        return
    if pid is None:
        return
    deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
    while _pid_is_alive(pid):
        if time.monotonic() > deadline:
            raise SiriusShutdownError(
                f"SIRIUS process {pid} did not exit within "
                f"{_SHUTDOWN_TIMEOUT_SECONDS:.0f}s of being shut down."
            )
        time.sleep(0.5)


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # alive, owned by another user
        pass
    return True

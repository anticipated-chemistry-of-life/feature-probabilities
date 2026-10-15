"""Tests for `sirius`'s pure helpers: `job_submission_from_config`, `_await_job`.

The real `Sirius` wrapper needs a licensed SIRIUS (see
`test_sirius_qtof_integration.py`); these helpers are pure and run anywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest
from PySirius import JobProgress, JobState

from feature_probabilities.sirius import (
    SiriusJobFailedError,
    SiriusJobStalledError,
    _await_job,
    job_submission_from_config,
)


def test_heuristic_thresholds_are_sent_as_config_map_strings() -> None:
    """SIRIUS 6.5.4 runs a `useHeuristic` object as 0/0 whatever it carries.

    The thresholds must therefore reach SIRIUS only as `configMap` strings,
    alongside any entries already configured, with no `useHeuristic` object
    left in `formulaIdParams` to compete with them.
    """
    submission = job_submission_from_config(
        {
            "formula_id_params": {
                "enabled": True,
                "use_heuristic": {
                    "use_heuristic_above_mz": 300,
                    "use_only_heuristic_above_mz": 650,
                },
            },
            "config_map": {"NumberOfStructureCandidates": "5"},
        },
        recompute=False,
    )

    wire = submission.to_dict()
    assert "useHeuristic" not in wire["formulaIdParams"]
    assert wire["formulaIdParams"]["enabled"] is True
    assert wire["configMap"] == {
        "NumberOfStructureCandidates": "5",
        "UseHeuristic.useHeuristicAboveMz": "300",
        "UseHeuristic.useOnlyHeuristicAboveMz": "650",
    }


class _Clock:
    """A fake monotonic clock that only advances when `_await_job` sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _progress(current: int, state: JobState = JobState.RUNNING) -> JobProgress:
    return JobProgress(state=state, currentProgress=current, maxProgress=100)


def _polls(*progresses: JobProgress) -> Callable[[], JobProgress]:
    """Each call returns the next progress; the last one repeats forever."""
    remaining: Iterator[JobProgress] = iter(progresses)
    last = progresses[-1]

    def fetch() -> JobProgress:
        return next(remaining, last)

    return fetch


def test_frozen_progress_is_a_stall_once_the_timeout_elapses() -> None:
    clock = _Clock()

    with pytest.raises(SiriusJobStalledError):
        _await_job(
            _polls(_progress(10)),
            stall_timeout_seconds=60.0,
            poll_interval_seconds=1.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

    assert clock.now == 60.0


def test_slow_but_advancing_progress_is_not_a_stall() -> None:
    """Only standstill counts: total runtime far beyond the timeout is fine.

    Every value is held for 59 polls -- just under the 60 s timeout --
    before it advances, so the job runs ~4 minutes with a 1-minute timeout.
    """
    clock = _Clock()
    held = [_progress(step) for step in range(4) for _ in range(59)]

    _await_job(
        _polls(*held, _progress(100, JobState.DONE)),
        stall_timeout_seconds=60.0,
        poll_interval_seconds=1.0,
        monotonic=clock.monotonic,
        sleep=clock.sleep,
    )

    assert clock.now > 60.0


@pytest.mark.parametrize("state", [JobState.FAILED, JobState.CANCELED])
def test_failed_or_canceled_job_raises_instead_of_counting_as_done(
    state: JobState,
) -> None:
    clock = _Clock()

    with pytest.raises(SiriusJobFailedError, match=state.value):
        _await_job(
            _polls(_progress(10), _progress(10, state)),
            stall_timeout_seconds=60.0,
            poll_interval_seconds=1.0,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )

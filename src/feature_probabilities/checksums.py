"""Pure functions computing the SIRIUS rerun-avoidance caching key's components.

Each ``sirius_runs`` row is keyed by three sha256 checksums:

- :func:`input_file_checksum` — the raw bytes of the input file (the mzML for
  field runs, the MGF chunk for ground-truth runs).
- :func:`import_params_checksum` — the canonical (sorted-keys) JSON of the
  SIRIUS ``LcmsSubmissionParameters`` object used for mzML peak-picking.
  ``None`` for ground-truth runs, whose pre-picked import path takes no such
  object.
- :func:`analysis_params_checksum` — the canonical JSON of the SIRIUS
  ``JobSubmission`` object, which already serializes every SIRIUS analysis
  parameter, plus the feature-quality filter selecting which imported
  features that job runs on (field runs only).

This module operates purely on bytes and plain ``to_dict``-shaped objects: it
imports neither DuckDB nor PySirius, so it can be exercised without either
dependency and later composed into the actual cache lookup by other modules.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Collection

type JSONValue = (
    str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]
)


class ToDictConvertible(Protocol):
    """Structural shape of PySirius's pydantic-generated parameter models.

    Both ``LcmsSubmissionParameters`` and ``JobSubmission`` expose a
    dependency-free ``to_dict`` method returning their canonical dict
    representation; this module never imports either class, only relies on
    this shape.
    """

    def to_dict(self) -> dict[str, JSONValue]: ...


def input_file_checksum(data: bytes) -> str:
    """Sha256 hex digest of a file's raw bytes."""
    return hashlib.sha256(data).hexdigest()


def import_params_checksum(params: ToDictConvertible | None) -> str | None:
    """Sha256 hex digest of an ``LcmsSubmissionParameters``-shaped object.

    Returns ``None`` when ``params`` is ``None`` — the ground-truth
    (pre-picked) case, which has no such object at all.
    """
    if params is None:
        return None
    return _canonical_json_checksum(params.to_dict())


#: `JobSubmission` keys that control *whether* SIRIUS redoes work rather than
#: *what* it computes, and so must not take part in a run's identity.
#: `recompute` is set from every SIRIUS-touching CLI's `--force` flag: leaving
#: it in the checksum meant a forced run's `sirius_runs` row could never be
#: found again by a normal run, so one `--force` made every later invocation
#: recompute from scratch and append a duplicate set of feature/annotation
#: rows.
_NON_IDENTIFYING_ANALYSIS_KEYS = frozenset({"recompute"})


#: Key the feature-quality filter is stored under in the checksummed payload.
#: Not a `JobSubmission` field, so it can never collide with one.
_ANNOTATED_QUALITIES_KEY = "annotated_qualities"


def analysis_params_checksum(
    params: ToDictConvertible, annotated_qualities: Collection[str] | None = None
) -> str:
    """Sha256 hex digest of a ``JobSubmission``-shaped object.

    Excludes :data:`_NON_IDENTIFYING_ANALYSIS_KEYS`, so two submissions that
    differ only in `recompute` -- which produce identical results -- share one
    cache identity.

    `annotated_qualities` are the `DataQuality` values a run restricts its
    analysis job to. A different filter annotates a different feature set,
    so it is part of the identity; `None` (no filter, every imported feature
    analysed) leaves the payload -- and so every existing checksum --
    unchanged.
    """
    payload = {
        key: value
        for key, value in params.to_dict().items()
        if key not in _NON_IDENTIFYING_ANALYSIS_KEYS
    }
    if annotated_qualities is not None:
        payload[_ANNOTATED_QUALITIES_KEY] = sorted(annotated_qualities)
    return _canonical_json_checksum(payload)


def _canonical_json_checksum(payload: dict[str, JSONValue]) -> str:
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

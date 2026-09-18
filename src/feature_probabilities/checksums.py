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
  parameter.

This module operates purely on bytes and plain ``to_dict``-shaped objects: it
imports neither DuckDB nor PySirius, so it can be exercised without either
dependency and later composed into the actual cache lookup by other modules.
"""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

type JSONValue = str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]


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


def analysis_params_checksum(params: ToDictConvertible) -> str:
    """Sha256 hex digest of a ``JobSubmission``-shaped object."""
    return _canonical_json_checksum(params.to_dict())


def _canonical_json_checksum(payload: dict[str, JSONValue]) -> str:
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

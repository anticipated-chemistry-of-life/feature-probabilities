"""Pinned-by-default fetch of MassSpecGym v1.5 from HuggingFace.

``fp-generate-groundtruth`` must always start from a byte-for-byte
reproducible ground-truth set, so :func:`fetch_massspecgym_tsv` resolves
``roman-bushuiev/MassSpecGym``'s ``data/MassSpecGym1.5.tsv`` at an exact,
documented commit SHA (:data:`PINNED_MASSSPECGYM_REVISION`) rather than at
``main``, unless a caller passes an explicit ``revision`` override to
deliberately test against newer data.

This module is a thin wrapper around ``huggingface_hub.hf_hub_download``: it
does not parse the downloaded TSV's contents, only resolves and downloads it,
returning a local file path for a parser to consume.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, HFValidationError

MASSSPECGYM_REPO_ID = "roman-bushuiev/MassSpecGym"
MASSSPECGYM_FILENAME = "data/MassSpecGym1.5.tsv"

# The `main` branch HEAD of roman-bushuiev/MassSpecGym as of 2026-09-18,
# resolved via `GET https://huggingface.co/api/datasets/roman-bushuiev/
# MassSpecGym/revision/main` and verified to serve an unchanged
# `data/MassSpecGym1.5.tsv`. Bump deliberately (checking the new SHA the same
# way) when a newer MassSpecGym release is actually needed; day-to-day callers
# should never need the `revision` override below.
PINNED_MASSSPECGYM_REVISION = "d2e86d0c3bd905a6d578c0dd6053ed2bd41f9c2a"

_FETCH_ERRORS = (HfHubHTTPError, EntryNotFoundError, HFValidationError, OSError)


class MassSpecGymFetchError(Exception):
    """Raised when the pinned (or overridden) MassSpecGym revision can't be fetched."""


def fetch_massspecgym_tsv(revision: str | None = None) -> Path:
    """Download MassSpecGym v1.5's TSV at an exact HuggingFace revision.

    Defaults to :data:`PINNED_MASSSPECGYM_REVISION`, not ``main``, so two runs
    months apart use byte-for-byte identical ground truth. Pass an explicit
    ``revision`` (a commit SHA, tag, or branch name) to deliberately fetch a
    different MassSpecGym snapshot.

    Returns a local file path to the downloaded (or already locally cached)
    TSV, ready for a parser to consume; this function never reads or parses
    the file's contents itself.

    Raises:
        MassSpecGymFetchError: the requested revision can't be resolved or
            downloaded — e.g. a bogus commit SHA, a renamed/missing file at
            that revision, or a network failure — instead of a raw
            ``huggingface_hub`` traceback.
    """
    resolved_revision = PINNED_MASSSPECGYM_REVISION if revision is None else revision
    try:
        local_path = hf_hub_download(
            repo_id=MASSSPECGYM_REPO_ID,
            filename=MASSSPECGYM_FILENAME,
            repo_type="dataset",
            revision=resolved_revision,
        )
    except _FETCH_ERRORS as exc:
        raise MassSpecGymFetchError(
            f"Could not fetch {MASSSPECGYM_FILENAME!r} from dataset "
            f"{MASSSPECGYM_REPO_ID!r} at revision {resolved_revision!r}. Check "
            "that the revision exists and is reachable on HuggingFace Hub "
            f"(original error: {exc})."
        ) from exc
    return Path(local_path)

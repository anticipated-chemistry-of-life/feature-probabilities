"""Tests for the pinned-by-default MassSpecGym HuggingFace fetch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest
from huggingface_hub.errors import RevisionNotFoundError

from feature_probabilities.massspecgym import (
    MASSSPECGYM_FILENAME,
    MASSSPECGYM_REPO_ID,
    PINNED_MASSSPECGYM_REVISION,
    MassSpecGymFetchError,
    fetch_massspecgym_tsv,
)


def _patch_hf_hub_download(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    monkeypatch.setattr("feature_probabilities.massspecgym.hf_hub_download", fake)


def test_fetch_with_no_arguments_requests_the_pinned_revision_not_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_kwargs: dict[str, Any] = {}

    def fake_hf_hub_download(**kwargs: Any) -> str:
        requested_kwargs.update(kwargs)
        return "/tmp/fake/MassSpecGym1.5.tsv"

    _patch_hf_hub_download(monkeypatch, fake_hf_hub_download)

    fetch_massspecgym_tsv()

    assert requested_kwargs["revision"] == PINNED_MASSSPECGYM_REVISION
    assert requested_kwargs["revision"] != "main"
    assert requested_kwargs["repo_id"] == MASSSPECGYM_REPO_ID
    assert requested_kwargs["filename"] == MASSSPECGYM_FILENAME
    assert requested_kwargs["repo_type"] == "dataset"


def test_fetch_with_explicit_revision_requests_that_revision_instead(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_kwargs: dict[str, Any] = {}

    def fake_hf_hub_download(**kwargs: Any) -> str:
        requested_kwargs.update(kwargs)
        return "/tmp/fake/MassSpecGym1.5.tsv"

    _patch_hf_hub_download(monkeypatch, fake_hf_hub_download)

    fetch_massspecgym_tsv(revision="deadbeefcafef00ddeadbeefcafef00ddeadbeef")

    assert requested_kwargs["revision"] == "deadbeefcafef00ddeadbeefcafef00ddeadbeef"


def test_fetch_returns_a_local_path_to_the_downloaded_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_hf_hub_download(monkeypatch, lambda **kwargs: "/tmp/fake/MassSpecGym1.5.tsv")

    result = fetch_massspecgym_tsv()

    assert result == Path("/tmp/fake/MassSpecGym1.5.tsv")


def test_unresolvable_revision_raises_a_clear_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_hf_hub_download(**kwargs: Any) -> str:
        response = httpx.Response(
            404, request=httpx.Request("GET", "https://huggingface.co/x")
        )
        raise RevisionNotFoundError("Revision Not Found", response=response)

    _patch_hf_hub_download(monkeypatch, fake_hf_hub_download)

    with pytest.raises(MassSpecGymFetchError) as exc_info:
        fetch_massspecgym_tsv(revision="0000000000000000000000000000000000bogus")

    message = str(exc_info.value)
    assert "0000000000000000000000000000000000bogus" in message
    assert MASSSPECGYM_REPO_ID in message
    assert MASSSPECGYM_FILENAME in message


def test_unresolvable_revision_error_chains_the_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_hf_hub_download(**kwargs: Any) -> str:
        raise OSError("network unreachable")

    _patch_hf_hub_download(monkeypatch, fake_hf_hub_download)

    with pytest.raises(MassSpecGymFetchError) as exc_info:
        fetch_massspecgym_tsv()

    assert isinstance(exc_info.value.__cause__, OSError)

"""Tests for `msnlib_ms1.fetch_msnlib_mzml_archives`: the archive cache.

How a spectrum's MS1 scan is picked is exercised end-to-end through the
CLI in `test_cli_generate_groundtruth.py`.
"""

from __future__ import annotations

import hashlib
import io
from typing import TYPE_CHECKING

import pytest

from feature_probabilities import msnlib_ms1
from feature_probabilities.msnlib import MsnlibFetchError

if TYPE_CHECKING:
    from pathlib import Path


def _serve(monkeypatch: pytest.MonkeyPatch, body: bytes) -> list[str]:
    """Answer every download with `body`; returns the URLs requested."""
    requested: list[str] = []

    def urlopen(url: str, timeout: float) -> io.BytesIO:
        requested.append(url)
        return io.BytesIO(body)

    monkeypatch.setattr(msnlib_ms1.urllib.request, "urlopen", urlopen)
    return requested


def test_an_archive_is_downloaded_once_and_then_read_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = b"archive bytes"
    monkeypatch.setattr(
        msnlib_ms1,
        "MSNLIB_MZML_ARCHIVES",
        {"a.zip": hashlib.md5(body).hexdigest()},
    )
    requested = _serve(monkeypatch, body)

    first = msnlib_ms1.fetch_msnlib_mzml_archives(tmp_path)
    second = msnlib_ms1.fetch_msnlib_mzml_archives(tmp_path)

    assert (
        first == second == [tmp_path / msnlib_ms1.MSNLIB_MZML_ZENODO_RECORD / "a.zip"]
    )
    assert first[0].read_bytes() == body
    assert requested == [
        (
            f"https://zenodo.org/api/records/{msnlib_ms1.MSNLIB_MZML_ZENODO_RECORD}"
            "/files/a.zip/content"
        )
    ]


def test_an_archive_whose_md5_mismatches_is_rejected_and_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        msnlib_ms1,
        "MSNLIB_MZML_ARCHIVES",
        {"a.zip": hashlib.md5(b"expected").hexdigest()},
    )
    _serve(monkeypatch, b"truncated")

    with pytest.raises(MsnlibFetchError, match="a.zip"):
        msnlib_ms1.fetch_msnlib_mzml_archives(tmp_path)

    assert list((tmp_path / msnlib_ms1.MSNLIB_MZML_ZENODO_RECORD).iterdir()) == []

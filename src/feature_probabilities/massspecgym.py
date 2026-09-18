"""Pinned-by-default fetch, parsing, and SIRIUS-chunking of MassSpecGym v1.5.

``generate-groundtruth`` must always start from a byte-for-byte
reproducible ground-truth set, so :func:`fetch_massspecgym_tsv` resolves
``roman-bushuiev/MassSpecGym``'s ``data/MassSpecGym1.5.tsv`` at an exact,
documented commit SHA (:data:`PINNED_MASSSPECGYM_REVISION`) rather than at
``main``, unless a caller passes an explicit ``revision`` override to
deliberately test against newer data. It is a thin wrapper around
``huggingface_hub.hf_hub_download``: it does not parse the downloaded TSV's
contents itself.

:func:`parse_massspecgym_spectra` and :func:`write_sirius_chunks` (or the
:func:`chunk_massspecgym_for_sirius` convenience that composes both) turn
that TSV into SIRIUS-ready input: one ``matchms.Spectrum`` per row, split into
``instrument_type``-grouped chunks of up to 30,000 spectra each
(:data:`DEFAULT_SIRIUS_CHUNK_SIZE`), written as MGF files -- matching
``ms2mol-evaluation``'s existing chunking pattern (generalized across every
instrument type present, not hardcoded to Orbitrap/QTOF). Each spectrum
carries the fields SIRIUS's pre-picked import path requires (``ms_level=2``,
``charge``, ``pepmass``, a ``feature_id`` mapping back to the MassSpecGym
``identifier``), mirroring ``ms2mol-evaluation``'s
``_add_required_metadata_for_sirius``, including clearing ``formula``/
``precursor_formula`` to ``None`` (dropped entirely from the written MGF) so
the ground-truth formula is never leaked to SIRIUS's own formula prediction.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, HFValidationError
from matchms import Spectrum
from matchms.exporting import save_as_mgf

if TYPE_CHECKING:
    from collections.abc import Sequence


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


#: Required MassSpecGym TSV columns, in the order the source dataset provides
#: them. A TSV missing any of these fails fast with :class:`MassSpecGymParseError`
#: rather than silently producing a malformed chunk.
MASSSPECGYM_COLUMNS = (
    "identifier",
    "mzs",
    "intensities",
    "smiles",
    "inchikey",
    "formula",
    "precursor_formula",
    "parent_mass",
    "precursor_mz",
    "adduct",
    "instrument_type",
    "collision_energy",
    "fold",
    "simulation_challenge",
)

#: Non-array columns carried through to each spectrum's metadata and
#: exported as MGF fields -- every value is required and non-blank, per
#: acceptance criterion 4 ("malformed or missing required fields ... produce
#: a clear error rather than a silently malformed chunk"), except the
#: columns in `_OPTIONAL_METADATA_COLUMNS`.
#:
#: `collision_energy` is optional per row: real MassSpecGym rows legitimately
#: leave it blank for spectra with no recorded collision energy, so a blank
#: value there is carried through as an absent MGF field rather than a parse
#: error.
_OPTIONAL_METADATA_COLUMNS = ("collision_energy",)
_METADATA_COLUMNS = tuple(
    column
    for column in MASSSPECGYM_COLUMNS
    if column not in ("mzs", "intensities", *_OPTIONAL_METADATA_COLUMNS)
)

#: `ms2mol-evaluation`'s existing per-chunk spectra cap for SIRIUS's
#: pre-picked import path.
DEFAULT_SIRIUS_CHUNK_SIZE = 30_000


class MassSpecGymParseError(Exception):
    """Raised when a MassSpecGym TSV is missing a required column, or a row is
    missing/malformed in a required field, rather than silently producing a
    malformed chunk.
    """


def _require_columns(df: pd.DataFrame, tsv_path: Path) -> None:
    missing = [column for column in MASSSPECGYM_COLUMNS if column not in df.columns]
    if missing:
        raise MassSpecGymParseError(
            f"{tsv_path} is missing required MassSpecGym column(s): "
            f"{', '.join(missing)}."
        )


def _is_blank(value: object) -> bool:
    return bool(pd.isna(value)) or not str(value).strip()


def _require_value(row: pd.Series, field_name: str, *, identifier: str) -> str:
    value = row[field_name]
    if _is_blank(value):
        raise MassSpecGymParseError(
            f"Row {identifier!r} is missing a required {field_name!r} value."
        )
    return str(value)


def _require_float(row: pd.Series, field_name: str, *, identifier: str) -> float:
    raw = _require_value(row, field_name, identifier=identifier)
    try:
        return float(raw)
    except ValueError as exc:
        raise MassSpecGymParseError(
            f"Row {identifier!r} has a malformed {field_name!r} value that "
            f"could not be parsed as a number: {raw!r}."
        ) from exc


def _parse_peak_array(row: pd.Series, field_name: str, *, identifier: str) -> np.ndarray:
    raw = _require_value(row, field_name, identifier=identifier)
    try:
        return np.array([float(value) for value in raw.split(",")], dtype=float)
    except ValueError as exc:
        raise MassSpecGymParseError(
            f"Row {identifier!r} has a malformed {field_name!r} value that "
            f"could not be parsed as comma-separated floats: {raw!r}."
        ) from exc


def _add_required_metadata_for_sirius(
    spectrum: Spectrum, *, identifier: str, precursor_mz: float
) -> None:
    """Set SIRIUS's pre-picked-import-required fields.

    Mirrors `ms2mol-evaluation`'s `_add_required_metadata_for_sirius`:
    `formula`/`precursor_formula` are cleared to `None` (dropped entirely
    from the written MGF) so the ground-truth formula is never leaked to
    SIRIUS's own formula prediction.
    """
    spectrum.set("ms_level", 2)
    spectrum.set("formula", None)
    spectrum.set("precursor_formula", None)
    spectrum.set("feature_id", identifier)
    # Only positive-mode adducts exist in the current MassSpecGym release.
    spectrum.set("charge", "1+")
    spectrum.set("pepmass", precursor_mz)


def _row_to_spectrum(row: pd.Series) -> Spectrum:
    identifier = _require_value(row, "identifier", identifier=str(row.name))

    mzs = _parse_peak_array(row, "mzs", identifier=identifier)
    intensities = _parse_peak_array(row, "intensities", identifier=identifier)
    if mzs.shape != intensities.shape or mzs.size == 0:
        raise MassSpecGymParseError(
            f"Row {identifier!r} has mismatched or empty 'mzs'/'intensities' arrays."
        )
    order = np.argsort(mzs, kind="stable")
    mzs = mzs[order]
    intensities = intensities[order]

    metadata: dict[str, str] = {
        column: _require_value(row, column, identifier=identifier)
        for column in _METADATA_COLUMNS
        if column != "identifier"
    }
    metadata["identifier"] = identifier
    for column in _OPTIONAL_METADATA_COLUMNS:
        if not _is_blank(row[column]):
            metadata[column] = str(row[column])
    precursor_mz = _require_float(row, "precursor_mz", identifier=identifier)

    spectrum = Spectrum(mz=mzs, intensities=intensities, metadata=metadata)
    _add_required_metadata_for_sirius(
        spectrum, identifier=identifier, precursor_mz=precursor_mz
    )
    return spectrum


def parse_massspecgym_spectra(tsv_path: Path) -> list[Spectrum]:
    """Parse a MassSpecGym TSV into SIRIUS-import-ready `matchms.Spectrum` objects.

    Expects the columns `identifier`, `mzs`, `intensities`, `smiles`,
    `inchikey`, `formula`, `precursor_formula`, `parent_mass`,
    `precursor_mz`, `adduct`, `instrument_type`, `collision_energy`, `fold`,
    `simulation_challenge`. `collision_energy` may be blank on a per-row
    basis (not every MassSpecGym spectrum records one); every other column
    is required and non-blank. Each returned spectrum carries the fields
    SIRIUS's pre-picked import path requires -- see
    `_add_required_metadata_for_sirius`.

    Raises:
        MassSpecGymParseError: `tsv_path` is missing a required column, or a
            row is missing a required field or has a malformed `mzs`/
            `intensities` value, rather than silently producing a malformed
            spectrum.
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str, keep_default_na=False)
    _require_columns(df, tsv_path)
    return [_row_to_spectrum(row) for _, row in df.iterrows()]


def spectrum_instrument_type(spectrum: Spectrum) -> str:
    """`spectrum`'s `instrument_type` metadata value, as the plain `str` every
    caller needs (grouping/chunking here, `--instrument-type` filtering in
    `cli_generate_groundtruth`) -- the one place that conversion happens.
    """
    return str(spectrum.get("instrument_type"))


def _instrument_type_slug(instrument_type: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", instrument_type.strip()).strip("_").lower()
    return slug or "unknown"


def write_sirius_chunks(
    spectra: Sequence[Spectrum],
    output_dir: Path,
    *,
    chunk_size: int = DEFAULT_SIRIUS_CHUNK_SIZE,
) -> dict[str, list[Path]]:
    """Group `spectra` by `instrument_type` and write each as `chunk_size`-capped MGF files.

    Returns a mapping of `instrument_type` -> ordered list of written chunk
    file paths, with exactly one key per instrument type present in
    `spectra`. A group larger than `chunk_size` is split across multiple
    files rather than written as one oversized file, matching
    `ms2mol-evaluation`'s existing 30,000-spectra-per-chunk pattern
    generalized across every instrument type.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Spectrum]] = {}
    for spectrum in spectra:
        instrument_type = spectrum_instrument_type(spectrum)
        grouped.setdefault(instrument_type, []).append(spectrum)

    chunk_paths: dict[str, list[Path]] = {}
    for instrument_type, group in grouped.items():
        slug = _instrument_type_slug(instrument_type)
        paths: list[Path] = []
        for chunk_number, start in enumerate(range(0, len(group), chunk_size)):
            chunk = group[start : start + chunk_size]
            path = output_dir / f"{slug}_{chunk_number}.mgf"
            save_as_mgf(chunk, str(path), file_mode="w")
            paths.append(path)
        chunk_paths[instrument_type] = paths
    return chunk_paths


def chunk_massspecgym_for_sirius(
    tsv_path: Path,
    output_dir: Path,
    *,
    chunk_size: int = DEFAULT_SIRIUS_CHUNK_SIZE,
) -> dict[str, list[Path]]:
    """Parse `tsv_path` and write SIRIUS-ready, instrument-type-chunked MGF files.

    Composes :func:`parse_massspecgym_spectra` and :func:`write_sirius_chunks`.
    """
    spectra = parse_massspecgym_spectra(tsv_path)
    return write_sirius_chunks(spectra, output_dir, chunk_size=chunk_size)

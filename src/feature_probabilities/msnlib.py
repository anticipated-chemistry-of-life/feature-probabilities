"""Fetch, filter, parse, and SIRIUS-chunk MSnLib, the Ground truth set (GLOSSARY.md).

:func:`fetch_msnlib_parquet` downloads ``msnlib_ms2.parquet`` from the
HuggingFace dataset ``salutmoic/msnlib-ms2-parquet`` at an exact revision,
through the hub cache. :func:`read_ground_truth_set` filters it to the Ground
truth set and parses every kept row into a :class:`GroundTruthSpectrum`, and
:func:`write_sirius_chunks` writes those as SIRIUS-ready MGF chunks, one
group per (Ionization mode, Instrument type). Each spectrum also names the
raw mzML file and MS2 scans it was merged from, which `msnlib_ms1` uses to
attach its measured MS1 scan (ADR 0003).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, HFValidationError
from matchms import Spectrum
from matchms.exporting import save_as_mgf

from feature_probabilities.instrument import (
    UnknownInstrumentError,
    resolve_instrument_type,
)
from feature_probabilities.kde import IONIZATION_MODES
from feature_probabilities.schema import INCHIKEY_FIRST_BLOCK_LENGTH

if TYPE_CHECKING:
    from collections.abc import Sequence

    from feature_probabilities.instrument import InstrumentType

MSNLIB_REPO_ID = "salutmoic/msnlib-ms2-parquet"
MSNLIB_FILENAME = "msnlib_ms2.parquet"

#: The MSnLib columns this module reads. A parquet missing any of them fails
#: with :class:`MsnlibParseError` before any row is parsed.
MSNLIB_COLUMNS = (
    "identifier",
    "mzs",
    "intensities",
    "inchikey",
    "precursor_mz",
    "adduct",
    "instrument_type",
    "charge",
    "ionmode",
    "spectype",
    "other_matched_compounds",
    "usi",
    "source_scan_usi",
    "scans",
)

#: MSnLib adducts SIRIUS cannot detect: none is among
#: `[sirius.analysis_params] detectable_adducts`, so such a spectrum can never
#: yield a Correct assignment (ADR 0002).
UNDETECTABLE_ADDUCTS = frozenset({"[M]+", "[M-H2O]+", "[M]-"})

#: Down from 10,000: SIRIUS 6.5.4 can deadlock its job pool when a chunk
#: holds many ElGordo-detected lipids (`FingerblastSearchJJob` workers all
#: block awaiting lipid-tagging subjobs that no free worker can run).
#: Smaller chunks make that pile-up less likely; they do not rule it out.
SIRIUS_CHUNK_SIZE = 2_000

_MGF_CHARGE_BY_IONIZATION_MODE = {"positive": "1+", "negative": "1-"}

#: A standard 27-character InChIKey: 14-letter skeleton block, 10-letter
#: stereo/isotope block, protonation letter.
_INCHIKEY = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")

#: The raw file of a USI, `mzspec:<dataset>:<raw file>:<scans>`.
_USI_RAW_FILE = re.compile(r"mzspec:[A-Z0-9]+:([^:\s,\[\]\"]+):")

#: MSnLib's `scans`: comma-separated scan numbers or inclusive ranges.
_SCANS = re.compile(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*")


class MsnlibFetchError(Exception):
    """Raised when MSnLib's parquet or raw mzML archives can't be fetched."""


class MsnlibParseError(Exception):
    """Raised when MSnLib's parquet lacks a required column, or a kept row has
    a blank or malformed required value, rather than writing a malformed chunk.
    """


_FETCH_ERRORS = (HfHubHTTPError, EntryNotFoundError, HFValidationError, OSError)


def fetch_msnlib_parquet(revision: str) -> Path:
    """Download MSnLib's parquet at `revision` through the HuggingFace hub cache.

    `revision` is the config's pinned `ground_truth_revision` (a commit SHA),
    so two runs months apart read byte-for-byte identical ground truth even
    when the dataset's `main` moves. Returns the local, cached file path; an
    already-cached revision is not downloaded again.

    Raises:
        MsnlibFetchError: `revision` can't be resolved or downloaded -- e.g.
            a bogus commit SHA, a file missing at that revision, or a network
            failure -- instead of a raw `huggingface_hub` traceback.
    """
    try:
        local_path = hf_hub_download(
            repo_id=MSNLIB_REPO_ID,
            filename=MSNLIB_FILENAME,
            repo_type="dataset",
            revision=revision,
        )
    except _FETCH_ERRORS as exc:
        raise MsnlibFetchError(
            f"Could not fetch {MSNLIB_FILENAME!r} from HuggingFace dataset "
            f"{MSNLIB_REPO_ID!r} at revision {revision!r}. Check that the "
            "revision exists and HuggingFace Hub is reachable, or pass "
            f"--ground-truth-revision (original error: {exc})."
        ) from exc
    return Path(local_path)


@dataclass(frozen=True, slots=True)
class GroundTruthSpectrum:
    """One Ground truth set spectrum: what SIRIUS sees, and the truth it doesn't."""

    identifier: str
    #: The first block of MSnLib's InChIKey, which a Correct assignment
    #: (GLOSSARY.md) is judged against.
    true_inchikey: str
    ionization_mode: str
    instrument_type: InstrumentType
    #: Exactly what is written to the MGF chunk SIRIUS reads (`_sirius_spectrum`).
    spectrum: Spectrum
    #: Stem of the raw mzML file the spectrum's MS2 scans were measured in.
    raw_file: str
    #: The MS2 scans MSnLib merged into `spectrum`, ascending.
    scans: tuple[int, ...]
    #: The measured MS1 scan written before `spectrum`, set by
    #: `msnlib_ms1.attach_measured_ms1`; `None` when no MS1 scan preceding
    #: `scans` holds the precursor.
    ms1: Spectrum | None = None


@dataclass(frozen=True, slots=True)
class FilterCounts:
    """How many MSnLib rows the Ground truth set kept, and each filter dropped."""

    kept: int
    #: Rows whose `spectype` is not `ALL_ENERGIES`.
    dropped_not_all_energies: int
    #: `ALL_ENERGIES` rows MSnLib also matched to other library compounds,
    #: so the identity of what was fragmented is ambiguous.
    dropped_ambiguous_identity: int
    #: Remaining rows whose adduct is one of `UNDETECTABLE_ADDUCTS`.
    dropped_undetectable_adduct: int


@dataclass(frozen=True, slots=True)
class GroundTruthSet:
    """The parsed Ground truth set, plus how many rows each filter dropped."""

    spectra: tuple[GroundTruthSpectrum, ...]
    filter_counts: FilterCounts


def _is_blank(value: object) -> bool:
    return value is None or bool(pd.isna(value)) or not str(value).strip()


def _require(row: dict[str, object], column: str, *, identifier: str) -> object:
    value = row[column]
    if _is_blank(value):
        raise MsnlibParseError(
            f"Row {identifier!r} is missing a required {column!r} value."
        )
    return value


def _malformed(identifier: str, column: str, value: object, expected: str) -> str:
    return (
        f"Row {identifier!r} has a malformed {column!r} value {value!r}: "
        f"expected {expected}."
    )


def _require_float(row: dict[str, object], column: str, *, identifier: str) -> float:
    value = _require(row, column, identifier=identifier)
    message = _malformed(identifier, column, value, "a number")
    if not isinstance(value, str | Real):
        raise MsnlibParseError(message)
    try:
        return float(value)
    except ValueError as exc:
        raise MsnlibParseError(message) from exc


def _require_peaks(
    row: dict[str, object], column: str, *, identifier: str
) -> np.ndarray:
    value = _require(row, column, identifier=identifier)
    try:
        return np.array([float(peak) for peak in str(value).split(",")], dtype=float)
    except ValueError as exc:
        raise MsnlibParseError(
            _malformed(identifier, column, value, "comma-separated numbers")
        ) from exc


def _require_ionization_mode(row: dict[str, object], *, identifier: str) -> str:
    """`row`'s Ionization mode: its `ionmode`, checked against its `charge`'s sign."""
    ionmode = _require(row, "ionmode", identifier=identifier)
    if ionmode not in IONIZATION_MODES:
        raise MsnlibParseError(
            _malformed(
                identifier, "ionmode", ionmode, "exactly 'positive' or 'negative'"
            )
        )
    charge = _require_float(row, "charge", identifier=identifier)
    if not charge.is_integer() or charge == 0:
        raise MsnlibParseError(
            _malformed(identifier, "charge", row["charge"], "a non-zero integer")
        )
    if (charge > 0) != (ionmode == "positive"):
        raise MsnlibParseError(
            f"Row {identifier!r} has ionmode {ionmode!r} but charge {int(charge)}, "
            "whose sign disagrees."
        )
    return str(ionmode)


def _require_true_inchikey(row: dict[str, object], *, identifier: str) -> str:
    """The first block of `row`'s InChIKey."""
    inchikey = str(_require(row, "inchikey", identifier=identifier))
    if not _INCHIKEY.fullmatch(inchikey):
        raise MsnlibParseError(
            _malformed(identifier, "inchikey", inchikey, "a 27-character InChIKey")
        )
    return inchikey[:INCHIKEY_FIRST_BLOCK_LENGTH]


def _require_instrument_type(
    row: dict[str, object], *, identifier: str
) -> InstrumentType:
    raw_value = str(_require(row, "instrument_type", identifier=identifier))
    try:
        return resolve_instrument_type(raw_value)
    except UnknownInstrumentError as exc:
        raise MsnlibParseError(f"Row {identifier!r}: {exc}") from exc


def _require_raw_file(row: dict[str, object], *, identifier: str) -> str:
    """The one raw mzML file `row`'s MS2 scans come from.

    `source_scan_usi` names it where present; otherwise `usi` does (for the
    libraries whose `usi` points at the raw file rather than a library MGF).
    """
    column = "usi" if _is_blank(row["source_scan_usi"]) else "source_scan_usi"
    value = str(_require(row, column, identifier=identifier))
    raw_files = set(_USI_RAW_FILE.findall(value))
    if len(raw_files) != 1:
        raise MsnlibParseError(
            _malformed(identifier, column, value, "USIs naming exactly one raw file")
        )
    return raw_files.pop()


def _require_scans(row: dict[str, object], *, identifier: str) -> tuple[int, ...]:
    """`row`'s MS2 scan numbers, ascending, from e.g. `217-219,425-427`."""
    value = str(_require(row, "scans", identifier=identifier))
    message = _malformed(
        identifier, "scans", value, "comma-separated scan numbers or ranges"
    )
    if not _SCANS.fullmatch(value):
        raise MsnlibParseError(message)
    scans: set[int] = set()
    for part in value.split(","):
        first, _, last = part.partition("-")
        start, stop = int(first), int(last or first)
        if stop < start:
            raise MsnlibParseError(message)
        scans.update(range(start, stop + 1))
    return tuple(sorted(scans))


def _sirius_spectrum(
    *,
    identifier: str,
    mzs: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float,
    ionization_mode: str,
    instrument_type: InstrumentType,
) -> Spectrum:
    """The spectrum SIRIUS is given: only what a *measured* spectrum carries.

    Built from an allow-list, not "every column the parquet has": a ground
    truth spectrum that tells SIRIUS the answer measures a SIRIUS no field
    run gets. Measured against real SIRIUS 6.3.3 runs over 25 QTOF spectra of
    the former ground truth set: leaving `smiles`/`inchikey` in the MGF moved
    top-1 accuracy from 55% to 68%. Excluded, and why:

    - `smiles`, `inchikey`, `formula`: the answer itself.
    - `parent_mass`: the neutral mass derived from the true formula.
    - `adduct`: never known for a field mzML run (SIRIUS detects it).
      Supplying it changed 25% of top-1 candidates, so keeping it would
      calibrate against conditions field runs cannot reproduce.
    - `collision_energy`: an `ALL_ENERGIES` spectrum merges several energies
      (e.g. `20.0,30.0,60.0`), which no single field spectrum carries.

    What remains is instrument-side fact plus the fields SIRIUS's pre-picked
    import path requires -- including `feature_id`, which comes back as
    `AlignedFeature.external_feature_id` and is how a run's features are
    joined to their true InChIKeys. `charge` follows the Ionization mode.
    The measured MS1 scan (`GroundTruthSpectrum.ms1`) is likewise instrument
    data, not the answer (ADR 0003).
    """
    order = np.argsort(mzs, kind="stable")
    return Spectrum(
        mz=mzs[order],
        intensities=intensities[order],
        metadata={
            "feature_id": identifier,
            "identifier": identifier,
            "pepmass": precursor_mz,
            "precursor_mz": precursor_mz,
            "charge": _MGF_CHARGE_BY_IONIZATION_MODE[ionization_mode],
            "ms_level": 2,
            "instrument_type": instrument_type,
        },
        metadata_harmonization=False,
    )


def _parse_row(row: dict[str, object], *, row_number: int) -> GroundTruthSpectrum:
    if _is_blank(row["identifier"]):
        raise MsnlibParseError(
            f"Parquet row {row_number} is missing a required 'identifier' value."
        )
    identifier = str(row["identifier"])
    mzs = _require_peaks(row, "mzs", identifier=identifier)
    intensities = _require_peaks(row, "intensities", identifier=identifier)
    if mzs.shape != intensities.shape:
        raise MsnlibParseError(
            f"Row {identifier!r} has {mzs.size} 'mzs' but {intensities.size} "
            "'intensities' values."
        )
    _require(row, "adduct", identifier=identifier)
    ionization_mode = _require_ionization_mode(row, identifier=identifier)
    instrument_type = _require_instrument_type(row, identifier=identifier)
    return GroundTruthSpectrum(
        identifier=identifier,
        true_inchikey=_require_true_inchikey(row, identifier=identifier),
        ionization_mode=ionization_mode,
        instrument_type=instrument_type,
        spectrum=_sirius_spectrum(
            identifier=identifier,
            mzs=mzs,
            intensities=intensities,
            precursor_mz=_require_float(row, "precursor_mz", identifier=identifier),
            ionization_mode=ionization_mode,
            instrument_type=instrument_type,
        ),
        raw_file=_require_raw_file(row, identifier=identifier),
        scans=_require_scans(row, identifier=identifier),
    )


def read_ground_truth_set(parquet_path: Path) -> GroundTruthSet:
    """Filter and parse MSnLib's parquet into the Ground truth set, in parquet order.

    Filters apply in order, each to the rows the previous one kept: `spectype
    == "ALL_ENERGIES"`, then a null `other_matched_compounds`, then an adduct
    outside `UNDETECTABLE_ADDUCTS`. Only kept rows are parsed, so a dropped
    row is never validated.

    Raises:
        MsnlibParseError: the parquet lacks one of `MSNLIB_COLUMNS`, or a
            kept row has a blank or malformed required value, an `ionmode`
            other than exactly `positive`/`negative` or disagreeing with its
            `charge`'s sign, or an instrument no Instrument type is known for.
    """
    missing = sorted(set(MSNLIB_COLUMNS) - set(pq.read_schema(parquet_path).names))
    if missing:
        raise MsnlibParseError(
            f"{parquet_path} is missing required MSnLib column(s): "
            f"{', '.join(missing)}."
        )
    df = pd.read_parquet(parquet_path, columns=list(MSNLIB_COLUMNS))
    all_energies = df[df["spectype"] == "ALL_ENERGIES"]
    unambiguous = all_energies[all_energies["other_matched_compounds"].isna()]
    kept = unambiguous[~unambiguous["adduct"].isin(UNDETECTABLE_ADDUCTS)]
    spectra = tuple(
        _parse_row(row, row_number=row_number)
        for row_number, row in zip(kept.index, kept.to_dict("records"), strict=True)
    )
    return GroundTruthSet(
        spectra=spectra,
        filter_counts=FilterCounts(
            kept=len(spectra),
            dropped_not_all_energies=len(df) - len(all_energies),
            dropped_ambiguous_identity=len(all_energies) - len(unambiguous),
            dropped_undetectable_adduct=len(unambiguous) - len(kept),
        ),
    )


@dataclass(frozen=True, slots=True)
class ChunkGroup:
    """The MGF chunks of one (Ionization mode, Instrument type)."""

    ionization_mode: str
    instrument_type: InstrumentType
    paths: tuple[Path, ...]


def write_sirius_chunks(
    spectra: Sequence[GroundTruthSpectrum], output_dir: Path
) -> list[ChunkGroup]:
    """Group `spectra` by (Ionization mode, Instrument type) into `SIRIUS_CHUNK_SIZE`-capped MGFs.

    Chunks are named from both, e.g. `orbitrap_negative_0.mgf`. Groups come
    in order of their first spectrum. A spectrum's measured MS1 scan, when it
    has one, is written right before its MS2 under the same `FEATURE_ID`, so
    SIRIUS imports both as one feature.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, InstrumentType], list[GroundTruthSpectrum]] = {}
    for spectrum in spectra:
        key = (spectrum.ionization_mode, spectrum.instrument_type)
        grouped.setdefault(key, []).append(spectrum)

    groups: list[ChunkGroup] = []
    for (ionization_mode, instrument_type), group in grouped.items():
        paths: list[Path] = []
        for chunk_number, start in enumerate(range(0, len(group), SIRIUS_CHUNK_SIZE)):
            path = (
                output_dir
                / f"{instrument_type.lower()}_{ionization_mode}_{chunk_number}.mgf"
            )
            save_as_mgf(
                [
                    block
                    for s in group[start : start + SIRIUS_CHUNK_SIZE]
                    for block in (s.ms1, s.spectrum)
                    if block is not None
                ],
                str(path),
                file_mode="w",
            )
            paths.append(path)
        groups.append(
            ChunkGroup(
                ionization_mode=ionization_mode,
                instrument_type=instrument_type,
                paths=tuple(paths),
            )
        )
    return groups

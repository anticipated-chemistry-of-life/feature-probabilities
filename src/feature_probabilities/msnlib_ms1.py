"""Measured MS1 scans for the Ground truth set (GLOSSARY.md; ADR 0003).

MSnLib's parquet carries only each spectrum's merged MS2 peaks. Without an
MS1 scan SIRIUS sees no isotope pattern and no co-measured ions of the same
compound, so it never assigns an adduct outside its fallback list. This
module gives each spectrum the MS1 scan its precursor was measured in, read
from MSnLib's own raw mzML files:

- :func:`fetch_msnlib_mzml_archives` downloads the mzML zip archives of
  Zenodo record `MSNLIB_MZML_ZENODO_RECORD` once into a local cache, each
  verified against its pinned MD5.
- :func:`attach_measured_ms1` reads each spectrum's raw file straight from
  its archive. Of the MS1 scans immediately preceding the spectrum's MS2
  scans, it takes the one where the precursor peak (within
  `PRECURSOR_TOLERANCE_PPM`) is most intense, cut to `MS1_WINDOW_DA` either
  side of the precursor.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import re
import urllib.request
import zipfile
from bisect import bisect_left
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
from matchms import Spectrum
from pyteomics import mzml

from feature_probabilities.msnlib import MsnlibFetchError, MsnlibParseError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from feature_probabilities.msnlib import GroundTruthSpectrum

#: Zenodo record "MsnLib multi-stage fragmentation mass spectral libraries -
#: mzml positive and negative": every raw file the Ground truth set's
#: spectra were merged from.
MSNLIB_MZML_ZENODO_RECORD = "15683784"

#: Each of the record's archives, by name, with its MD5 as Zenodo lists it.
MSNLIB_MZML_ARCHIVES = {
    "20220601_mzml_mce_bioactive_positive.zip": "4b19bc3082e036c895002b72a9744510",
    "20230130_mzml_mce_bioactive_negative.zip": "b938218b68053aa22a09feb4817f73c6",
    "20230404_mzml_pluskal_nih_negative.zip": "3ee592f53a143d1791f416bf767613db",
    "20230404_mzml_pluskal_nih_positive.zip": "660ed1bb97d0a7f6200edccd86624c88",
    "20231123_mzml_mce_scaffold_negative.zip": "51d4c420552729534748a8d2a9a2caa8",
    "20231123_mzml_mce_scaffold_positive.zip": "63b99c669125d089a44af2911fce7c76",
    "20231124_mzml_otavapep_negative.zip": "68887e1049c993da8df184b517f638b8",
    "20231124_mzml_otavapep_positive.zip": "44ec77fe54eb742facc2b9bd2958f3ac",
    "mzml_20240405_pluskal_enammol_MSn_negative.zip": "88c3ef947075687c021eed3c04d592e1",
    "mzml_20240405_pluskal_enammol_MSn_positive.zip": "9f1be3f82797c5a22bd2f380d4a15439",
    "mzml_20240408_pluskal_mcedrug_MSn_negative.zip": "22b88fecf3bb7644ff8909e7635ea4ad",
    "mzml_20240408_pluskal_mcedrug_MSn_positive.zip": "9b2748361415f400644873ffed552337",
    "mzml_20240502_pluskal_enamdisc_MSn_negative.zip": "245774ab6e3ad8266f863d32814e5ea7",
    "mzml_20240502_pluskal_enamdisc_MSn_positive.zip": "daa92bc6f4bc28579d2895a55eab979e",
    "mzml_20241113_pluskal_mcediv_20k_MSn_negative.zip": "981105ddee43b472b1f677162ed093ac",
    "mzml_20241113_pluskal_mcediv_20k_MSn_positive.zip": "bb0d2e16a721b920abf7615b39a02be9",
    "mzml_20241120_pluskal_targetmolnphts_MSn_negative.zip": "5d185be5694f481ffd06da3f03d1d737",
    "mzml_20241120_pluskal_targetmolnphts_MSn_positive.zip": "f8238837af88f91729a211d5651f40d1",
}

_ARCHIVE_URL = "https://zenodo.org/api/records/{record}/files/{name}/content"

#: Where the archives (21.6 GB) are kept between runs.
MZML_CACHE_DIR = Path.home() / ".cache" / "feature-probabilities" / "msnlib-mzml"

#: Half-width of the MS1 window kept around the precursor. Wide enough to
#: hold the ions SIRIUS pairs a precursor with to recognise its adduct
#: (e.g. `[M-H]-` 46 Da below `[M+FA]-`), narrow enough to leave out most
#: other compounds of the same injection (ADR 0003).
MS1_WINDOW_DA = 50.0

#: How close an MS1 peak must be to the precursor m/z to count as it.
PRECURSOR_TOLERANCE_PPM = 10.0

_SCAN_NUMBER = re.compile(r"\bscan=(\d+)\b")
_DOWNLOAD_CHUNK_BYTES = 1 << 20


def fetch_msnlib_mzml_archives(cache_dir: Path = MZML_CACHE_DIR) -> list[Path]:
    """Every archive of `MSNLIB_MZML_ARCHIVES`, downloaded into `cache_dir` if absent.

    An archive is only moved into place once its MD5 matches, so one that
    is present is complete; it is never downloaded again.

    Raises:
        MsnlibFetchError: a download failed or its MD5 does not match.
    """
    record_dir = cache_dir / MSNLIB_MZML_ZENODO_RECORD
    record_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, md5 in MSNLIB_MZML_ARCHIVES.items():
        path = record_dir / name
        if not path.exists():
            _download(
                _ARCHIVE_URL.format(record=MSNLIB_MZML_ZENODO_RECORD, name=name),
                path,
                md5,
            )
        paths.append(path)
    return paths


def _download(url: str, path: Path, md5: str) -> None:
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    digest = hashlib.md5()
    try:
        with (
            urllib.request.urlopen(url, timeout=60) as response,
            partial.open("wb") as out,
        ):
            while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                digest.update(chunk)
                out.write(chunk)
    except OSError as exc:
        partial.unlink(missing_ok=True)
        raise MsnlibFetchError(
            f"Could not download MSnLib mzML archive {path.name!r} from {url} "
            f"(original error: {exc})."
        ) from exc
    if digest.hexdigest() != md5:
        partial.unlink()
        raise MsnlibFetchError(
            f"MSnLib mzML archive {path.name!r} from {url} has MD5 "
            f"{digest.hexdigest()}, expected {md5}."
        )
    partial.replace(path)


@dataclass(frozen=True, slots=True)
class MeasuredMs1:
    """The spectra with their measured MS1 attached, in input order."""

    spectra: tuple[GroundTruthSpectrum, ...]
    #: Spectra no MS1 scan preceding their MS2 scans holds the precursor of;
    #: they go to SIRIUS with MS2 only.
    without_ms1: int


def attach_measured_ms1(
    spectra: Sequence[GroundTruthSpectrum], archives: Sequence[Path]
) -> MeasuredMs1:
    """Set each spectrum's `ms1` from its raw file in `archives`.

    Raw files are read in parallel, one per task, by up to one process per
    CPU.

    Raises:
        MsnlibFetchError: an archive is not a readable zip, a raw file is in
            two archives, or a spectrum's raw file is in none of them.
        MsnlibParseError: a raw file lacks one of a spectrum's MS2 scans, or
            has a spectrum without a scan number.
    """
    raw_files = _index_archives(archives)
    requests: dict[str, list[_Ms1Request]] = {}
    for spectrum in spectra:
        if spectrum.raw_file not in raw_files:
            raise MsnlibFetchError(
                f"Spectrum {spectrum.identifier!r}: raw file {spectrum.raw_file!r} "
                "is in none of the MSnLib mzML archives."
            )
        requests.setdefault(spectrum.raw_file, []).append(
            _Ms1Request(
                identifier=spectrum.identifier,
                precursor_mz=float(spectrum.spectrum.get("precursor_mz")),
                scans=spectrum.scans,
            )
        )

    windows: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    if requests:
        workers = min(os.cpu_count() or 1, len(requests))
        # Forked workers segfaulted once the parent had loaded the rest of the
        # pipeline's native libraries (seen in the full test suite);
        # forkserver workers start from a clean process.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("forkserver")
        ) as pool:
            futures = [
                pool.submit(_measured_ms1_windows, *raw_files[raw_file], file_requests)
                for raw_file, file_requests in requests.items()
            ]
            for future in futures:
                windows.update(future.result())

    attached = tuple(
        spectrum
        if windows[spectrum.identifier] is None
        else replace(
            spectrum, ms1=_ms1_spectrum(spectrum, *windows[spectrum.identifier])
        )
        for spectrum in spectra
    )
    return MeasuredMs1(
        spectra=attached,
        without_ms1=sum(spectrum.ms1 is None for spectrum in attached),
    )


def _index_archives(archives: Sequence[Path]) -> dict[str, tuple[Path, str]]:
    """Each raw file's (archive, member name), keyed by the raw file's stem."""
    index: dict[str, tuple[Path, str]] = {}
    for archive in archives:
        try:
            with zipfile.ZipFile(archive) as zf:
                members = [name for name in zf.namelist() if name.endswith(".mzML")]
        except (OSError, zipfile.BadZipFile) as exc:
            raise MsnlibFetchError(
                f"MSnLib mzML archive {str(archive)!r} is not a readable zip; "
                f"delete it to download it again (original error: {exc})."
            ) from exc
        for member in members:
            stem = PurePosixPath(member).stem
            if stem in index:
                raise MsnlibFetchError(
                    f"Raw file {stem!r} is in both {index[stem][0].name!r} and "
                    f"{archive.name!r}."
                )
            index[stem] = (archive, member)
    return index


@dataclass(frozen=True, slots=True)
class _Ms1Request:
    identifier: str
    precursor_mz: float
    scans: tuple[int, ...]


def _measured_ms1_windows(
    archive: Path, member: str, requests: list[_Ms1Request]
) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    """Each request's MS1 window (m/z, intensity) from raw file `member`, or `None`."""
    scan_numbers: set[int] = set()
    ms1_peaks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with zipfile.ZipFile(archive) as zf, zf.open(member) as raw_file:
        for scan in mzml.MzML(raw_file, use_index=False):
            match = _SCAN_NUMBER.search(scan["id"])
            if match is None:
                raise MsnlibParseError(
                    f"Raw file {member!r} has a spectrum without a scan number: "
                    f"{scan['id']!r}."
                )
            number = int(match.group(1))
            scan_numbers.add(number)
            if scan["ms level"] == 1:
                ms1_peaks[number] = (
                    np.asarray(scan["m/z array"], dtype=float),
                    np.asarray(scan["intensity array"], dtype=float),
                )
    ms1_scans = sorted(ms1_peaks)

    windows: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
    for request in requests:
        missing = [scan for scan in request.scans if scan not in scan_numbers]
        if missing:
            raise MsnlibParseError(
                f"Spectrum {request.identifier!r}: raw file {member!r} has no "
                f"scan(s) {missing}."
            )
        best_scan, best_intensity = None, 0.0
        for scan in request.scans:
            position = bisect_left(ms1_scans, scan)
            if position == 0:
                continue
            preceding = ms1_scans[position - 1]
            intensity = _precursor_intensity(
                *ms1_peaks[preceding], request.precursor_mz
            )
            if intensity > best_intensity:
                best_scan, best_intensity = preceding, intensity
        if best_scan is None:
            windows[request.identifier] = None
            continue
        mzs, intensities = ms1_peaks[best_scan]
        keep = np.abs(mzs - request.precursor_mz) <= MS1_WINDOW_DA
        windows[request.identifier] = (mzs[keep], intensities[keep])
    return windows


def _precursor_intensity(
    mzs: np.ndarray, intensities: np.ndarray, precursor_mz: float
) -> float:
    """The highest intensity within `PRECURSOR_TOLERANCE_PPM` of `precursor_mz`, or 0."""
    near = np.abs(mzs - precursor_mz) <= precursor_mz * PRECURSOR_TOLERANCE_PPM * 1e-6
    return float(intensities[near].max()) if near.any() else 0.0


def _ms1_spectrum(
    spectrum: GroundTruthSpectrum, mzs: np.ndarray, intensities: np.ndarray
) -> Spectrum:
    """The MS1 block SIRIUS reads: same `FEATURE_ID`, `PEPMASS`, `CHARGE` as the MS2."""
    return Spectrum(
        mz=mzs,
        intensities=intensities,
        metadata={
            "feature_id": spectrum.identifier,
            "pepmass": spectrum.spectrum.get("pepmass"),
            "charge": spectrum.spectrum.get("charge"),
            "mslevel": 1,
        },
        metadata_harmonization=False,
    )

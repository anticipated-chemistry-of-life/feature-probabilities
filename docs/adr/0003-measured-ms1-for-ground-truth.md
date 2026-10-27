# Ground truth spectra carry their measured MS1 scan, cut to ±50 Da

MSnLib's parquet holds only each spectrum's merged MS2 peaks. With MS2 alone, SIRIUS sees no isotope pattern and no other ions of the same compound, so it only ever tries its four fallback adducts. Every Ground truth spectrum whose MSnLib adduct is `[M+FA]-`, `[M+Cl]-`, `[M+H-H2O]+`, `[M+H-2H2O]+` or `[M+NH4]+` (21.6% of the set) therefore had essentially no Correct assignment. Field runs do not have this problem: SIRIUS reads their MS1 and detects these adducts. So `generate-groundtruth` now writes each spectrum's measured MS1 scan before its MS2, under the same `FEATURE_ID`.

The MS1 scan comes from MSnLib's raw mzML files (Zenodo record 15683784, 18 zip archives, 21.6 GB, MD5-pinned in `msnlib_ms1.py`). Every spectrum names its raw file in `source_scan_usi` (the mcediv and targetmol libraries) or `usi` (all other libraries), and its MS2 scans in `scans`. Of the MS1 scans immediately preceding those MS2 scans, the one where the precursor peak (within 10 ppm) is most intense is used. It is cut to the precursor m/z ± 50 Da. A spectrum with no such scan goes to SIRIUS with MS2 only, and the run summary counts it. A raw file or scan that cannot be found fails the run before any SIRIUS work.

The MS1 scan is instrument data, not the answer, so it falls under the same rule as `_sirius_spectrum`'s allow-list. Adduct, formula and structure still never reach the MGF.

## Evidence

The pilot covered 512 spectra from 80 raw files: 40 per Ionization mode, 30 of them chosen because they contain rare-adduct spectra. Each spectrum went through SIRIUS 6.5.4 under the production config with four MS1 variants. The table shows the share whose correct molecule is in the top 200 / at rank 1 (%):

| MSnLib adduct | n | no MS1 | full scan | precursor −1.5/+5.5 Da | ±50 Da |
|---|---|---|---|---|---|
| `[M-H]-` | 138 | 60.1 / 42.0 | 61.6 / 43.5 | 60.9 / 44.2 | 60.9 / 42.0 |
| `[M+FA]-` | 62 | 0 / 0 | 41.9 / 32.3 | 0 / 0 | 51.6 / 38.7 |
| `[M+Cl]-` | 37 | 0 / 0 | 16.2 / 16.2 | 0 / 0 | 13.5 / 13.5 |
| `[M+H]+` | 211 | 49.8 / 33.6 | 51.7 / 34.1 | 53.6 / 34.6 | 53.6 / 34.1 |
| `[M+H-H2O]+` | 28 | 0 / 0 | 53.6 / 25.0 | 0 / 0 | 57.1 / 28.6 |
| `[M+NH4]+` | 25 | 0 / 0 | 36.0 / 24.0 | 52.0 / 28.0 | 60.0 / 36.0 |
| `[M+Na]+` | 9 | 22.2 / 11.1 | 22.2 / 11.1 | 22.2 / 11.1 | 22.2 / 11.1 |
| All | 512 | 37.1 / 25.4 | 49.6 / 33.6 | 41.4 / 27.7 | 52.3 / 34.6 |

SIRIUS recognises these adducts in its formula step, from MS1 peaks of other ions of the same compound. It does not do so at import: `detected_adducts` stayed `[M + ?]` in every variant. That explains the variants:

- The isotope-only window recovered only `[M+NH4]+`, whose `[M+Na]+` partner sits 4.955 Da higher, inside the window (24 of 25 spectra had that peak).
- ±50 Da also holds the partners of `[M+FA]-` (46 Da), `[M+Cl]-` (36 Da) and `[M+H-H2O]+` (18 Da).
- The full scan adds unrelated compounds from the same injection. Against ±50 Da, it found the true formula among the candidates less often (76.8% vs 79.9%) and matched MSnLib's adduct at rank 1 less often (71.7% vs 74.8%).

Weighting each adduct's gain by its share of the 88,677-spectrum set projects about +11 points top-200 (30.8% → ≈42%) and +6.6 points top-1 (17.5% → ≈24%). This is extrapolated from the pilot, not measured on the full set.

## Considered Options

- **Supply MSnLib's adduct or formula.** Rejected: it tells SIRIUS the answer, which no field run gets (`_sirius_spectrum`).
- **Add the rare adducts to `fallback_adducts`.** Not chosen: every field run would then try them too, whether or not its MS1 supports them.
- **Drop the rare-adduct spectra like `UNDETECTABLE_ADDUCTS`.** Not chosen: that removes 21.6% of the set instead of letting SIRIUS assign their adducts the way it does for field runs.
- **Full MS1 scan, or the isotope window only.** Rejected on the pilot evidence above.
- **Import the raw mzML through SIRIUS's LC-MS preprocessing, as for field runs.** Not chosen: MSnLib's flow-injection runs mix several compounds per injection, and each Ground truth spectrum is MSnLib's merge of several MS2 scans, which preprocessing would not reproduce.
- **MassIVE MSV000094528 as the raw file source.** Rejected: it lacks the mcediv and targetmol raw files and 126 others, covering 31,754 Ground truth spectra. The Zenodo record has all of them.

## Consequences

- The first `generate-groundtruth` run, including `--smoke-test`, downloads 21.6 GB into `~/.cache/feature-probabilities/msnlib-mzml/` (about 17 minutes at the measured 21 MB/s). Later runs read the cache.
- Every chunk's MGF changes, so its `input_file_checksum` does too. Earlier MS2-only ground truth runs are never cache hits. They are also not replaced, and `fit-kde` reads every ground truth run, so regenerate into a fresh database or delete the old ground truth runs first.
- MSnLib's MS1 comes from flow injection at standard concentrations and may be cleaner than field MS1. Calibration may still be optimistic for adducts SIRIUS detects from MS1.

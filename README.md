# feature-probabilities

## SIRIUS credentials

`generate-groundtruth` (and every other CLI that talks to SIRIUS) needs a
locally-installed SIRIUS desktop application plus an account login. Provide
these three variables:

- `SIRIUS_USER` / `SIRIUS_PW` — your SIRIUS account credentials.
- `SIRIUS_EXE` — absolute path to the SIRIUS executable (only needed if
  `sirius` isn't already on your `PATH`, e.g. macOS:
  `/Applications/sirius.app/Contents/MacOS/sirius`).

Set them either by:

1. Adding them to this repo's `.env` file (loaded automatically via
   `python-dotenv`; `.env` is gitignored, never commit real credentials):

   ```
   SIRIUS_USER=you@example.com
   SIRIUS_PW=your-password
   SIRIUS_EXE=/Applications/sirius.app/Contents/MacOS/sirius
   ```

2. Or exporting them in your shell profile (`~/.bashrc`, `~/.zshrc`, etc.):

   ```
   export SIRIUS_USER=you@example.com
   export SIRIUS_PW=your-password
   export SIRIUS_EXE=/Applications/sirius.app/Contents/MacOS/sirius
   ```

Variables already exported in your shell take precedence over `.env` values.

## Stalled SIRIUS jobs

SIRIUS 6.5.4 can deadlock its job pool on lipid-heavy input, leaving a job
stuck forever at 0% CPU. When an analysis job's progress stands still for
20 minutes (`JOB_STALL_TIMEOUT_SECONDS` in `sirius.py`), the SIRIUS process
is killed and the job fails with `SiriusJobStalledError`.
`generate-groundtruth` records the chunk as failed and continues with a
fresh SIRIUS; `annotate` does the same per mzML file. Rerunning retries only
the failed chunks or files.

## Ground truth

`generate-groundtruth` builds the Ground truth set from MSnLib
(`msnlib_ms2.parquet` of the HuggingFace dataset
`salutmoic/msnlib-ms2-parquet`), in both Ionization modes. It fetches the
revision pinned as `ground_truth_revision` in `config.toml` through the
HuggingFace hub cache; `--ground-truth-revision` overrides it for one run.

Only `ALL_ENERGIES` spectra are kept. Spectra MSnLib also matched to other
library compounds are dropped, as are spectra whose adduct SIRIUS cannot
detect (`[M]+`, `[M-H2O]+`, `[M]-`). The run summary reports how many
spectra each filter dropped. A missing column, a blank or malformed value,
an `ionmode` that is not exactly `positive` or `negative` or disagrees with
the sign of `charge`, or an unknown instrument fails the run before any
SIRIUS work, naming the spectrum.

Each spectrum is sent to SIRIUS with the MS1 scan its precursor was
measured in, so SIRIUS can use the isotope pattern and recognise adducts
outside its fallback list, as it does for field runs (ADR 0003). The MS1
scan comes from MSnLib's raw mzML files: the Zenodo record's 18 zip
archives (21.6 GB, MD5-pinned in `msnlib_ms1.py`). They are downloaded on
the first run, `--smoke-test` included, into
`~/.cache/feature-probabilities/msnlib-mzml/`. Of the MS1 scans preceding
the spectrum's MS2 scans, the one where the precursor is most intense is
used, cut to the precursor m/z ± 50 Da and written right before the MS2
under the same `FEATURE_ID`. Spectra without such a scan go with MS2 only;
the run summary counts them. A raw file or MS2 scan missing from the
archives fails the run before any SIRIUS work.

Spectra are chunked by (Ionization mode, Instrument type), e.g.
`orbitrap_negative_0.mgf`, at most 2,000 spectra per chunk. Each chunk is
written with `CHARGE=1+` or `CHARGE=1-`, analysed under its Instrument
type's SIRIUS profile, and recorded as a ground truth run (source kind
`ground_truth_msnlib`) with its Ionization mode and Instrument type.
`--ionization-mode {positive,negative,all}` restricts which chunks run.
SIRIUS is never shown the answer: no adduct, structure, formula, parent
mass or collision energy reaches the MGF.

A database created before MSnLib ground truth is not migrated: delete it
and rerun `generate-groundtruth` and `annotate` (ADR 0002). Ground truth
runs from before measured MS1 are not replaced by a rerun, and `fit-kde`
reads every ground truth run: regenerate into a fresh database (ADR 0003).

## Calibration

`fit-kde` fits two densities per Ionization mode, `positive` and `negative`:
the correct density on correct assignments and the incorrect density on
incorrect ones (a candidate whose first-block InChIKey differs from the true
structure's). Each is a 2D Gaussian KDE over (ion mass, CSI:FingerID score)
with Scott's-rule bandwidth. Rows come from the Features kept by the
(true InChIKey, SIRIUS-detected adduct) dedup, including Features whose
truth ranked below Top-k. Instrument type does not split calibration, and
there is no fallback KDE (ADR 0002). The incorrect density is fitted on a
seeded uniform subsample of at most 20,000 incorrect rows per mode (ADR
0004); the correct density is never subsampled. The seed and size are stored
on the `kde_models` row, with the rows each mode used. If either mode has no correct or no incorrect
assignments, `fit-kde` fails and names that mode.

`annotate` scores each field run's Features with the correct density for the
Ionization mode recorded on that run. An artifact missing either density for
either mode, such as an old one-KDE-per-mode artifact, is rejected before
any mzML reaches SIRIUS; rerun `fit-kde`. A database from before this change
needs its `kde_models` and `calibration_scores` tables dropped first (`kde_models` gains non-null
columns and `calibration_scores` references it).

## Annotate

`annotate` reads each mzML's Ionization mode and Instrument type from that
file's row in the metadata table (`--metadata-csv`), so one batch can mix
polarities and instruments. Two columns are required:

- `ionization_mode`: exactly `positive` or `negative`.
- `instrument_model`: the instrument product name, e.g.
  `LC-MS QExactive Plus`. It resolves to an Instrument type, `Orbitrap` or
  `QTOF`, after normalization (lowercased, ASCII letters and digits only), so
  `LCMS – Synapt XS` resolves to `QTOF`. The mapping table is
  `_INSTRUMENT_TYPE_BY_NORMALIZED_VALUE` in `instrument.py`; an unknown model
  needs a line added there.

Each field run records both values, is analysed under its Instrument type's
SIRIUS profile, and is scored with the KDE for its Ionization mode. Neither
column is copied into the Extract's extra metadata. A missing column, a
blank or invalid `ionization_mode`, or an `instrument_model` with no known
Instrument type fails the run with an error naming the file, just like an
mzML with no metadata row.

## Smoke test

`--smoke-test` runs the whole pipeline on a small, isolated input so a change
can be checked end to end in minutes. Every setting lives in `config.toml`'s
`[smoke_test]` section:

```
generate-groundtruth --smoke-test   # resets the smoke DB; first N spectra per Ionization mode
fit-kde --smoke-test                # fits from the smoke DB
annotate --smoke-test               # annotates data/smoke_test/ into the smoke DB and exports
```

The smoke DB, KDE pickles and export live under `db/smoke_test/` and
`models/smoke_test/`, never touching the real database. `--db` cannot be
combined with `--smoke-test`; `annotate`'s other flags override their
`[smoke_test]` defaults.

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

The summary also compares, per Ionization mode, incorrect rows of Features
whose truth is listed with those whose truth is unlisted: row count, CSI score
median and q10/q90, and the mean and q10/q90 of ln g0_listed − ln g0_unlisted
at the pooled g0's training subsample. It is diagnostic only: the artifact and
`kde_models` row stay pooled, and an empty subpopulation is reported, not an
error.

`annotate` calibrates each field run's Features: for every (Feature, candidate)
pair it stores the natural log of the correct density and of the incorrect
density, both for the Ionization mode recorded on that run, in the
`calibration_densities` table (one row per annotation and KDE model). Both
Feature-probability table exports (`annotate --export` and
`queries/feature_molecule_scores.sql`) carry them as `log_correct_density` and
`log_incorrect_density`; the old `calibration_score` column is gone. The
incorrect density (g0) is the one fitted on the seeded subsample above, so the
difference of the two columns is a log likelihood ratio on the subsample's
scale (ADR 0004). An artifact missing either density for either mode, such as
an old one-KDE-per-mode artifact, is rejected before any mzML reaches SIRIUS;
rerun `fit-kde`.

### Evaluating the weighting

`evaluate-calibration` cross-validates two ways of weighting a Feature's
candidates on the Ground truth set: the **baseline**, correct density over the
Feature's best row (what the MRF uses today), and the **likelihood ratio**,
correct density over incorrect density. It takes `--config`, `--db` and
`--seed` (default 0), reads the same Features as `fit-kde` and prints both
methods side by side. It writes nothing: no artifact, no `kde_models` row, no
file. With no ground truth in the DB it fails and points at
`generate-groundtruth`.

Each distinct true InChIKey goes to exactly one of 5 folds by a seeded
shuffle. In every fold both densities are refitted on the training Features
with the function `fit-kde` uses, so the incorrect density is subsampled from
training rows only. Weights are w_j = g1_j / max_k g1_k (baseline) or
w_j = g1_j / g0_j (likelihood ratio), with a uniform prior over candidates; a
candidate's posterior is w_j / (Σw + c) and the Unknown molecule's is
c / (Σw + c). Per method and fold, c is fitted on the training Features by a
bounded 1-D search over ln c minimising the mean −ln posterior of the truth
(the Unknown molecule for a Feature without a correct assignment). The report,
pooled over the held-out folds, gives:

- top-1 and top-5 accuracy over Features with a correct assignment, ranking
  candidates only, ties broken by input order;
- a 10-bin equal-width reliability table over all held-out candidate rows:
  count, mean posterior, fraction correct;
- the mean and median Unknown-molecule posterior over Features without a
  correct assignment;
- the held-out mean log-loss, and each fold's fitted c.

The same data and seed give an identical report. Fewer than 5 distinct true
InChIKeys, or a training split left without an Ionization mode's correct or
incorrect candidates, fails the run.

### Migrating an existing database

A database from before this change keeps all its SIRIUS results. Only two
derived tables change: `kde_models` (gains non-null columns) and
`calibration_scores` (now `calibration_densities`, referencing `kde_models`).
Drop and recreate only those two, then refit and recalibrate:

1. Drop the tables, referencing table first: `DROP TABLE calibration_scores;`
   (and `DROP TABLE IF EXISTS calibration_densities;`), then
   `DROP TABLE kde_models;`. Leave every other table alone.
2. Run `fit-kde`. It recreates the dropped tables, empty, and fits a new KDE
   model.
3. Rerun `annotate` on the same batches. Every file is a SIRIUS cache hit, so
   SIRIUS does not rerun; the annotations are recalibrated against the new
   KDE model.

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
SIRIUS profile, and is calibrated with the densities for its Ionization mode. Neither
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

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

## Smoke test

`--smoke-test` runs the whole pipeline on a small, isolated input so a change
can be checked end to end in minutes. Every setting lives in `config.toml`'s
`[smoke_test]` section:

```
generate-groundtruth --smoke-test   # resets the smoke DB; first N spectra per instrument type
fit-kde --smoke-test                # fits from the smoke DB
annotate --smoke-test               # annotates data/smoke_test/ into the smoke DB and exports
```

The smoke DB, KDE pickles and export live under `db/smoke_test/` and
`models/smoke_test/`, never touching the real database. `--db` cannot be
combined with `--smoke-test`; `annotate`'s other flags override their
`[smoke_test]` defaults.

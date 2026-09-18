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

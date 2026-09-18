"""Species/extract metadata upsert from a user-supplied CSV.

`fp-annotate` always requires and upserts a metadata CSV on every invocation
(issue #9's user story 17) rather than needing a separate one-time ingestion
step -- there is no dedicated ingestion executable (issue #9's Out of Scope).
This module is standalone from SIRIUS entirely: it only touches the
`species`/`extracts` tables (`schema.py`) via a real SQLAlchemy `Session`.

The CSV's columns map onto the two tables as follows:

- `sample_code` (required) -- the upsert key, `extracts.sample_code`.
- `taxon_name` (required) -- `species.taxon_name`, the species upsert key.
- `ncbi_taxid`, `family` (optional) -- the rest of `species`.
- `organ` (optional) -- `extracts.organ`.
- every other column -- captured verbatim into `extracts.extra_metadata`
  rather than requiring a fixed schema, per issue #17's "any additional
  columns" requirement.

Two entry points cover the two ways later tickets consume this module:
`upsert_metadata_csv` upserts every row in one pass (this ticket's own
acceptance criteria), and `find_metadata_row_for_mzml` +
`upsert_metadata_row` together let `fp-annotate` (#23) look up and upsert
just the one row a given mzML file needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import select

from feature_probabilities.schema import Extract, Species

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

#: The upsert key column, both for `extracts.sample_code` and for matching
#: an mzML filename against a CSV row.
SAMPLE_CODE_COLUMN = "sample_code"

#: Columns that map directly onto `species` columns (besides the species
#: upsert key, `taxon_name`, which is also required -- see `REQUIRED_COLUMNS`).
SPECIES_COLUMNS = ("taxon_name", "ncbi_taxid", "family")

#: Columns that map directly onto `extracts` columns, besides
#: `SAMPLE_CODE_COLUMN` itself.
EXTRACT_COLUMNS = ("organ",)

#: Every column with a fixed destination. Anything else in the CSV lands in
#: `extracts.extra_metadata` instead of requiring a fixed schema.
PINNED_COLUMNS = (SAMPLE_CODE_COLUMN, *SPECIES_COLUMNS, *EXTRACT_COLUMNS)

#: Columns a row must carry a non-blank value for; anything else is optional.
REQUIRED_COLUMNS = (SAMPLE_CODE_COLUMN, "taxon_name")


class MetadataError(Exception):
    """Raised when a metadata CSV is malformed, or a row can't be upserted or found."""


def _clean(value: object) -> str | None:
    """Normalize a raw CSV cell: blank/NaN becomes `None`, else a stripped string."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_optional_int(
    value: str | None, *, column: str, sample_code: str, csv_path: Path
) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise MetadataError(
            f"Metadata CSV {csv_path}: sample_code={sample_code!r} has a "
            f"non-integer '{column}' value: {value!r}"
        ) from exc


def _extra_metadata(row: pd.Series) -> dict[str, str] | None:
    """Every non-pinned column's non-blank value, keyed by column name."""
    extra = {
        column: value
        for column in row.index
        if column not in PINNED_COLUMNS
        if (value := _clean(row[column])) is not None
    }
    return extra or None


def load_metadata_csv(csv_path: Path | str) -> pd.DataFrame:
    """Parse a metadata CSV, validating that every required column is present.

    Every cell is read as a string (rather than pandas' inferred numeric/bool
    dtypes) so extra columns land in `extra_metadata` exactly as written and
    `ncbi_taxid` parsing stays this module's own, explicit responsibility.

    Raises:
        MetadataError: `csv_path` doesn't exist, or the CSV is missing one of
            `REQUIRED_COLUMNS`.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise MetadataError(f"Metadata CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise MetadataError(
            f"Metadata CSV {path} is missing required column(s): {', '.join(missing)}"
        )
    return df


def find_metadata_row_for_mzml(
    df: pd.DataFrame, mzml_filename: str | Path
) -> pd.Series:
    """Find the metadata row for `mzml_filename`, matched by `sample_code`.

    Matches `Path(mzml_filename).stem` -- the filename without directory or
    extension -- against `sample_code` exactly, so both a bare sample code
    (e.g. `"EX-001"`) and a real mzML filename (e.g. `"EX-001.mzML"`) resolve
    the same row.

    Raises:
        MetadataError: no row's `sample_code` matches, or more than one does
            (an ambiguous CSV, which would otherwise silently pick one).
    """
    stem = Path(mzml_filename).stem
    matches = df.index[df[SAMPLE_CODE_COLUMN].map(_clean) == stem]
    if len(matches) == 0:
        raise MetadataError(
            f"No metadata row found for mzML file {mzml_filename!r} "
            f"(expected a row with sample_code={stem!r})"
        )
    if len(matches) > 1:
        raise MetadataError(
            f"Metadata CSV has {len(matches)} rows with sample_code={stem!r}; "
            f"expected exactly one for mzML file {mzml_filename!r}"
        )
    return df.loc[matches[0]]


def _get_or_create[T: (Species, Extract)](
    session: Session, model: type[T], **filters: str
) -> T:
    """Fetch the row matching `filters`, or insert-and-return a fresh one.

    Generic over `Species`/`Extract`: both are upserted by looking up on
    their unique key (`taxon_name`/`sample_code` respectively) and creating
    a new row only on a miss -- the shared shape both `_upsert_species` and
    `_upsert_extract` build on.
    """
    existing = session.scalars(select(model).filter_by(**filters)).one_or_none()
    if existing is not None:
        return existing
    created = model(**filters)
    session.add(created)
    return created


def _upsert_species(
    session: Session,
    taxon_name: str,
    row: pd.Series,
    *,
    sample_code: str,
    csv_path: Path,
) -> Species:
    species = _get_or_create(session, Species, taxon_name=taxon_name)
    species.ncbi_taxid = _parse_optional_int(
        _clean(row.get("ncbi_taxid")),
        column="ncbi_taxid",
        sample_code=sample_code,
        csv_path=csv_path,
    )
    species.family = _clean(row.get("family"))
    session.flush()
    return species


def _upsert_extract(
    session: Session, sample_code: str, species: Species, row: pd.Series
) -> Extract:
    extract = _get_or_create(session, Extract, sample_code=sample_code)
    extract.species_id = species.species_id
    extract.organ = _clean(row.get("organ"))
    extract.extra_metadata = _extra_metadata(row)
    session.flush()
    return extract


def upsert_metadata_row(
    session: Session, row: pd.Series, *, csv_path: Path | str = "<metadata row>"
) -> Extract:
    """Upsert one metadata row's `species`/`extracts` rows, keyed by `sample_code`.

    Inserts if no row with this `sample_code` (resp. `taxon_name`) exists
    yet; otherwise updates the existing row in place -- re-running with
    unchanged row content is a no-op, and a changed value overwrites the
    existing row rather than inserting a second one.

    `csv_path` is used only to make error messages actionable; pass it when
    calling this directly (e.g. `fp-annotate` looking up one row via
    `find_metadata_row_for_mzml`) so a bad row still names its source file.

    Raises:
        MetadataError: the row has no `sample_code`, no `taxon_name` (so it
            can't be resolved to a `species` row), or an unparseable
            `ncbi_taxid`.
    """
    path = Path(csv_path)
    sample_code = _clean(row.get(SAMPLE_CODE_COLUMN))
    if sample_code is None:
        raise MetadataError(
            f"Metadata CSV {path}: row {int(row.name) + 2} is missing a "
            f"'{SAMPLE_CODE_COLUMN}'"
        )

    taxon_name = _clean(row.get("taxon_name"))
    if taxon_name is None:
        raise MetadataError(
            f"Metadata CSV {path}: sample_code={sample_code!r} has no 'taxon_name', "
            f"so it can't be resolved to a species row"
        )

    species = _upsert_species(
        session, taxon_name, row, sample_code=sample_code, csv_path=path
    )
    return _upsert_extract(session, sample_code, species, row)


def upsert_metadata_csv(session: Session, csv_path: Path | str) -> list[Extract]:
    """Load `csv_path` and upsert every row's `species`/`extracts` rows.

    Raises:
        MetadataError: propagated from `load_metadata_csv` or
            `upsert_metadata_row` -- the first invalid row aborts the whole
            upsert (rows before it remain flushed on `session`, but nothing
            is committed here; callers control the transaction).
    """
    path = Path(csv_path)
    df = load_metadata_csv(path)
    return [
        upsert_metadata_row(session, row, csv_path=path) for _, row in df.iterrows()
    ]

"""Tests for the species/extract metadata CSV upsert.

Exercises the real SQLAlchemy models against a real (in-memory) DuckDB
engine, per the project's testing decisions -- no seam needed here, since
this module never touches SIRIUS.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.metadata import (
    MetadataError,
    find_metadata_row_for_mzml,
    load_metadata_csv,
    upsert_metadata_csv,
    upsert_metadata_row,
)
from feature_probabilities.schema import Extract, Species, create_database

BASE_CSV = """sample_code,taxon_name,ncbi_taxid,family,organ,collector
EX-001,Panthera leo,9689,Felidae,leaf,field team A
EX-002,Bos taurus,9913,Bovidae,root,field team B
"""


def write_csv(tmp_path: Path, text: str = BASE_CSV) -> Path:
    csv_path = tmp_path / "metadata.csv"
    csv_path.write_text(text)
    return csv_path


def test_upserting_a_fixture_csv_into_an_empty_db_inserts_expected_rows(
    tmp_path: Path,
) -> None:
    csv_path = write_csv(tmp_path)
    engine = create_database(":memory:")
    with Session(engine) as session:
        extracts = upsert_metadata_csv(session, csv_path)
        session.commit()

        assert len(extracts) == 2
        assert session.scalar(
            select(Species).where(Species.taxon_name == "Panthera leo")
        )
        ex001 = session.scalar(select(Extract).where(Extract.sample_code == "EX-001"))
        assert ex001 is not None
        assert ex001.organ == "leaf"
        assert ex001.extra_metadata == {"collector": "field team A"}
        species = session.get(Species, ex001.species_id)
        assert species is not None
        assert species.taxon_name == "Panthera leo"
        assert species.ncbi_taxid == 9689
        assert species.family == "Felidae"


def test_rerunning_the_upsert_with_unchanged_csv_does_not_duplicate_rows(
    tmp_path: Path,
) -> None:
    csv_path = write_csv(tmp_path)
    engine = create_database(":memory:")
    with Session(engine) as session:
        upsert_metadata_csv(session, csv_path)
        session.commit()

        upsert_metadata_csv(session, csv_path)
        session.commit()

        assert (
            session.scalar(
                select(Species.species_id).where(Species.taxon_name == "Bos taurus")
            )
            is not None
        )
        species_count = session.scalar(
            select(Species).where(Species.taxon_name == "Panthera leo")
        )
        assert species_count is not None
        extract_count = len(session.scalars(select(Extract)).all())
        species_total = len(session.scalars(select(Species)).all())
        assert extract_count == 2
        assert species_total == 2


def test_rerunning_with_a_changed_value_updates_the_existing_row_in_place(
    tmp_path: Path,
) -> None:
    csv_path = write_csv(tmp_path)
    engine = create_database(":memory:")
    with Session(engine) as session:
        upsert_metadata_csv(session, csv_path)
        session.commit()

        changed_csv = write_csv(
            tmp_path,
            BASE_CSV.replace(
                "EX-001,Panthera leo,9689,Felidae,leaf,field team A",
                "EX-001,Panthera leo,9689,Felidae,root,field team A",
            ),
        )
        upsert_metadata_csv(session, changed_csv)
        session.commit()

        extracts = session.scalars(select(Extract)).all()
        assert len(extracts) == 2
        ex001 = session.scalar(select(Extract).where(Extract.sample_code == "EX-001"))
        assert ex001 is not None
        assert ex001.organ == "root"


def test_row_missing_sample_code_raises_a_clear_error(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        "sample_code,taxon_name\n,Panthera leo\n",
    )
    engine = create_database(":memory:")
    with Session(engine) as session, pytest.raises(MetadataError, match="sample_code"):
        upsert_metadata_csv(session, csv_path)


def test_row_missing_taxon_name_raises_a_clear_error(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        "sample_code,taxon_name\nEX-001,\n",
    )
    engine = create_database(":memory:")
    with Session(engine) as session, pytest.raises(MetadataError, match="taxon_name"):
        upsert_metadata_csv(session, csv_path)


def test_csv_missing_a_required_column_raises_a_clear_error(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path, "sample_code\nEX-001\n")

    with pytest.raises(MetadataError, match="taxon_name"):
        load_metadata_csv(csv_path)


def test_missing_csv_file_raises_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(MetadataError, match="not found"):
        load_metadata_csv(tmp_path / "does-not-exist.csv")


def test_find_metadata_row_for_mzml_matches_by_filename_stem(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path)
    df = load_metadata_csv(csv_path)

    row = find_metadata_row_for_mzml(df, "EX-001.mzML")

    assert row["sample_code"] == "EX-001"
    assert row["taxon_name"] == "Panthera leo"


def test_find_metadata_row_for_mzml_matches_a_bare_sample_code(tmp_path: Path) -> None:
    csv_path = write_csv(tmp_path)
    df = load_metadata_csv(csv_path)

    row = find_metadata_row_for_mzml(df, "EX-002")

    assert row["sample_code"] == "EX-002"


def test_find_metadata_row_for_mzml_with_no_match_raises_a_clear_error(
    tmp_path: Path,
) -> None:
    csv_path = write_csv(tmp_path)
    df = load_metadata_csv(csv_path)

    with pytest.raises(MetadataError, match="EX-999"):
        find_metadata_row_for_mzml(df, "EX-999.mzML")


def test_find_metadata_row_for_mzml_with_duplicate_sample_codes_raises_a_clear_error(
    tmp_path: Path,
) -> None:
    csv_path = write_csv(
        tmp_path,
        "sample_code,taxon_name\nEX-001,Panthera leo\nEX-001,Bos taurus\n",
    )
    df = load_metadata_csv(csv_path)

    with pytest.raises(MetadataError, match="2 rows"):
        find_metadata_row_for_mzml(df, "EX-001.mzML")


def test_upsert_metadata_row_uses_find_result_to_upsert_a_single_row(
    tmp_path: Path,
) -> None:
    csv_path = write_csv(tmp_path)
    df = load_metadata_csv(csv_path)
    row = find_metadata_row_for_mzml(df, "EX-001.mzML")

    engine = create_database(":memory:")
    with Session(engine) as session:
        extract = upsert_metadata_row(session, row, csv_path=csv_path)
        session.commit()

        assert extract.sample_code == "EX-001"
        assert len(session.scalars(select(Extract)).all()) == 1


def test_row_with_non_integer_ncbi_taxid_raises_a_clear_error(tmp_path: Path) -> None:
    csv_path = write_csv(
        tmp_path,
        "sample_code,taxon_name,ncbi_taxid\nEX-001,Panthera leo,not-a-number\n",
    )
    engine = create_database(":memory:")
    with Session(engine) as session, pytest.raises(MetadataError, match="ncbi_taxid"):
        upsert_metadata_csv(session, csv_path)

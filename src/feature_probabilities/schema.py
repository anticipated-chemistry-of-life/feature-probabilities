"""The single-source-of-truth DuckDB schema: SQLAlchemy models for all eight tables.

Every table the pipeline writes to is defined exactly once, here, as a
SQLAlchemy ORM model. There is no hand-maintained SQL DDL file to drift out of
sync — :func:`create_database` derives a fresh (or already-initialized)
DuckDB file's tables directly from these models via the ``duckdb-engine``
dialect.

Table shape and FK-respecting insert order (species must exist before
extracts, extracts before sirius_runs, sirius_runs before features, molecules
before annotations, features/annotations before calibration_scores):

``species`` -> ``extracts`` -> ``sirius_runs`` -> ``features``/``molecules``
-> ``annotations`` -> ``kde_models`` -> ``calibration_scores``

See the DuckDB schema decision (issue #4) and its checksum-column follow-up
(issue #5) for the full column-level derivation, and ``CONTEXT.md`` for the
domain vocabulary (Feature, Structure candidate, Molecule, Annotation,
Extract, SIRIUS run, Calibration score) these tables encode.

Primary keys use an explicit :class:`sqlalchemy.Sequence` rather than
``autoincrement=True``/``Identity()``: duckdb-engine's Postgres-flavored DDL
compiler emits ``SERIAL`` (a type DuckDB doesn't have) for the former and a
``GENERATED ... AS IDENTITY`` clause DuckDB rejects when combined with a
``PRIMARY KEY`` constraint for the latter; an explicit sequence is the
combination that actually works against DuckDB.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

#: Closed set of `sirius_runs.source_kind` values, per the DuckDB schema
#: decision: a ground-truth run (no `extract_id`) or a field run (has one).
SOURCE_KIND_FIELD_MZML = "field_mzml"
SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM = "ground_truth_massspecgym"

#: First-block (14-character, skeleton-level) InChIKey length, as returned
#: unmodified by both SIRIUS's `StructureCandidateFormula.inchiKey` and
#: MassSpecGym's `inchikey` column -- see CONTEXT.md's Molecule entry.
INCHIKEY_FIRST_BLOCK_LENGTH = 14


class Base(DeclarativeBase):
    """Declarative base shared by every feature-probabilities table."""


class Species(Base):
    """A taxon a field-sample Extract was collected from."""

    __tablename__ = "species"

    species_id: Mapped[int] = mapped_column(
        sa.Integer, sa.Sequence("species_id_seq"), primary_key=True
    )
    taxon_name: Mapped[str] = mapped_column(sa.String, unique=True)
    ncbi_taxid: Mapped[int | None] = mapped_column(sa.Integer)
    family: Mapped[str | None] = mapped_column(sa.String)


class Extract(Base):
    """The physical sample (species plus e.g. organ) a SIRIUS run is acquired from."""

    __tablename__ = "extracts"

    extract_id: Mapped[int] = mapped_column(
        sa.Integer, sa.Sequence("extract_id_seq"), primary_key=True
    )
    sample_code: Mapped[str] = mapped_column(sa.String, unique=True)
    species_id: Mapped[int] = mapped_column(sa.ForeignKey("species.species_id"))
    organ: Mapped[str | None] = mapped_column(sa.String)
    #: Whatever additional columns a real metadata CSV turns out to carry,
    #: upserted alongside the pinned columns above rather than requiring a
    #: schema migration per new metadata field.
    extra_metadata: Mapped[dict[str, object] | None] = mapped_column(sa.JSON)


class SiriusRun(Base):
    """One SIRIUS execution over one input file with one fixed parameter set.

    A field run has ``extract_id`` set and ``source_kind ==
    SOURCE_KIND_FIELD_MZML``; a ground-truth run has ``extract_id`` NULL and
    ``source_kind == SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM``. Reprocessing the
    same input under different parameters inserts a new, coexisting row --
    this table is never updated in place.
    """

    __tablename__ = "sirius_runs"
    __table_args__ = (
        sa.CheckConstraint(
            f"source_kind IN ('{SOURCE_KIND_FIELD_MZML}', "
            f"'{SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM}')",
            name="ck_sirius_runs_source_kind",
        ),
    )

    run_id: Mapped[int] = mapped_column(
        sa.Integer, sa.Sequence("sirius_runs_run_id_seq"), primary_key=True
    )
    extract_id: Mapped[int | None] = mapped_column(sa.ForeignKey("extracts.extract_id"))
    source_kind: Mapped[str] = mapped_column(sa.String)
    input_file_checksum: Mapped[str] = mapped_column(sa.String)
    import_params_checksum: Mapped[str | None] = mapped_column(sa.String)
    analysis_params_checksum: Mapped[str] = mapped_column(sa.String)
    sirius_version: Mapped[str] = mapped_column(sa.String)
    pysirius_client_version: Mapped[str] = mapped_column(sa.String)
    ionization_mode: Mapped[str] = mapped_column(sa.String)
    instrument_type: Mapped[str] = mapped_column(sa.String)
    input_file_path: Mapped[str] = mapped_column(sa.String)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )


class Feature(Base):
    """One MS2 spectrum treated as an analytical unit by SIRIUS.

    ``true_inchikey`` is populated only for ground-truth-run features (from
    MassSpecGym's known structure); it is NULL for field-run features, where
    the true identity is unknown.
    """

    __tablename__ = "features"

    feature_id: Mapped[int] = mapped_column(
        sa.Integer, sa.Sequence("features_feature_id_seq"), primary_key=True
    )
    run_id: Mapped[int] = mapped_column(sa.ForeignKey("sirius_runs.run_id"))
    external_feature_id: Mapped[str] = mapped_column(sa.String)
    ion_mass: Mapped[float] = mapped_column(sa.Double)
    charge: Mapped[int] = mapped_column(sa.Integer)
    rt_start_seconds: Mapped[float | None] = mapped_column(sa.Double)
    rt_end_seconds: Mapped[float | None] = mapped_column(sa.Double)
    rt_apex_seconds: Mapped[float | None] = mapped_column(sa.Double)
    quality: Mapped[str | None] = mapped_column(sa.String)
    true_inchikey: Mapped[str | None] = mapped_column(
        sa.String(INCHIKEY_FIRST_BLOCK_LENGTH)
    )


class Molecule(Base):
    """The globally deduplicated, skeleton-level chemical identity of a structure candidate.

    ``inchikey`` is the first-block (14-character) key exactly as SIRIUS's
    ``StructureCandidateFormula.inchiKey`` returns it -- no truncation step,
    since SIRIUS never returns a longer key. ``smiles`` is RDKit-canonicalized
    with stereochemistry stripped before insert.
    """

    __tablename__ = "molecules"

    molecule_id: Mapped[int] = mapped_column(
        sa.Integer, sa.Sequence("molecules_molecule_id_seq"), primary_key=True
    )
    inchikey: Mapped[str] = mapped_column(
        sa.String(INCHIKEY_FIRST_BLOCK_LENGTH), unique=True
    )
    smiles: Mapped[str] = mapped_column(sa.String)
    molecular_formula: Mapped[str] = mapped_column(sa.String)


class Annotation(Base):
    """One feature-structure-candidate pair: a candidate's score joined to its feature."""

    __tablename__ = "annotations"
    __table_args__ = (
        sa.UniqueConstraint(
            "feature_id", "molecule_id", name="uq_annotations_feature_molecule"
        ),
    )

    annotation_id: Mapped[int] = mapped_column(
        sa.Integer, sa.Sequence("annotations_annotation_id_seq"), primary_key=True
    )
    feature_id: Mapped[int] = mapped_column(sa.ForeignKey("features.feature_id"))
    molecule_id: Mapped[int] = mapped_column(sa.ForeignKey("molecules.molecule_id"))
    rank: Mapped[int] = mapped_column(sa.Integer)
    csi_score: Mapped[float] = mapped_column(sa.Double)
    tanimoto_similarity: Mapped[float] = mapped_column(sa.Double)
    mces_dist_to_top_hit: Mapped[float] = mapped_column(sa.Double)
    xlogp: Mapped[float] = mapped_column(sa.Double)
    adduct: Mapped[str] = mapped_column(sa.String)
    formula_id: Mapped[str] = mapped_column(sa.String)


class KdeModel(Base):
    """One `fp-fit-kde` run's pickled calibration model: a dict keyed by stratum name."""

    __tablename__ = "kde_models"

    kde_model_id: Mapped[int] = mapped_column(
        sa.Integer, sa.Sequence("kde_models_kde_model_id_seq"), primary_key=True
    )
    fitted_at: Mapped[datetime] = mapped_column(
        sa.DateTime, server_default=sa.func.now()
    )
    artifact_path: Mapped[str] = mapped_column(sa.String)


class CalibrationScore(Base):
    """One KDE model version's score for one annotation.

    Many rows per annotation -- one per KDE model version ever applied to it
    -- so refitting the calibration never erases the ability to compare old
    and new scores.
    """

    __tablename__ = "calibration_scores"

    annotation_id: Mapped[int] = mapped_column(
        sa.ForeignKey("annotations.annotation_id"), primary_key=True
    )
    kde_model_id: Mapped[int] = mapped_column(
        sa.ForeignKey("kde_models.kde_model_id"), primary_key=True
    )
    score: Mapped[float] = mapped_column(sa.Double)


def create_database(db_path: str | Path) -> Engine:
    """Open (creating if needed) the DuckDB file at ``db_path`` with all eight tables.

    Idempotent and non-destructive: safe to call on every CLI startup to
    guarantee the schema exists before writing. Existing tables and their
    data are left untouched; only missing tables are created. Pass the
    literal string ``":memory:"`` for an in-memory database (used in tests).

    Returns the :class:`~sqlalchemy.Engine` bound to the database, ready for
    callers to open sessions/connections against.
    """
    if str(db_path) == ":memory:":
        engine = sa.create_engine("duckdb:///:memory:")
    else:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = sa.create_engine(f"duckdb:///{path}")
    Base.metadata.create_all(engine)
    return engine

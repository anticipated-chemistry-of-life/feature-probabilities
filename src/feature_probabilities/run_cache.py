"""SIRIUS run cache lookup and persistence: the core idempotency guarantee.

Ties the `Sirius` wrapper (`sirius.py`), the checksum functions
(`checksums.py`), and the DuckDB schema (`schema.py`) together into the
actual rerun-avoidance behavior both `generate-groundtruth` and
`annotate` are built on:

- Before any SIRIUS call, compute `input_file_checksum`,
  `import_params_checksum` (nullable), and `analysis_params_checksum`, then
  query `sirius_runs` for a matching row (`input_file_checksum` = ?,
  `import_params_checksum` IS NOT DISTINCT FROM ?, `analysis_params_checksum`
  = ?, `sirius_version` = ?).
- On a hit, return the existing run's `features`/`annotations` without
  calling into the `Sirius` wrapper's `import_spectra`/`run` at all.
- On a miss, call the wrapper to import, restrict the analysis job to the
  imported features passing the request's `annotated_qualities` filter (if
  any), and run it, then persist a new `sirius_runs` row plus those
  `features`, the annotations of each one's `top_k` best-ranked structure
  candidates (CONTEXT.md's Top-k), and (deduplicated globally by `inchikey`)
  `molecules` rows. The quality filter and `top_k` are both part of
  `analysis_params_checksum`, so runs differing in either never share a
  cache identity.
- `force=True` skips the cache lookup entirely and always calls the wrapper,
  regardless of an existing matching row -- this table is never updated in
  place, so this always inserts a new, coexisting `sirius_runs` row rather
  than overwriting the old one, matching every other cache-miss path.

Each `molecules` row's `inchikey` is stored exactly as
`StructureCandidateFormula.inchi_key` returns it (already first-block/14-
character -- no truncation step), and `smiles` is RDKit-canonicalized with
stereochemistry stripped before insert -- or stored verbatim when RDKit
can't parse it (SIRIUS's candidate databases contain valence-violating
structures, e.g. a divalent `[Cl-]`), so one bad candidate never fails a run.

Callers control the transaction: this module `flush`es (so newly-inserted
rows get their autoincrement ids) but never `commit`s, matching
`metadata.py`'s convention.

A run's `molecules` and `annotations` are written in bulk (issue #27): one
set-based lookup of the run's already-stored molecules, then Arrow-backed
`INSERT`s with ids drawn from the tables' own sequences, all on the
session's DuckDB connection and so inside the caller's transaction. A
per-candidate lookup cannot be used: DuckDB's unique index on `inchikey`
only covers committed rows, so each lookup also scans every molecule
inserted earlier in the same transaction, which made persisting one chunk
take time quadratic in its structure candidates.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import pyarrow as pa
from rdkit import Chem
from sqlalchemy import func, select

from feature_probabilities.checksums import (
    analysis_params_checksum,
    import_params_checksum,
    input_file_checksum,
)
from feature_probabilities.schema import (
    ANNOTATION_ID_SEQUENCE,
    MOLECULE_ID_SEQUENCE,
    Annotation,
    Feature,
    Molecule,
    SiriusRun,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    from PySirius import (
        DataQuality,
        JobSubmission,
        LcmsSubmissionParameters,
        StructureCandidateFormula,
    )
    from sqlalchemy import Sequence as IdSequence
    from sqlalchemy.orm import Session

    from feature_probabilities.sirius import (
        FeatureStructureCandidate,
        SiriusInterface,
    )


class _DuckDBResult(Protocol):
    def fetchall(self) -> list[tuple[object, ...]]: ...


class _DuckDBConnection(Protocol):
    """The slice of DuckDB's Python connection the bulk writes use.

    The session's DBAPI connection is `duckdb_engine`'s wrapper, which
    forwards these to the underlying `duckdb.DuckDBPyConnection` -- the
    same connection, and so the same transaction, the ORM writes through.
    """

    def register(self, view_name: str, python_object: object) -> object: ...

    def unregister(self, view_name: str) -> object: ...

    def execute(self, query: str) -> _DuckDBResult: ...


class RunCacheError(Exception):
    """Raised when a SIRIUS run request can't be built for its input."""


@dataclass(frozen=True, slots=True)
class SiriusRunRequest:
    """Everything needed to identify, and on a cache miss execute, one SIRIUS run.

    `sirius_version` is supplied by the caller (already validated against
    the config's required version by the CLI's fail-fast version check, per
    issue #9's CLI contract) rather than fetched here -- this module never
    calls `sirius.get_version()`.

    `annotated_qualities` restricts the analysis job, and the persisted
    `features`, to imported features whose `AlignedFeature.quality` is one
    of them; `None` analyses every imported feature. Only meaningful for
    peak-picked (mzML) imports: a pre-picked import carries no quality, so
    every one of its features is `NOT_APPLICABLE`.

    `top_k` is the Top-k (CONTEXT.md): only each feature's `top_k`
    best-ranked structure candidates become `annotations`.
    """

    input_file: Path
    project_path: Path
    source_kind: str
    extract_id: int | None
    import_params: LcmsSubmissionParameters | None
    analysis_params: JobSubmission
    sirius_version: str
    pysirius_client_version: str
    ionization_mode: str
    instrument_type: str
    top_k: int
    project_name: str | None = None
    annotated_qualities: frozenset[DataQuality] | None = None


@dataclass(frozen=True, slots=True)
class SiriusRunResult:
    """The outcome of `get_or_create_run`: one run plus its features/annotations."""

    run: SiriusRun
    features: list[Feature]
    annotations: list[Annotation]
    from_cache: bool


def _canonical_smiles(smiles: str) -> str:
    """RDKit-canonicalized, stereochemistry-stripped form of `smiles`.

    Returns `smiles` unchanged when RDKit can't parse it: SIRIUS's candidate
    databases contain chemically invalid structures (e.g. `C[Cl-]N`,
    `C=BrC`), and rejecting them would discard every other candidate in
    the run along with them.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True)


def _find_matching_run(
    session: Session,
    *,
    input_checksum: str,
    import_checksum: str | None,
    analysis_checksum: str,
    sirius_version: str,
) -> SiriusRun | None:
    stmt = (
        select(SiriusRun)
        .where(
            SiriusRun.input_file_checksum == input_checksum,
            SiriusRun.import_params_checksum.is_not_distinct_from(import_checksum),
            SiriusRun.analysis_params_checksum == analysis_checksum,
            SiriusRun.sirius_version == sirius_version,
        )
        .order_by(SiriusRun.run_id.desc())
    )
    return session.scalars(stmt).first()


def _existing_run_data(
    session: Session, run: SiriusRun
) -> tuple[list[Feature], list[Annotation]]:
    features = list(
        session.scalars(
            select(Feature)
            .where(Feature.run_id == run.run_id)
            .order_by(Feature.feature_id)
        )
    )
    return features, _annotations_of(session, features)


def _annotations_of(session: Session, features: list[Feature]) -> list[Annotation]:
    if not features:
        return []
    feature_ids = [feature.feature_id for feature in features]
    return list(
        session.scalars(
            select(Annotation)
            .where(Annotation.feature_id.in_(feature_ids))
            .order_by(Annotation.annotation_id)
        )
    )


def _persist_candidates(
    session: Session,
    candidate_rows: list[FeatureStructureCandidate],
    feature_by_aligned_id: dict[str | None, Feature],
) -> None:
    """Bulk-insert `candidate_rows` as `annotations`, adding any `molecules` not yet stored.

    Ids are allocated in candidate order -- a molecule's at its first
    occurrence -- exactly as the ORM's row-by-row inserts assigned them.
    """
    if not candidate_rows:
        return
    first_candidate_by_inchikey: dict[str | None, StructureCandidateFormula] = {}
    for row in candidate_rows:
        first_candidate_by_inchikey.setdefault(row.candidate.inchi_key, row.candidate)

    connection = _duckdb_connection(session)
    molecule_id_by_inchikey = _stored_molecule_ids(
        connection, list(first_candidate_by_inchikey)
    )
    new_molecule_candidates = [
        candidate
        for inchikey, candidate in first_candidate_by_inchikey.items()
        if inchikey not in molecule_id_by_inchikey
    ]
    new_molecule_ids = _next_ids(
        session, MOLECULE_ID_SEQUENCE, len(new_molecule_candidates)
    )
    _insert_arrow(
        connection,
        Molecule,
        _molecule_rows(new_molecule_ids, new_molecule_candidates),
    )
    molecule_id_by_inchikey.update(
        zip(
            (candidate.inchi_key for candidate in new_molecule_candidates),
            new_molecule_ids,
            strict=True,
        )
    )
    _insert_arrow(
        connection,
        Annotation,
        _annotation_rows(
            _next_ids(session, ANNOTATION_ID_SEQUENCE, len(candidate_rows)),
            candidate_rows,
            feature_id_by_aligned_id={
                aligned_id: feature.feature_id
                for aligned_id, feature in feature_by_aligned_id.items()
            },
            molecule_id_by_inchikey=molecule_id_by_inchikey,
        ),
    )


def _molecule_rows(
    molecule_ids: list[int], first_candidates: list[StructureCandidateFormula]
) -> pa.Table:
    """`molecules` rows for new molecules, each built from its first structure candidate."""
    return pa.table(
        {
            "molecule_id": pa.array(molecule_ids, pa.int32()),
            "inchikey": pa.array([c.inchi_key for c in first_candidates], pa.string()),
            "smiles": pa.array(
                [_canonical_smiles(c.smiles) for c in first_candidates], pa.string()
            ),
            "molecular_formula": pa.array(
                [c.molecular_formula for c in first_candidates], pa.string()
            ),
        }
    )


def _annotation_rows(
    annotation_ids: list[int],
    candidate_rows: list[FeatureStructureCandidate],
    *,
    feature_id_by_aligned_id: dict[str | None, int],
    molecule_id_by_inchikey: dict[str | None, int],
) -> pa.Table:
    """One `annotations` row per feature-structure-candidate pair, in candidate order."""
    candidates = [row.candidate for row in candidate_rows]
    return pa.table(
        {
            "annotation_id": pa.array(annotation_ids, pa.int32()),
            "feature_id": pa.array(
                [
                    feature_id_by_aligned_id[row.feature.aligned_feature_id]
                    for row in candidate_rows
                ],
                pa.int32(),
            ),
            "molecule_id": pa.array(
                [molecule_id_by_inchikey[c.inchi_key] for c in candidates], pa.int32()
            ),
            "rank": pa.array([c.rank for c in candidates], pa.int32()),
            "csi_score": pa.array([c.csi_score for c in candidates], pa.float64()),
            "tanimoto_similarity": pa.array(
                [c.tanimoto_similarity for c in candidates], pa.float64()
            ),
            "mces_dist_to_top_hit": pa.array(
                [c.mces_dist_to_top_hit for c in candidates], pa.float64()
            ),
            "xlogp": pa.array([c.xlog_p for c in candidates], pa.float64()),
            "adduct": pa.array([c.adduct for c in candidates], pa.string()),
            "formula_id": pa.array([c.formula_id for c in candidates], pa.string()),
        }
    )


def _duckdb_connection(session: Session) -> _DuckDBConnection:
    """The raw DuckDB connection behind `session`, inside its current transaction."""
    driver_connection = session.connection().connection.driver_connection
    if driver_connection is None:
        raise RuntimeError("The session's database connection is closed.")
    return driver_connection


@contextmanager
def _registered_view(
    connection: _DuckDBConnection, name: str, rows: pa.Table
) -> Generator[str]:
    """`rows` queryable as view `name` for the duration of the block."""
    connection.register(name, rows)
    try:
        yield name
    finally:
        connection.unregister(name)


def _stored_molecule_ids(
    connection: _DuckDBConnection, inchikeys: list[str | None]
) -> dict[str | None, int]:
    """`molecule_id` of each of `inchikeys` already in `molecules`, in one set-based query."""
    keys = pa.table({"inchikey": pa.array(inchikeys, pa.string())})
    with _registered_view(connection, "_run_cache_inchikeys", keys) as view:
        rows = connection.execute(
            f"SELECT m.inchikey, m.molecule_id FROM {Molecule.__tablename__} m "
            f"JOIN {view} k ON m.inchikey = k.inchikey"
        ).fetchall()
    # The SELECT above fixes each row's shape: (VARCHAR inchikey, INTEGER id).
    return dict(cast("list[tuple[str, int]]", rows))


def _next_ids(session: Session, sequence: IdSequence, count: int) -> list[int]:
    """`count` fresh values of `sequence`, ascending."""
    if count == 0:
        return []
    return sorted(
        session.scalars(select(sequence.next_value()).select_from(func.range(count)))
    )


def _insert_arrow(
    connection: _DuckDBConnection, model: type[Molecule | Annotation], rows: pa.Table
) -> None:
    """Append `rows` to `model`'s table, matching columns by name."""
    if rows.num_rows == 0:
        return
    table = model.__tablename__
    with _registered_view(connection, f"_run_cache_{table}", rows) as view:
        connection.execute(f"INSERT INTO {table} BY NAME SELECT * FROM {view}")


def _run_and_persist(
    session: Session,
    sirius: SiriusInterface,
    request: SiriusRunRequest,
    *,
    input_checksum: str,
    import_checksum: str | None,
    analysis_checksum: str,
) -> SiriusRunResult:
    try:
        sirius.create_project(request.project_path, request.project_name)
        sirius.import_spectra(request.input_file, params=request.import_params)

        aligned_features = sirius.get_features()
        job_submission = request.analysis_params
        if request.annotated_qualities is not None:
            aligned_features = [
                aligned_feature
                for aligned_feature in aligned_features
                if aligned_feature.quality in request.annotated_qualities
            ]
            # Without explicit ids SIRIUS analyses every feature in the
            # project, the excluded ones included.
            job_submission = job_submission.model_copy(
                update={
                    "aligned_feature_ids": [
                        aligned_feature.aligned_feature_id
                        for aligned_feature in aligned_features
                    ]
                }
            )
        # An empty id list is no restriction to SIRIUS: with nothing passing
        # the filter, there is nothing to run.
        if aligned_features:
            sirius.run(job_submission)
        candidate_rows = sirius.get_structure_candidates(
            aligned_features, top_k=request.top_k
        )
    finally:
        # A project left open stays registered in the SIRIUS instance for its
        # whole lifetime, holding its database open even once the temporary
        # directory is gone -- one leak per chunk otherwise. Closed here, on
        # the failure path too, since a failed chunk's project is just as
        # stale.
        sirius.close_project()

    run = SiriusRun(
        extract_id=request.extract_id,
        source_kind=request.source_kind,
        input_file_checksum=input_checksum,
        import_params_checksum=import_checksum,
        analysis_params_checksum=analysis_checksum,
        sirius_version=request.sirius_version,
        pysirius_client_version=request.pysirius_client_version,
        ionization_mode=request.ionization_mode,
        instrument_type=request.instrument_type,
        input_file_path=str(request.input_file),
    )
    session.add(run)
    session.flush()

    feature_by_aligned_id: dict[str | None, Feature] = {}
    for aligned_feature in aligned_features:
        feature = Feature(
            run_id=run.run_id,
            external_feature_id=aligned_feature.external_feature_id,
            ion_mass=aligned_feature.ion_mass,
            charge=aligned_feature.charge,
            rt_start_seconds=aligned_feature.rt_start_seconds,
            rt_end_seconds=aligned_feature.rt_end_seconds,
            rt_apex_seconds=aligned_feature.rt_apex_seconds,
            quality=aligned_feature.quality.value if aligned_feature.quality else None,
        )
        session.add(feature)
        feature_by_aligned_id[aligned_feature.aligned_feature_id] = feature
    session.flush()

    _persist_candidates(session, candidate_rows, feature_by_aligned_id)
    features = list(feature_by_aligned_id.values())

    return SiriusRunResult(
        run=run,
        features=features,
        annotations=_annotations_of(session, features),
        from_cache=False,
    )


def get_or_create_run(
    session: Session,
    sirius: SiriusInterface,
    request: SiriusRunRequest,
    *,
    force: bool = False,
) -> SiriusRunResult:
    """Reuse a matching `sirius_runs` row, or run SIRIUS and persist a fresh one.

    On a cache hit (an existing row matches on `input_file_checksum`,
    `import_params_checksum`, `analysis_params_checksum`, and
    `sirius_version`) and `force` is not set, returns that run's existing
    `features`/`annotations` without calling `sirius.import_spectra`/`run`
    at all. Otherwise runs SIRIUS via `sirius` and persists a new,
    coexisting `sirius_runs` row plus its `features` (only those passing
    `request.annotated_qualities`, if set), `annotations`, and
    (deduplicated globally by `inchikey`) `molecules` rows.
    """
    input_checksum = input_file_checksum(request.input_file.read_bytes())
    import_checksum = import_params_checksum(request.import_params)
    analysis_checksum = analysis_params_checksum(
        request.analysis_params,
        None
        if request.annotated_qualities is None
        else [quality.value for quality in request.annotated_qualities],
        top_k=request.top_k,
    )

    if not force:
        existing = _find_matching_run(
            session,
            input_checksum=input_checksum,
            import_checksum=import_checksum,
            analysis_checksum=analysis_checksum,
            sirius_version=request.sirius_version,
        )
        if existing is not None:
            features, annotations = _existing_run_data(session, existing)
            return SiriusRunResult(
                run=existing,
                features=features,
                annotations=annotations,
                from_cache=True,
            )

    return _run_and_persist(
        session,
        sirius,
        request,
        input_checksum=input_checksum,
        import_checksum=import_checksum,
        analysis_checksum=analysis_checksum,
    )

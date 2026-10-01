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
  `features`, their `annotations`, and (deduplicated globally by `inchikey`)
  `molecules` rows. The quality filter is part of
  `analysis_params_checksum`, so a filtered and an unfiltered run never
  share a cache identity.
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
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rdkit import Chem
from sqlalchemy import select

from feature_probabilities.checksums import (
    analysis_params_checksum,
    import_params_checksum,
    input_file_checksum,
)
from feature_probabilities.schema import Annotation, Feature, Molecule, SiriusRun

if TYPE_CHECKING:
    from PySirius import DataQuality, JobSubmission, LcmsSubmissionParameters
    from sqlalchemy.orm import Session

    from feature_probabilities.sirius import SiriusInterface


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
    if not features:
        return features, []
    feature_ids = [feature.feature_id for feature in features]
    annotations = list(
        session.scalars(
            select(Annotation)
            .where(Annotation.feature_id.in_(feature_ids))
            .order_by(Annotation.annotation_id)
        )
    )
    return features, annotations


def _get_or_create_molecule(
    session: Session,
    molecule_cache: dict[str, Molecule],
    *,
    inchikey: str,
    smiles: str,
    molecular_formula: str,
) -> Molecule:
    cached = molecule_cache.get(inchikey)
    if cached is not None:
        return cached
    existing = session.scalars(
        select(Molecule).where(Molecule.inchikey == inchikey)
    ).one_or_none()
    if existing is not None:
        molecule_cache[inchikey] = existing
        return existing
    created = Molecule(
        inchikey=inchikey,
        smiles=_canonical_smiles(smiles),
        molecular_formula=molecular_formula,
    )
    session.add(created)
    session.flush()
    molecule_cache[inchikey] = created
    return created


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
        candidate_rows = sirius.get_structure_candidates(aligned_features)
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

    feature_by_aligned_id: dict[str, Feature] = {}
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

    molecule_cache: dict[str, Molecule] = {}
    annotations: list[Annotation] = []
    for row in candidate_rows:
        feature = feature_by_aligned_id[row.feature.aligned_feature_id]
        candidate = row.candidate
        molecule = _get_or_create_molecule(
            session,
            molecule_cache,
            inchikey=candidate.inchi_key,
            smiles=candidate.smiles,
            molecular_formula=candidate.molecular_formula,
        )
        annotation = Annotation(
            feature_id=feature.feature_id,
            molecule_id=molecule.molecule_id,
            rank=candidate.rank,
            csi_score=candidate.csi_score,
            tanimoto_similarity=candidate.tanimoto_similarity,
            mces_dist_to_top_hit=candidate.mces_dist_to_top_hit,
            xlogp=candidate.xlog_p,
            adduct=candidate.adduct,
            formula_id=candidate.formula_id,
        )
        session.add(annotation)
        annotations.append(annotation)
    session.flush()

    return SiriusRunResult(
        run=run,
        features=list(feature_by_aligned_id.values()),
        annotations=annotations,
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

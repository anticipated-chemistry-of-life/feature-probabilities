"""Pure data transform: raw ground-truth rows -> deduplicated correct assignments.

Turns the ground-truth-run `features`/`annotations`/`molecules` rows #19
persists into the deduplicated set of correct-assignment rows `fp-fit-kde`
(#22) fits its KDE on. Deliberately does **not** read the DuckDB schema, run
`scipy.stats.gaussian_kde`, or stratify by instrument type -- it only
answers two questions, per the KDE calibration method decision (issue #7,
amended by #9's InChIKey/SMILES correction):

- :func:`is_correct_assignment` -- does a candidate's InChIKey match the
  ground truth's known structure? Both sides are already first-block
  (14-character, skeleton-level) InChIKeys as returned by their sources
  (SIRIUS's `StructureCandidateFormula.inchiKey`, MassSpecGym's `inchikey`
  column), so this is a plain equality with no truncation step.
- :func:`correct_assignment_rows` -- filters to correct assignments, then
  deduplicates by `(true_inchikey, adduct, instrument_type)`, sorted by
  `feature_id` ascending first and keeping the first row per group, so that
  frequently-replicated molecules in MassSpecGym don't inflate the KDE's
  density estimate and refitting on unchanged data is byte-for-byte
  reproducible.

Callers (#22) are responsible for joining `features`/`annotations`/
`molecules`/`sirius_runs` into :class:`GroundTruthAnnotationRow` and for any
later stratification by `instrument_type`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True, slots=True)
class GroundTruthAnnotationRow:
    """One ground-truth-run feature-structure-candidate pair, flattened and joined.

    The row shape :func:`correct_assignment_rows` consumes: `feature_id` and
    `true_inchikey` come from a ground-truth `features` row, `candidate_inchikey`
    from the `annotations` row's joined `molecules.inchikey`, `adduct` from
    the `annotations` row, `instrument_type` from the owning `sirius_runs`
    row, and `ion_mass`/`csi_score` from `features`/`annotations`
    respectively -- the pair the KDE fit consumes for a kept row.
    """

    feature_id: int
    candidate_inchikey: str
    true_inchikey: str
    adduct: str
    instrument_type: str
    ion_mass: float
    csi_score: float


@dataclass(frozen=True, slots=True)
class CorrectAssignmentRow:
    """One deduplicated correct-assignment row the KDE fit consumes."""

    feature_id: int
    true_inchikey: str
    adduct: str
    instrument_type: str
    ion_mass: float
    csi_score: float


def is_correct_assignment(row: GroundTruthAnnotationRow) -> bool:
    """Whether `row`'s candidate InChIKey matches its ground truth's `true_inchikey`.

    Plain equality: both `candidate_inchikey` and `true_inchikey` are already
    first-block (14-character) InChIKeys as returned by SIRIUS and
    MassSpecGym respectively, so no truncation/partial/prefix matching is
    performed here.
    """
    return row.candidate_inchikey == row.true_inchikey


def _to_correct_assignment_row(row: GroundTruthAnnotationRow) -> CorrectAssignmentRow:
    return CorrectAssignmentRow(
        feature_id=row.feature_id,
        true_inchikey=row.true_inchikey,
        adduct=row.adduct,
        instrument_type=row.instrument_type,
        ion_mass=row.ion_mass,
        csi_score=row.csi_score,
    )


def correct_assignment_rows(
    rows: Iterable[GroundTruthAnnotationRow],
) -> list[CorrectAssignmentRow]:
    """Filter `rows` to correct assignments, then deduplicate deterministically.

    Deduplicates by `(true_inchikey, adduct, instrument_type)`: rows are
    sorted by `feature_id` ascending first, and the first row per group is
    kept. The result is returned in that same `feature_id`-ascending order,
    so re-running this on the same underlying data always produces
    byte-identical output regardless of `rows`' input order.
    """
    correct = sorted(
        (row for row in rows if is_correct_assignment(row)),
        key=lambda row: row.feature_id,
    )
    kept_by_key: dict[tuple[str, str, str], GroundTruthAnnotationRow] = {}
    for row in correct:
        key = (row.true_inchikey, row.adduct, row.instrument_type)
        if key not in kept_by_key:
            kept_by_key[key] = row
    return [_to_correct_assignment_row(row) for row in kept_by_key.values()]

"""Applies a fitted KDE pickle to a batch of already-annotated features.

Issue #24's slice: given a loaded stratified-KDE dict (`fit-kde`'s #22 pickle
shape, `kde.fit_stratified_kdes`'s output) and an explicit set of
`annotations` already in the DB, computes each annotation's calibration
score -- `kde[stratum].pdf((ion_mass, csi_score))`, per the KDE calibration
method decision (issue #7) -- and upserts one `calibration_scores` row per
`(annotation_id, kde_model_id)`.

Stratum selection: `feature.instrument_type` (via the owning `sirius_runs`
row) if it has a dedicated trained stratum in the pickle, else
`kde.GLOBAL_STRATUM` (`"__global__"`) -- the same fallback semantics
`fit-kde` documents, so a feature from an instrument type neither trained
stratum covers is scored, never errors.

Deliberately does **not** touch the `Sirius` wrapper, and never writes
`sirius_runs`/`features`/`annotations` -- pure DB read/compute/write against
`calibration_scores` only, as issue #24 requires. The caller always states
which annotations to score (there is no "every annotation in the DB"
default): the whole DB includes ground-truth MassSpecGym annotations, which
are never meant to receive a calibration score. Selecting *which*
pickle/`kde_model_id` to apply (`--kde-model`, defaulting to the most
recently fitted `kde_models` row) is `annotate`'s remaining CLI surface
(issue #25), which also relies on this module's upsert semantics to stay
idempotent when reprocessing an unchanged, cache-hit batch.

Trust boundary: :func:`load_kdes` unpickles a `fit-kde`-produced artifact.
Per `cli_fit_kde._pickle_kdes`'s same documented trust boundary, this never
crosses a trust boundary -- it's a same-deployment, operator-controlled
filesystem path (`kde_models.artifact_path`), never untrusted/network input.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.kde import GLOBAL_STRATUM
from feature_probabilities.schema import (
    Annotation,
    CalibrationScore,
    Feature,
    SiriusRun,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scipy.stats import gaussian_kde


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """A basic summary of what one `apply_calibration` call did."""

    annotations_scored: int
    kde_model_id: int


@dataclass(frozen=True, slots=True)
class AnnotationScoringTarget:
    """One annotation to score: its id plus the `(ion_mass, csi_score)` pair.

    `instrument_type` comes from the owning `sirius_runs` row and drives
    stratum selection (`_select_stratum`).
    """

    annotation_id: int
    ion_mass: float
    csi_score: float
    instrument_type: str


def load_kdes(artifact_path: str | Path) -> dict[str, gaussian_kde]:
    """Unpickle a `fit-kde` artifact back into its stratum-keyed KDE dict.

    Trust boundary: `artifact_path` is always a `kde_models.artifact_path`
    written by this same deployment's `fit-kde` (never untrusted/network
    input) -- see this module's docstring.
    """
    with Path(artifact_path).open("rb") as handle:
        return pickle.load(handle)


def _select_stratum(instrument_type: str, kdes: dict[str, gaussian_kde]) -> str:
    """`instrument_type` if it has a dedicated trained stratum in `kdes`, else the fallback."""
    return instrument_type if instrument_type in kdes else GLOBAL_STRATUM


def _annotation_targets(
    session: Session, annotation_ids: Sequence[int]
) -> list[AnnotationScoringTarget]:
    """Every `annotation_ids` row to score, joined with its feature's ion mass and instrument type.

    Joins `annotations` -> `features` -> `sirius_runs` to pull in the
    feature's ion mass and owning run's instrument type.
    """
    stmt = (
        select(
            Annotation.annotation_id,
            Feature.ion_mass,
            Annotation.csi_score,
            SiriusRun.instrument_type,
        )
        .join(Feature, Annotation.feature_id == Feature.feature_id)
        .join(SiriusRun, Feature.run_id == SiriusRun.run_id)
        .where(Annotation.annotation_id.in_(annotation_ids))
    )
    return [
        AnnotationScoringTarget(
            annotation_id=annotation_id,
            ion_mass=ion_mass,
            csi_score=csi_score,
            instrument_type=instrument_type,
        )
        for annotation_id, ion_mass, csi_score, instrument_type in session.execute(
            stmt
        ).all()
    ]


def apply_calibration(
    session: Session,
    kdes: dict[str, gaussian_kde],
    kde_model_id: int,
    *,
    annotation_ids: Sequence[int],
) -> CalibrationSummary:
    """Score every `annotation_ids` row against `kdes` and persist+commit the result.

    For each targeted annotation, selects its stratum (`_select_stratum`)
    and computes `kdes[stratum].pdf((ion_mass, csi_score))`, then upserts
    one `calibration_scores` row keyed `(annotation_id, kde_model_id)`: a
    fresh pair is inserted, an already-scored pair has its `score` updated
    in place.

    Running this a second time with a *different* `kde_model_id` for the
    same annotations inserts a second, coexisting set of rows rather than
    overwriting the first, per the schema's `calibration_scores` versioning
    design (composite `(annotation_id, kde_model_id)` primary key).
    Re-running with the *same* `kde_model_id` (e.g. `annotate` reprocessing
    an unchanged, cache-hit batch) is idempotent -- it recomputes and
    overwrites those rows' scores rather than raising a primary-key
    conflict.

    Raises:
        sqlalchemy.exc.IntegrityError: `kde_model_id` doesn't reference an
            existing `kde_models` row.
    """
    targets = _annotation_targets(session, annotation_ids)
    for target in targets:
        stratum = _select_stratum(target.instrument_type, kdes)
        score = float(kdes[stratum].pdf((target.ion_mass, target.csi_score))[0])
        existing = session.get(CalibrationScore, (target.annotation_id, kde_model_id))
        if existing is not None:
            existing.score = score
        else:
            session.add(
                CalibrationScore(
                    annotation_id=target.annotation_id,
                    kde_model_id=kde_model_id,
                    score=score,
                )
            )
    session.commit()

    return CalibrationSummary(
        annotations_scored=len(targets),
        kde_model_id=kde_model_id,
    )

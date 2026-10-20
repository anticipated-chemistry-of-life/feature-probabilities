"""Applies a fitted KDE pickle to a batch of already-annotated features.

Issue #24's slice: given a loaded per-Ionization-mode KDE dict (`fit-kde`'s
#22 pickle shape, `kde.fit_stratified_kdes`'s output) and an explicit set of
`annotations` already in the DB, computes each annotation's calibration
score -- `kdes[ionization_mode].pdf((ion_mass, csi_score))`, per the KDE
calibration method decision (issue #7, ADR 0002) -- and upserts one
`calibration_scores` row per `(annotation_id, kde_model_id)`.

KDE selection: the owning `sirius_runs` row's Ionization mode, with no
fallback. :func:`load_kdes` refuses an artifact lacking either mode (e.g. an
old instrument-keyed pickle), so every SIRIUS run's mode always has a KDE.

Deliberately does **not** touch the `Sirius` wrapper, and never writes
`sirius_runs`/`features`/`annotations` -- pure DB read/compute/write against
`calibration_scores` only, as issue #24 requires. The caller always states
which annotations to score (there is no "every annotation in the DB"
default): the whole DB includes ground-truth annotations, which are never
meant to receive a calibration score. Selecting *which*
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

from feature_probabilities.kde import IONIZATION_MODES
from feature_probabilities.schema import (
    Annotation,
    CalibrationScore,
    Feature,
    SiriusRun,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scipy.stats import gaussian_kde


class IncompleteKdeArtifactError(Exception):
    """Raised when a KDE artifact lacks a KDE for one of `kde.IONIZATION_MODES`."""


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """A basic summary of what one `apply_calibration` call did."""

    annotations_scored: int
    kde_model_id: int


@dataclass(frozen=True, slots=True)
class AnnotationScoringTarget:
    """One annotation to score: its id plus the `(ion_mass, csi_score)` pair.

    `ionization_mode` comes from the owning `sirius_runs` row and selects
    the KDE that scores it.
    """

    annotation_id: int
    ion_mass: float
    csi_score: float
    ionization_mode: str


def load_kdes(artifact_path: str | Path) -> dict[str, gaussian_kde]:
    """Unpickle a `fit-kde` artifact back into its Ionization-mode-keyed KDE dict.

    Trust boundary: `artifact_path` is always a `kde_models.artifact_path`
    written by this same deployment's `fit-kde` (never untrusted/network
    input) -- see this module's docstring.

    Raises:
        IncompleteKdeArtifactError: the artifact has no KDE for one of
            `kde.IONIZATION_MODES`, e.g. it predates per-mode calibration.
    """
    with Path(artifact_path).open("rb") as handle:
        kdes = pickle.load(handle)
    missing_modes = [mode for mode in IONIZATION_MODES if mode not in kdes]
    if missing_modes:
        raise IncompleteKdeArtifactError(
            f"KDE artifact {str(artifact_path)!r} has no KDE for Ionization "
            f"mode(s) {', '.join(missing_modes)}."
        )
    return kdes


def _annotation_targets(
    session: Session, annotation_ids: Sequence[int]
) -> list[AnnotationScoringTarget]:
    """Every `annotation_ids` row to score, joined with its feature's ion mass and Ionization mode.

    Joins `annotations` -> `features` -> `sirius_runs` to pull in the
    feature's ion mass and owning run's Ionization mode.
    """
    stmt = (
        select(
            Annotation.annotation_id,
            Feature.ion_mass,
            Annotation.csi_score,
            SiriusRun.ionization_mode,
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
            ionization_mode=ionization_mode,
        )
        for annotation_id, ion_mass, csi_score, ionization_mode in session.execute(
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

    For each targeted annotation, computes
    `kdes[ionization_mode].pdf((ion_mass, csi_score))` with its SIRIUS run's
    Ionization mode -- there is no fallback KDE -- then upserts one
    `calibration_scores` row keyed `(annotation_id, kde_model_id)`: a fresh
    pair is inserted, an already-scored pair has its `score` updated in
    place.

    Running this a second time with a *different* `kde_model_id` for the
    same annotations inserts a second, coexisting set of rows rather than
    overwriting the first, per the schema's `calibration_scores` versioning
    design (composite `(annotation_id, kde_model_id)` primary key).
    Re-running with the *same* `kde_model_id` (e.g. `annotate` reprocessing
    an unchanged, cache-hit batch) is idempotent -- it recomputes and
    overwrites those rows' scores rather than raising a primary-key
    conflict.

    Raises:
        KeyError: `kdes` has no KDE for an annotation's Ionization mode.
        sqlalchemy.exc.IntegrityError: `kde_model_id` doesn't reference an
            existing `kde_models` row.
    """
    targets = _annotation_targets(session, annotation_ids)
    for target in targets:
        kde = kdes[target.ionization_mode]
        score = float(kde.pdf((target.ion_mass, target.csi_score))[0])
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

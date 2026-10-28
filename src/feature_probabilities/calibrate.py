"""Applies a fitted KDE pickle to a batch of already-annotated features.

Given a loaded per-Ionization-mode artifact (`fit-kde`'s pickle shape,
`kde.fit_stratified_kdes`'s output: a correct and an incorrect density per
mode) and an explicit set of `annotations` already in the DB, calibration
evaluates each annotation's natural-log correct density and natural-log
incorrect density -- `logpdf` of each density at `(ion_mass, csi_score)`
(ADR 0002, ADR 0004) -- and upserts one `calibration_densities` row per
`(annotation_id, kde_model_id)`.

Each Ionization mode's densities are evaluated over that mode's
annotations in chunks of at most `EVALUATION_CHUNK_SIZE`, one `logpdf` call
per density per chunk -- never one call per annotation, since every call
does work proportional to the density's training set before it evaluates a
single point.

Density selection: the owning `sirius_runs` row's Ionization mode, with no
fallback. :func:`load_kdes` refuses an artifact lacking either mode (e.g. an
old instrument-keyed pickle), so every SIRIUS run's mode always has both
densities.

Deliberately does **not** touch the `Sirius` wrapper, and never writes
`sirius_runs`/`features`/`annotations` -- pure DB read/compute/write against
`calibration_densities` only, as issue #24 requires. The caller always states
which annotations to calibrate (there is no "every annotation in the DB"
default): the whole DB includes ground-truth annotations, which are never
meant to receive calibration densities. Selecting *which*
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
from collections import defaultdict
from dataclasses import dataclass
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities.kde import IONIZATION_MODES, ModeDensities
from feature_probabilities.schema import (
    Annotation,
    CalibrationDensities,
    Feature,
    SiriusRun,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


#: Most annotations one density `logpdf` call evaluates. scipy whitens a KDE's whole
#: training set on every call, so a call per annotation repeats that work for
#: each of them; a chunk pays it once, while keeping the arrays a call holds
#: a bounded size however large the batch.
EVALUATION_CHUNK_SIZE = 10_000


class IncompleteKdeArtifactError(Exception):
    """Raised when a KDE artifact lacks a correct or incorrect density for an Ionization mode."""


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """A basic summary of what one `apply_calibration` call did."""

    annotations_calibrated: int
    kde_model_id: int


@dataclass(frozen=True, slots=True)
class CalibrationTarget:
    """One annotation to calibrate: its id plus the `(ion_mass, csi_score)` pair.

    `ionization_mode` comes from the owning `sirius_runs` row and selects
    the correct and incorrect density that are evaluated at the pair.
    """

    annotation_id: int
    ion_mass: float
    csi_score: float
    ionization_mode: str


def load_kdes(artifact_path: str | Path) -> dict[str, ModeDensities]:
    """Unpickle a `fit-kde` artifact back into its Ionization-mode-keyed densities.

    Trust boundary: `artifact_path` is always a `kde_models.artifact_path`
    written by this same deployment's `fit-kde` (never untrusted/network
    input) -- see this module's docstring.

    Raises:
        IncompleteKdeArtifactError: the artifact has no pair of correct and
            incorrect densities for one of `kde.IONIZATION_MODES`, e.g. it
            is an old single-KDE-per-mode artifact.
    """
    with Path(artifact_path).open("rb") as handle:
        artifact = pickle.load(handle)
    problems = [
        f"{mode} (no correct and incorrect density)"
        for mode in IONIZATION_MODES
        if not isinstance(artifact.get(mode), ModeDensities)
    ]
    if problems:
        raise IncompleteKdeArtifactError(
            f"KDE artifact {str(artifact_path)!r} is not a two-density artifact: "
            f"Ionization mode(s) {', '.join(problems)}."
        )
    return artifact


def _annotation_targets(
    session: Session, annotation_ids: Sequence[int]
) -> list[CalibrationTarget]:
    """Every `annotation_ids` row to calibrate, joined with its feature's ion mass and Ionization mode.

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
        CalibrationTarget(
            annotation_id=annotation_id,
            ion_mass=ion_mass,
            csi_score=csi_score,
            ionization_mode=ionization_mode,
        )
        for annotation_id, ion_mass, csi_score, ionization_mode in session.execute(
            stmt
        ).all()
    ]


def _evaluate_in_chunks(
    densities: dict[str, ModeDensities],
    targets: Sequence[CalibrationTarget],
    chunk_size: int,
) -> Iterator[tuple[int, float, float]]:
    """Yield `(annotation_id, log_correct_density, log_incorrect_density)` per target.

    Groups `targets` by Ionization mode, then evaluates that mode's correct
    and incorrect densities' `logpdf` over its group `chunk_size` points at
    a time. `logpdf` rather than `pdf`: far from the training data a density
    underflows to 0.0, whose log is -inf.

    Raises:
        KeyError: `densities` has none for a target's Ionization mode.
    """
    targets_by_mode: defaultdict[str, list[CalibrationTarget]] = defaultdict(list)
    for target in targets:
        targets_by_mode[target.ionization_mode].append(target)

    for ionization_mode, mode_targets in targets_by_mode.items():
        mode_densities = densities[ionization_mode]
        for chunk in batched(mode_targets, chunk_size):
            points = np.array(
                [
                    [target.ion_mass for target in chunk],
                    [target.csi_score for target in chunk],
                ]
            )
            for target, log_correct, log_incorrect in zip(
                chunk,
                mode_densities.correct.logpdf(points),
                mode_densities.incorrect.logpdf(points),
                strict=True,
            ):
                yield target.annotation_id, float(log_correct), float(log_incorrect)


def apply_calibration(
    session: Session,
    densities: dict[str, ModeDensities],
    kde_model_id: int,
    *,
    annotation_ids: Sequence[int],
    chunk_size: int = EVALUATION_CHUNK_SIZE,
) -> CalibrationSummary:
    """Evaluate both densities for every `annotation_ids` row and persist+commit them.

    Groups the targeted annotations by their SIRIUS run's Ionization mode --
    there is no fallback -- and evaluates that mode's correct and incorrect
    density `logpdf` over each group's `(ion_mass, csi_score)` points in one
    call per chunk of at most `chunk_size` annotations, never one call per
    annotation. `chunk_size` only bounds each call; it never changes a stored
    value. Then upserts one `calibration_densities` row keyed
    `(annotation_id, kde_model_id)` per annotation: a fresh pair is inserted,
    an already-calibrated pair has both log densities updated in place.

    Running this a second time with a *different* `kde_model_id` for the
    same annotations inserts a second, coexisting set of rows rather than
    overwriting the first, per the schema's `calibration_densities` versioning
    design (composite `(annotation_id, kde_model_id)` primary key).
    Re-running with the *same* `kde_model_id` (e.g. `annotate` reprocessing
    an unchanged, cache-hit batch) is idempotent -- it recomputes and
    overwrites those rows' values rather than raising a primary-key
    conflict.

    Raises:
        KeyError: `densities` has none for an annotation's Ionization mode.
        ValueError: `chunk_size` is below 1 and an annotation is targeted.
        sqlalchemy.exc.IntegrityError: `kde_model_id` doesn't reference an
            existing `kde_models` row.
    """
    targets = _annotation_targets(session, annotation_ids)
    for annotation_id, log_correct, log_incorrect in _evaluate_in_chunks(
        densities, targets, chunk_size
    ):
        existing = session.get(CalibrationDensities, (annotation_id, kde_model_id))
        if existing is not None:
            existing.log_correct_density = log_correct
            existing.log_incorrect_density = log_incorrect
        else:
            session.add(
                CalibrationDensities(
                    annotation_id=annotation_id,
                    kde_model_id=kde_model_id,
                    log_correct_density=log_correct,
                    log_incorrect_density=log_incorrect,
                )
            )
    session.commit()

    return CalibrationSummary(
        annotations_calibrated=len(targets),
        kde_model_id=kde_model_id,
    )

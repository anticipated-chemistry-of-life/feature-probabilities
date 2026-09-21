"""End-to-end `generate-groundtruth` run against a **real** SIRIUS instance.

Every other test in this suite talks to `sirius_fake.FakeSirius`, because
SIRIUS is a licensed desktop application that cannot run in CI. This module
is the one exception: it drives the production code path
(`cli_generate_groundtruth.main` -> `generate_groundtruth` ->
`run_cache.get_or_create_run` -> `sirius.Sirius`) against the SIRIUS that is
actually installed on the developer's machine, using a single real
MassSpecGym v1.5 spectrum, and is skipped entirely when SIRIUS or its
account credentials aren't available (`_sirius_available`).

It covers exactly what a fake cannot: that SIRIUS accepts the MGF chunk
`massspecgym.write_sirius_chunks` writes, creates a project-space at the
path `generate_groundtruth` hands it, accepts the `JobSubmission` built from
`[sirius.analysis_params]`, and returns features whose
`external_feature_id` still carries the MassSpecGym identifier the
ground-truth `true_inchikey` mapping is keyed on.

Two SIRIUS-6.3.3 facts this test pins down, both of which produced opaque
failures before it existed:

- A project path's parent directory must already exist, or SIRIUS answers
  `423 Locked` (`NitriteIOException: Directory ... does not exists`).
- The tool chain SIRIUS builds from a `JobSubmission` must be
  `formulas -> fingerprints -> classes -> structures`; omitting CANOPUS
  (`classes`) makes SIRIUS' own CLI parser reject `structures` with
  `400 Cannot create Job Command!`, and an empty submission enables no tool
  at all and fails the same way.
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session

from feature_probabilities import cli_generate_groundtruth
from feature_probabilities.cli_generate_groundtruth import main
from feature_probabilities.schema import (
    SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM,
    Annotation,
    Feature,
    SiriusRun,
    create_database,
)
from feature_probabilities.sirius import Sirius

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO_ROOT = Path(__file__).resolve().parent.parent

#: One row copied verbatim out of MassSpecGym v1.5 at the revision pinned in
#: `config.toml` (`massspecgym.PINNED_MASSSPECGYM_REVISION`): a QTOF spectrum
#: of a `[M+H]+` precursor with 92 peaks. Checked in so the test needs
#: neither the 261 MB dataset download nor network access to HuggingFace.
QTOF_FIXTURE = Path(__file__).parent / "fixtures" / "massspecgym_qtof_single.tsv"
QTOF_IDENTIFIER = "MassSpecGymID0000755"
QTOF_INCHIKEY = "CBGDIJWINPWWJW"
QTOF_PRECURSOR_MZ = 251.0913

#: The checked-in config is what this test drives -- via `--config` with only
#: `--db` overridden -- so it verifies the parameters an operator actually
#: runs with, not a set invented here. An empty `[sirius.analysis_params]`,
#: or one whose tool chain omits `classes`, makes SIRIUS reject every job
#: with `400 Cannot create Job Command!`; that is the failure this test
#: exists to catch.
REPO_CONFIG = REPO_ROOT / "config.toml"


def _sirius_available() -> bool:
    """Whether a licensed SIRIUS this test can actually drive is present.

    Mirrors how `sirius.Sirius` finds SIRIUS: `load_dotenv` then
    `SIRIUS_USER`/`SIRIUS_PW` for the account login, plus either a running
    instance (a `~/.sirius/sirius-*.port` file, which PySirius' SDK attaches
    to) or an executable it can start (`SIRIUS_EXE`, or `sirius` on `PATH`).
    """
    load_dotenv()
    if not (os.getenv("SIRIUS_USER") and os.getenv("SIRIUS_PW")):
        return False
    if any(Path.home().glob(".sirius/sirius-*.port")):
        return True
    executable = os.getenv("SIRIUS_EXE")
    if executable and Path(executable).is_file():
        return True
    return shutil.which("sirius") is not None


pytestmark = pytest.mark.skipif(
    not _sirius_available(),
    reason=(
        "needs a local SIRIUS install plus SIRIUS_USER/SIRIUS_PW; every other "
        "test uses sirius_fake.FakeSirius instead"
    ),
)


@contextmanager
def _open_db(db_path: Path) -> Iterator[Session]:
    engine = create_database(str(db_path))
    try:
        with Session(engine) as session:
            yield session
    finally:
        engine.dispose()


def _open_project_ids(sirius: Sirius) -> set[str]:
    """Project ids currently registered in `sirius`' instance.

    Takes an existing wrapper rather than building its own: every `Sirius()`
    attaches afresh, and a caller that constructs one per query multiplies
    the setup work for a question that only needs one connection.

    Reaches through `Sirius._api` deliberately -- a project left open after
    its chunk, holding its database in a long-lived instance, is invisible
    through `SiriusInterface`. Ask only whether one specific id you created
    is present: the instance is shared with whatever else uses SIRIUS, so
    the set as a whole is not stable.
    """
    return {project.project_id for project in sirius._api.projects().get_projects()}


def test_single_qtof_massspecgym_spectrum_runs_through_real_sirius(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "groundtruth.duckdb"
    monkeypatch.setattr(
        cli_generate_groundtruth,
        "fetch_massspecgym_tsv",
        lambda _revision: QTOF_FIXTURE,
    )
    result = CliRunner().invoke(
        main,
        [
            "--config",
            str(REPO_CONFIG),
            "--db",
            str(db_path),
            "--instrument-type",
            "QTOF",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "FAILED" not in result.output

    with _open_db(db_path) as session:
        runs = session.scalars(select(SiriusRun)).all()
        assert len(runs) == 1
        run = runs[0]
        assert run.instrument_type == "QTOF"
        assert run.source_kind == SOURCE_KIND_GROUND_TRUTH_MASSSPECGYM
        assert run.extract_id is None
        assert run.input_file_path.endswith("qtof_0.mgf")
        assert run.ionization_mode == "positive"

        features = session.scalars(select(Feature)).all()
        assert len(features) == 1
        feature = features[0]
        assert feature.run_id == run.run_id
        assert feature.external_feature_id == QTOF_IDENTIFIER
        assert feature.ion_mass == pytest.approx(QTOF_PRECURSOR_MZ, abs=0.01)
        assert feature.true_inchikey == QTOF_INCHIKEY

        annotations = session.scalars(select(Annotation)).all()
        assert annotations, "structure database search returned no candidates"
        assert {annotation.feature_id for annotation in annotations} == {
            feature.feature_id
        }
        assert min(annotation.rank for annotation in annotations) == 1


def test_close_project_releases_the_project_from_the_sirius_instance(
    tmp_path: Path,
) -> None:
    """`Sirius.close_project` must actually deregister the project.

    A project stays registered -- holding its Nitrite database open -- until
    it is closed, even once its directory is deleted, so a batch that
    creates one project per chunk and closes none accumulates them for the
    instance's lifetime. `run_cache._run_and_persist` calls this per chunk;
    here it is exercised against the real instance, asking only about the
    one project id this test created so a shared instance cannot confuse it.
    """
    sirius = Sirius()
    sirius.create_project(tmp_path / "projects" / "close-me")
    project_id = sirius._require_project_id()
    assert project_id in _open_project_ids(sirius)

    sirius.close_project()

    assert project_id not in _open_project_ids(sirius)


def test_rerunning_the_same_qtof_spectrum_hits_the_cache_instead_of_sirius(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "groundtruth.duckdb"
    monkeypatch.setattr(
        cli_generate_groundtruth,
        "fetch_massspecgym_tsv",
        lambda _revision: QTOF_FIXTURE,
    )
    runner = CliRunner()
    args = [
        "--config",
        str(REPO_CONFIG),
        "--db",
        str(db_path),
        "--instrument-type",
        "QTOF",
    ]

    first = runner.invoke(main, args)
    assert first.exit_code == 0, first.output
    assert "0 cache hit(s), 1 newly run" in first.output

    second = runner.invoke(main, [*args, "--force"])
    assert second.exit_code == 0, second.output
    assert "0 cache hit(s), 1 newly run" in second.output

    third = runner.invoke(main, args)

    assert third.exit_code == 0, third.output
    assert "1 cache hit(s), 0 newly run" in third.output, (
        "a --force run must stay reachable by a later unforced run"
    )
    with _open_db(db_path) as session:
        assert len(session.scalars(select(SiriusRun)).all()) == 2
        assert len(session.scalars(select(Feature)).all()) == 2

"""Command line access to the ISDB class-conditional score densities.

uv run python cli.py fit
uv run python cli.py score --entropy 0.42 --cosine 0.38 --mz 610.3
"""

import json as jsonlib
import os

import click
import numpy as np

from densities import (
    MZ_CUTS,
    ConditionalDensities,
    fit_all,
    posterior_over_candidates,
)


def _load(instrument: str, model_dir: str) -> ConditionalDensities:
    path = os.path.join(model_dir, f"model_{instrument}.pkl")
    if not os.path.exists(path):
        raise click.ClickException(f"No model at {path}. Run `fit` first.")
    return ConditionalDensities.load(path)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """P(score | correct) and P(score | incorrect) for ISDB metabolite annotations."""


@cli.command()
@click.option(
    "--data",
    default="cfmid_scores.parquet",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Labelled ISDB run.",
)
@click.option(
    "--out-dir",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False),
    help="Where to write model_<instrument>.pkl.",
)
@click.option("--quiet", is_flag=True, help="Skip the held-out evaluation report.")
@click.option(
    "--eval-fold",
    type=click.Choice(["val", "test"]),
    default="test",
    show_default=True,
    help="Use val while tuning; test only to report a final number.",
)
@click.option(
    "--mz-cut",
    "mz_cuts",
    type=float,
    multiple=True,
    default=MZ_CUTS,
    show_default=True,
    help="Precursor m/z band edges. Repeat the flag; pass none for no m/z conditioning.",
)
def fit(data, out_dir, quiet, eval_fold, mz_cuts):
    """Fit the densities on the train fold and evaluate on a held-out fold."""
    for inst, path in fit_all(
        data, out_dir, quiet=quiet, eval_fold=eval_fold, mz_cuts=tuple(mz_cuts)
    ).items():
        click.echo(f"-> wrote {path}")


@cli.command()
@click.option(
    "--entropy", type=float, required=True, help="entropy_similarity, in [0, 1]."
)
@click.option(
    "--cosine", type=float, required=True, help="ModifiedCosineGreedy, in [0, 1]."
)
@click.option("--mz", type=float, required=True, help="Precursor m/z of the feature.")
@click.option(
    "--instrument",
    type=click.Choice(["Orbitrap", "QTOF"]),
    default="Orbitrap",
    show_default=True,
    help="Score behaviour differs sharply between the two.",
)
@click.option(
    "--model-dir", default=".", show_default=True, type=click.Path(file_okay=False)
)
@click.option(
    "--prior",
    type=float,
    default=None,
    help="Per-row P(correct) used to turn the ratio into a posterior. "
    "Defaults to the fitted rate for this m/z band.",
)
@click.option("--n-candidates", type=int, default=None,
              help="Rivals this candidate competes with. Adds the per-feature posterior, "
                   "assuming the others are uninformative (LR=1).")
@click.option("--p-absent", type=float, default=None,
              help="P(truth not in the library), for --n-candidates. "
                   "Defaults to the fitted train-fold rate.")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON instead of a table.")
def score(entropy, cosine, mz, instrument, model_dir, prior, n_candidates, p_absent, as_json):
    """P(scores | correct), P(scores | incorrect), and P(correct | scores)."""
    for name, v in (("entropy", entropy), ("cosine", cosine)):
        if not 0.0 <= v <= 1.0:
            raise click.BadParameter(f"{name} must be in [0, 1], got {v}")
    if mz <= 0:
        raise click.BadParameter(f"mz must be positive, got {mz}")
    for name, v in (("prior", prior), ("p-absent", p_absent)):
        if v is not None and not 0.0 < v < 1.0:
            raise click.BadParameter(f"{name} must be in (0, 1), got {v}")
    if n_candidates is not None and n_candidates < 1:
        raise click.BadParameter(f"n-candidates must be >= 1, got {n_candidates}")

    d = _load(instrument, model_dir)
    f1 = float(d.pdf(entropy, cosine, mz, cls=1)[0])
    f0 = float(d.pdf(entropy, cosine, mz, cls=0)[0])
    lr = float(d.likelihood_ratio(entropy, cosine, mz)[0])
    used_prior = float(d.band_prior(mz)[0]) if prior is None else prior
    post = float(d.posterior(entropy, cosine, mz, prior=prior)[0])
    atom = bool(d.is_atom(entropy, cosine)[0])

    feature_post = None
    if n_candidates is not None:
        pa = float(d.p_absent) if p_absent is None else p_absent
        # Rivals are unscored here, so they are given LR=1 -- the neutral assumption.
        feature_post = float(posterior_over_candidates(
            np.r_[lr, np.ones(n_candidates - 1)], pa)[0])

    # A score of exactly 0 means "no shared fragments" and carries finite probability
    # mass, so on those faces the answer is a mass and not a density.
    kind = "probability mass" if atom else "density (per unit score^2)"
    if as_json:
        click.echo(
            jsonlib.dumps(
                {
                    "p_score_given_correct": f1,
                    "p_score_given_incorrect": f0,
                    "likelihood_ratio": lr,
                    "prior": used_prior,
                    "p_correct_given_score": post,
                    "p_correct_given_score_in_feature": feature_post,
                    "quantity": kind,
                    "instrument": instrument,
                    "entropy_similarity": entropy,
                    "ModifiedCosineGreedy": cosine,
                    "precursor_mz": mz,
                },
                indent=2,
            )
        )
        return

    click.echo(f"instrument={instrument}  entropy={entropy}  cosine={cosine}  m/z={mz}")
    click.echo(f"reported as: {kind}")
    click.echo(f"  P(scores | correct)   = {f1:.6g}")
    click.echo(f"  P(scores | incorrect) = {f0:.6g}")
    click.echo(f"  likelihood ratio      = {lr:.4g}")
    if lr > 1:
        click.echo(f"  -> {lr:.3g}x more likely under a correct annotation")
    else:
        click.echo(f"  -> {1 / lr:.3g}x more likely under an incorrect annotation")
    click.echo(f"\n  prior P(correct) for this m/z band = {used_prior:.5f}")
    click.echo(f"  P(correct | scores)   = {post:.5f}")
    if feature_post is not None:
        click.echo(f"  P(correct | scores, {n_candidates} candidates) = {feature_post:.5f}")
    else:
        click.echo(
            "\nThat posterior treats the candidate in isolation. Within a feature only one\n"
            "candidate can be correct, so pass --n-candidates, or score the whole list with\n"
            "posterior_over_candidates(), which also returns P(truth not in library)."
        )


if __name__ == "__main__":
    cli()

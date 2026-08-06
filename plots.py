"""Diagnostic figures for the fitted class-conditional densities.

Two panels per instrument show the fitted marginals P(score | correct) and
P(score | incorrect) against the empirical histograms they are meant to reproduce; a
third shows the log likelihood ratio over the score plane, which is the only part a
consensus model actually consumes.

uv run python plots.py
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from simple_densities import COS, ENT, EPS, PATTERNS, SimpleDensities, load_labeled, logit

# Two series -- identity, so categorical slots 1 and 2. Diverging blue<->red with a
# neutral grey midpoint for the log-ratio, whose zero means "no evidence either way".
CORRECT, INCORRECT = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list(
    "lr", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f0a58c", "#d03b3b", "#7d1f1f"]
)

mpl.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
    }
)


def fitted_marginal(model: SimpleDensities, cls: int, which: int, n: int = 400):
    """Marginal of one score under the fitted model, on the raw [0, 1] scale.

    The model lives on the logit scale inside each zero-pattern, so this integrates out
    the other coordinates, sums the patterns where the score is live, and applies the
    Jacobian |dlogit/ds| = 1/(s(1-s)) to land back on score units. Returns the point
    mass at score == 0 alongside the continuous part -- they are different kinds of
    number and the plot keeps them apart.
    """
    atom = sum(
        np.exp(model.log_pi[p, cls]) for p in PATTERNS if not PATTERNS[p][which]
    )
    s = np.linspace(EPS, 1 - EPS, n)
    dens = np.zeros(n)
    for p in PATTERNS:
        if not PATTERNS[p][which] or (p, cls) not in model.dens:
            continue
        d = model.dens[p, cls]
        # Position of this score among the pattern's live coordinates.
        axis = sum(PATTERNS[p][:which]) if which else 0
        c, f = d.marginal(axis)
        dens += np.exp(model.log_pi[p, cls]) * np.interp(logit(s), c, f, left=0, right=0)
    return s, dens / (s * (1 - s)), atom


def lr_surface(model: SimpleDensities, mz: float, n: int = 160, support: float = 1e-4):
    """log10 likelihood ratio over the (entropy, cosine) plane at one precursor m/z.

    Cells where almost no incorrect candidate ever lands are masked out. The ratio there
    is a quotient of two near-zero smoothed counts -- it reaches ±280 log-units purely
    from the density floor, which would set the colour scale from noise and leave the
    populated region flat.
    """
    s = np.linspace(EPS, 1 - EPS, n)
    ee, cc = np.meshgrid(s, s, indexing="ij")
    q = pl.DataFrame(
        {ENT: ee.ravel(), COS: cc.ravel(), "precursor_mz": np.full(ee.size, mz)}
    )
    lr = (model.log_likelihood_ratio(q) / np.log(10)).reshape(ee.shape)
    f0 = model.pdf(q, 0).reshape(ee.shape)
    return s, np.ma.masked_where(f0 < support * f0.max(), lr)


def figure(method: str = "kde", path: str = "densities_fit.png") -> str:
    df = load_labeled()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2))

    for r, inst in enumerate(("Orbitrap", "QTOF")):
        model = SimpleDensities.load(f"simple_{method}_{inst}.pkl")
        tr = df.filter((pl.col("instrument") == inst) & (pl.col("fold") == "train"))
        mz_med = float(tr["precursor_mz"].median())

        for k, (col, label) in enumerate(
            ((ENT, "entropy_similarity"), (COS, "ModifiedCosineGreedy"))
        ):
            ax = axes[r, k]
            top = 0.0
            for cls, colour, name in ((0, INCORRECT, "incorrect"), (1, CORRECT, "correct")):
                obs = tr.filter(pl.col("is_correct") == cls)[col].to_numpy()
                live = obs[obs > 0]
                top = max(top, np.histogram(live, bins=60, range=(0, 1), density=True)[0].max())
                # Empirical histogram scaled to the same footing as the fitted curve:
                # it integrates to P(score > 0), not to 1.
                ax.hist(
                    live, bins=60, range=(0, 1), density=True, weights=None,
                    histtype="stepfilled", color=colour, alpha=0.16,
                    edgecolor="none", zorder=1,
                )
                s, f, atom = fitted_marginal(model, cls, k)
                # hist(density=True) normalises over the plotted subset, so rescale the
                # fitted curve by the same factor to make the two directly comparable.
                ax.plot(s, f / max(1 - atom, 1e-9), color=colour, lw=2, zorder=3,
                        label=f"{name}  (P(=0) = {atom:.3f})")
            ax.set_xlim(0, 1)
            # The change of variables from logit to score units carries a 1/(s(1-s))
            # Jacobian, so the fitted curve diverges as s -> 0. Frame on the histogram.
            ax.set_ylim(0, top * 1.25)
            ax.set_xlabel(label)
            ax.set_ylabel("density | class" if k == 0 else "")
            ax.grid(axis="y", zorder=0)
            ax.set_axisbelow(True)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
            if k == 0:
                ax.text(-0.16, 1.06, inst, transform=ax.transAxes, fontsize=11,
                        weight="bold", color=INK)

        ax = axes[r, 2]
        s, lr = lr_surface(model, mz_med)
        lim = float(np.percentile(np.abs(lr.compressed()), 99))
        im = ax.pcolormesh(s, s, lr.T, cmap=DIVERGING, shading="auto",
                           norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
        ax.contour(s, s, lr.T, levels=[0.0], colors=[INK2], linewidths=1.2)
        ax.set_facecolor(GRID)
        ax.set_xlabel("entropy_similarity")
        ax.set_ylabel("ModifiedCosineGreedy")
        ax.set_title(f"log₁₀ LR   at m/z {mz_med:.0f}   (grey = no data)",
                     fontsize=9, color=INK2, pad=6)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(color=MUTED, labelcolor=MUTED)

    fig.suptitle(
        f"P(score | correct) vs P(score | incorrect) — {method.upper()} fit, train fold\n"
        "filled = observed, line = fitted; the contour marks LR = 1, where the scores "
        "say nothing either way",
        fontsize=10.5, color=INK, y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    return path


if __name__ == "__main__":
    print("->", figure())

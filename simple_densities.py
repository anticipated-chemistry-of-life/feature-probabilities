"""Simple class-conditional score densities: LDA / QDA / KDE.

Estimates P(entropy, cosine, precursor_mz | correct) and P(... | incorrect) for ISDB
annotations on MassSpecGym, using textbook density estimators rather than the boosted
likelihood-ratio fit in `densities.py`. Same target quantity, deliberately simpler
machinery, so the two can be compared on the same held-out folds.

Three data facts drive the structure:

* A score of exactly 0 means "no shared fragments", not "very dissimilar". 12.6% of
  incorrect rows and 8.5% of correct rows have entropy_similarity == 0. That is a point
  mass, and no continuous density can represent it -- squashed through a logit it
  becomes a 12% spike at the far tail and wrecks any Gaussian fit. So the joint is
  written as a pattern mixture: first which scores are zero, then a continuous density
  over the ones that are not.
* The two scores correlate at r = 0.81, so they are modelled jointly. Multiplying two
  marginal likelihood ratios would double-count the same fragment matches.
* No transform makes both classes Gaussian at once. On the raw scale the negatives are
  right-skewed (skew +0.9 / +1.2) and the positives are near-flat (kurtosis -0.5 /
  -1.2); a logit fixes the negatives and over-corrects the positives. The logit is used
  anyway for LDA/QDA because it at least puts both classes on an unbounded scale with
  comparable spread -- but this is exactly the assumption the KDE exists to drop, and
  the gap between them measures what it costs.

precursor_mz enters as a third density dimension, so the fitted joint is
f(entropy, cosine, mz | class) and the likelihood ratio carries an f(mz | class) factor
alongside the score-quality one. `mz_split()` separates the two. Note that mz is
constant across a feature's candidates, so it cannot help rank them -- it only helps
decide whether the true molecule is in the library at all.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

ENT, COS = "entropy_similarity", "ModifiedCosineGreedy"

# Zero-score patterns, indexed by 2*(ent > 0) + (cos > 0). The tuple lists which
# continuous coordinates are live; log(mz) is always live, hence the trailing True.
PATTERNS = {
    0: (False, False),  # both zero  -> density over log(mz) alone
    1: (False, True),  # entropy zero, cosine live
    2: (True, False),  # cosine zero, entropy live
    3: (True, True),  # both live
}
EPS = 1e-3  # logit clip; scores are reported to ~3 decimals so this is below resolution


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------- data


def load_labeled(path: str = "cfmid_scores.parquet") -> pl.DataFrame:
    """Load the ISDB run, label rows, recover precursor m/z, keep [M+H]+.

    Precursor m/z is not stored but follows from the two difference columns:
    ppm = diff / mz * 1e6. That breaks when the difference is exactly 0 (389k rows), so
    those are filled from their feature-mates -- m/z is constant within a feature to
    ~5 mDa, which is the rounding in the stored diffs. Only 3% of features have no
    recoverable m/z at all.

    [M+Na]+ rows are dropped: ISDB covers [M+H]+ only, so all 551k of them are labelled
    incorrect by construction. Keeping them would inflate f0 with rows that carry no
    matching f1.
    """
    df = (
        pl.read_parquet(path)
        .filter(pl.col("adduct") == "[M+H]+")
        .with_columns(
            (pl.col("inchikey_isdb") == pl.col("inchikey_msg"))
            .cast(pl.Int8)
            .alias("is_correct"),
            pl.when(pl.col("ppm_precursor_mz_diff").abs() > 1e-9)
            .then(
                pl.col("abs_precursor_mz_diff") / pl.col("ppm_precursor_mz_diff") * 1e6
            )
            .otherwise(None)
            .alias("precursor_mz"),
        )
    )
    return (
        df.with_columns(
            pl.col("precursor_mz").fill_null(
                pl.col("precursor_mz").median().over("feature_id")
            )
        )
        .drop_nulls("precursor_mz")
        .with_columns(
            pl.col("is_correct").max().over("feature_id").alias("truth_present"),
        )
    )


def coords(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(n, 3) matrix of [logit(entropy), logit(cosine), log(mz)] and the pattern index.

    Columns where the underlying score is 0 hold a placeholder; the pattern index says
    which ones to ignore. Nothing downstream reads a dead column.
    """
    ent = df[ENT].to_numpy().astype(float)
    cos = df[COS].to_numpy().astype(float)
    mz = df["precursor_mz"].to_numpy().astype(float)
    X = np.column_stack([logit(ent), logit(cos), np.log(mz)])
    pat = 2 * (ent > 0).astype(int) + (cos > 0).astype(int)
    return X, pat


def _live(pat: int) -> np.ndarray:
    """Column indices of the live coordinates for a pattern (m/z is always live)."""
    return np.array([i for i, on in enumerate(PATTERNS[pat] + (True,)) if on])


# ------------------------------------------------------------------ density backends


@dataclass
class Gaussian:
    """Multivariate normal fitted by moments. Shared covariance across classes = LDA."""

    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    cov: np.ndarray = field(default_factory=lambda: np.array([]))
    _chol: np.ndarray = field(default_factory=lambda: np.array([]))
    _logdet: float = 0.0

    def fit(self, X: np.ndarray, cov: np.ndarray | None = None) -> "Gaussian":
        self.mean = X.mean(axis=0)
        c = np.atleast_2d(np.cov(X, rowvar=False)) if cov is None else cov
        # Ridge on the diagonal: with a handful of positives in a sparse pattern the
        # empirical covariance can be near-singular.
        self.cov = c + 1e-6 * np.eye(c.shape[0])
        self._chol = np.linalg.cholesky(self.cov)
        self._logdet = 2.0 * np.log(np.diag(self._chol)).sum()
        return self

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        z = np.linalg.solve(self._chol, (X - self.mean).T)
        d = X.shape[1]
        return -0.5 * ((z**2).sum(axis=0) + self._logdet + d * np.log(2 * np.pi))


@dataclass
class BinnedKDE:
    """Product-Gaussian KDE evaluated on a grid, so scoring is a table lookup.

    A direct KDE over 4.3M reference points costs O(n) per query. Binning first and
    convolving with a Gaussian gives the same estimator up to the bin width, at fixed
    cost. Bandwidth is Scott's rule per dimension; `reflect` padding keeps mass from
    leaking off the edge of the logit range, where a lot of the data actually sits.
    """

    n_bins: int = 48
    edges: list[np.ndarray] = field(default_factory=list)
    grid: np.ndarray = field(default_factory=lambda: np.array([]))

    def fit(self, X: np.ndarray) -> "BinnedKDE":
        n, d = X.shape
        # Pad the range so the reflected boundary sits outside the support of the data.
        lo, hi = X.min(axis=0), X.max(axis=0)
        pad = 0.05 * np.maximum(hi - lo, 1e-6)
        self.edges = [
            np.linspace(lo[j] - pad[j], hi[j] + pad[j], self.n_bins + 1)
            for j in range(d)
        ]
        idx = [
            np.clip(np.digitize(X[:, j], self.edges[j][1:-1]), 0, self.n_bins - 1)
            for j in range(d)
        ]
        flat = np.ravel_multi_index(idx, (self.n_bins,) * d)
        counts = np.bincount(flat, minlength=self.n_bins**d).astype(float)
        counts = counts.reshape((self.n_bins,) * d)

        widths = np.array([e[1] - e[0] for e in self.edges])
        sigma = n ** (-1.0 / (d + 4)) * X.std(axis=0)
        smoothed = gaussian_filter(counts, sigma=sigma / widths, mode="reflect")
        # Normalise to a density: total mass 1 over the grid, divided by cell volume.
        self.grid = smoothed / (smoothed.sum() * np.prod(widths))
        return self

    def logpdf(self, X: np.ndarray) -> np.ndarray:
        idx = tuple(
            np.clip(np.digitize(X[:, j], self.edges[j][1:-1]), 0, self.n_bins - 1)
            for j in range(X.shape[1])
        )
        # A query can land in a cell no reference point reached; floor it rather than
        # returning -inf, which would make the ratio meaningless.
        return np.log(np.maximum(self.grid[idx], 1e-300))

    def marginal(self, keep: int) -> tuple[np.ndarray, np.ndarray]:
        """Integrate out every dimension but `keep`. Returns (bin centres, density)."""
        drop = tuple(j for j in range(self.grid.ndim) if j != keep)
        widths = [self.edges[j][1] - self.edges[j][0] for j in drop]
        dens = self.grid.sum(axis=drop) * np.prod(widths) if drop else self.grid
        e = self.edges[keep]
        return 0.5 * (e[:-1] + e[1:]), dens


# -------------------------------------------------------------------- the full model


@dataclass
class SimpleDensities:
    """P(entropy, cosine, mz | class) as a mixture over zero-score patterns.

    P(x, pattern | y) = pi[pattern | y] * f[pattern, y](live coordinates)

    The pattern probabilities carry real signal on their own: a row with no shared
    fragments is *less* likely to be correct than one with fragments, but *more* likely
    than one that shares fragments and still scores near zero. Folding zeros into the
    bottom of the continuous scale loses that.

    `method`:
      "lda" -- Gaussian per class, covariance pooled across classes (classic LDA).
      "qda" -- Gaussian per class, each with its own covariance.
      "kde" -- binned Gaussian KDE per class, no distributional assumption.
    """

    method: str = "kde"
    n_bins: int = 48
    min_fit: int = 200  # below this a pattern/class cell is not fitted separately
    prior: float = 0.0
    p_absent: float = 0.0
    log_pi: dict = field(
        default_factory=dict
    )  # (pattern, class) -> log P(pattern|class)
    dens: dict = field(default_factory=dict)  # (pattern, class) -> density backend

    def fit(self, df: pl.DataFrame) -> "SimpleDensities":
        X, pat = coords(df)
        y = df["is_correct"].to_numpy()
        self.prior = float(y.mean())
        self.p_absent = 1.0 - float(
            df.group_by("feature_id")
            .agg(pl.col("truth_present").first())["truth_present"]
            .mean()
        )

        for cls in (0, 1):
            m = y == cls
            # Laplace-smoothed pattern probabilities; some patterns are rare in the
            # positive class and an empty one must not produce a zero likelihood.
            n_pat = np.bincount(pat[m], minlength=4) + 0.5
            for p in PATTERNS:
                self.log_pi[p, cls] = float(np.log(n_pat[p] / n_pat.sum()))

        for p in PATTERNS:
            cols = _live(p)
            sub = {cls: X[np.ix_((pat == p) & (y == cls), cols)] for cls in (0, 1)}
            # A pattern too sparse in either class gets no continuous density; the
            # pattern probability alone then carries all of its evidence.
            if min(len(sub[0]), len(sub[1])) < self.min_fit:
                continue
            if self.method == "kde":
                for cls in (0, 1):
                    self.dens[p, cls] = BinnedKDE(n_bins=self.n_bins).fit(sub[cls])
            else:
                pooled = None
                if self.method == "lda":
                    # Pooled within-class scatter, the LDA assumption: one covariance,
                    # two means. Weighted by class size, so the 1.2% positives barely
                    # move it -- which is the point, and also its main weakness here.
                    dfree = len(sub[0]) + len(sub[1]) - 2
                    pooled = (
                        sum(
                            np.atleast_2d(np.cov(sub[c], rowvar=False))
                            * (len(sub[c]) - 1)
                            for c in (0, 1)
                        )
                        / dfree
                    )
                for cls in (0, 1):
                    self.dens[p, cls] = Gaussian().fit(sub[cls], cov=pooled)
        return self

    # -- scoring ---------------------------------------------------------------
    def _logpdf(self, X: np.ndarray, pat: np.ndarray, cls: int) -> np.ndarray:
        out = np.empty(len(X))
        for p in PATTERNS:
            m = pat == p
            if not m.any():
                continue
            out[m] = self.log_pi[p, cls]
            d = self.dens.get((p, cls))
            if d is not None:
                out[m] += d.logpdf(X[np.ix_(m, _live(p))])
        return out

    def log_likelihood_ratio(self, df: pl.DataFrame) -> np.ndarray:
        """log P(x | correct) - log P(x | incorrect). Dimensionless within a pattern."""
        X, pat = coords(df)
        return self._logpdf(X, pat, 1) - self._logpdf(X, pat, 0)

    def pdf(self, df: pl.DataFrame, cls: int) -> np.ndarray:
        """P(entropy, cosine, mz | cls).

        Mixed units by construction: a density in the live coordinates, a probability
        mass in the ones pinned to zero. Comparable across rows only within a pattern.
        Use `log_likelihood_ratio` for anything that must compare rows.
        """
        X, pat = coords(df)
        return np.exp(self._logpdf(X, pat, cls))

    def posterior(self, df: pl.DataFrame, prior: float | None = None) -> np.ndarray:
        """P(correct | x) for a candidate considered on its own, ignoring its rivals."""
        p = self.prior if prior is None else prior
        log_odds = self.log_likelihood_ratio(df) + np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-log_odds))

    def mz_split(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Split the log-LR into an m/z part and a score-quality-given-m/z part.

        The fitted joint is f(ent, cos, mz | y), so its ratio mixes "correct answers are
        more common at low m/z" with "the scores discriminate better at low m/z". Only
        the second can rank candidates within a feature -- m/z is identical across them.
        Returns (log ratio of the m/z marginals, remainder).
        """
        X, pat = coords(df)
        total = self._logpdf(X, pat, 1) - self._logpdf(X, pat, 0)
        # Pattern 0 has m/z as its only live coordinate, so its density *is* the
        # m/z marginal. Reuse it for every row regardless of that row's own pattern.
        mz_only = np.column_stack([X[:, 2]])
        d1, d0 = self.dens.get((0, 1)), self.dens.get((0, 0))
        if d1 is None or d0 is None:
            return np.zeros(len(X)), total
        mz_part = d1.logpdf(mz_only) - d0.logpdf(mz_only)
        return mz_part, total - mz_part

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "SimpleDensities":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def posterior_over_candidates(log_lr: np.ndarray, p_absent: float) -> np.ndarray:
    """Posterior that each candidate is correct, given at most one of them can be.

    Within a feature the f0 term is shared by every candidate and cancels, so only the
    ratio survives -- which is why the ratio, not the two densities, is what a consensus
    model needs. Returns len(log_lr) + 1 entries; the last is P(truth not in library).
    """
    log_lr = np.asarray(log_lr, float)
    log_w = np.r_[
        log_lr + np.log((1.0 - p_absent) / len(log_lr)),
        np.log(max(p_absent, 1e-12)),
    ]
    w = np.exp(log_w - log_w.max())
    return w / w.sum()


# ----------------------------------------------------------------------- evaluation


def cluster_bootstrap_auc(
    df: pl.DataFrame, score: np.ndarray, n_boot: int = 200, seed: int = 0
) -> tuple[float, float, float]:
    """AUC with a 95% CI, resampling whole molecules rather than rows.

    MassSpecGym carries many replicate spectra per molecule, so rows are nowhere near
    independent. A row-level interval understates the spread by more than an order of
    magnitude.
    """
    y = df["is_correct"].to_numpy()
    codes = df["inchikey_msg"].to_physical().rank("dense").to_numpy() - 1
    order = np.argsort(codes, kind="stable")
    groups = np.split(
        order, np.searchsorted(codes[order], np.arange(1, codes.max() + 1))
    )

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        yb = y[idx]
        if 0 < yb.sum() < len(yb):
            boots.append(roc_auc_score(yb, score[idx]))
    return (
        float(roc_auc_score(y, score)),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def _top1(scores: list, labels: list) -> float:
    return float(
        np.mean([int(np.argmax(s) == np.argmax(y)) for s, y in zip(scores, labels)])
    )


def report(name: str, model: SimpleDensities, te: pl.DataFrame) -> dict:
    """Held-out evaluation on the three things the densities are actually for."""
    llr = model.log_likelihood_ratio(te)
    t = te.with_columns(pl.Series("llr", llr))
    y = te["is_correct"].to_numpy()
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(
        f"rows={len(y):,}  correct={y.sum():,}  "
        f"distinct correct molecules="
        f"{te.filter(pl.col('is_correct') == 1)['inchikey_msg'].n_unique():,}"
    )

    # 1. Ranking candidates inside one feature. m/z is constant here, so it contributes
    #    nothing; this isolates what the two scores jointly add over either alone.
    g = (
        t.filter(pl.col("truth_present") == 1)
        .group_by("feature_id")
        .agg(pl.col("llr"), pl.col(ENT), pl.col(COS), pl.col("is_correct"))
    )
    lab = g["is_correct"].to_list()
    top1 = {
        ENT: _top1(g[ENT].to_list(), lab),
        COS: _top1(g[COS].to_list(), lab),
        "joint log-LR": _top1(g["llr"].to_list(), lab),
    }
    print(f"\nwithin-feature top-1 accuracy (truth present, n={g.height:,}):")
    for k, v in top1.items():
        print(f"  {k:22s} {v:.3f}")

    # 2. Is the true molecule in the library at all -- the feature-level call, and the
    #    one a consensus model most needs calibrated.
    gf = t.group_by("feature_id").agg(pl.col("llr"), pl.col("truth_present").first())
    pa = model.p_absent
    p_none = np.array(
        [posterior_over_candidates(np.asarray(x), pa)[-1] for x in gf["llr"].to_list()]
    )
    absent = 1 - gf["truth_present"].to_numpy()
    brier, base = np.mean((p_none - absent) ** 2), np.mean((pa - absent) ** 2)
    print(
        f"\nP(truth not in library):  train prior={pa:.3f}  held-out actual={absent.mean():.3f}"
    )
    print(f"  AUC  = {roc_auc_score(absent, p_none):.3f}")
    print(f"  Brier= {brier:.4f}   constant-prior baseline = {base:.4f}")

    # 3. Pooled per-row AUC, reported last and with a caveat: it mixes within-feature
    #    discrimination with telling easy features from hard ones, so it flatters the
    #    model relative to what it contributes to ranking.
    auc, lo, hi = cluster_bootstrap_auc(te, llr)
    print("\npooled per-row AUC (between- and within-feature combined):")
    print(f"  {'joint log-LR':22s} {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    for s in (ENT, COS):
        a, l, h = cluster_bootstrap_auc(te, te[s].to_numpy().astype(float))
        print(f"  {s:22s} {a:.3f}  95% CI [{l:.3f}, {h:.3f}]")

    return {"top1": top1, "auc": auc, "brier": brier, "brier_base": base}


def fit_all(
    path: str = "cfmid_scores.parquet",
    method: str = "kde",
    out_dir: str = ".",
    eval_fold: str = "val",
    quiet: bool = False,
) -> dict:
    """Fit one model per instrument on the train fold, evaluate on a held-out fold.

    Orbitrap and QTOF are fitted separately: the score distributions differ enough that
    a pooled fit is a compromise between two populations rather than a description of
    either. Use `eval_fold="val"` while choosing settings and "test" only to report.
    """
    df = load_labeled(path)
    written = {}
    for inst in ("Orbitrap", "QTOF"):
        sub = df.filter(pl.col("instrument") == inst)
        model = SimpleDensities(method=method).fit(
            sub.filter(pl.col("fold") == "train")
        )
        if not quiet:
            report(
                f"{inst}  method={method}  evaluated on: {eval_fold}",
                model,
                sub.filter(pl.col("fold") == eval_fold),
            )
        out = f"{out_dir.rstrip('/')}/simple_{method}_{inst}.pkl"
        model.save(out)
        written[inst] = out
    return written


if __name__ == "__main__":
    for method in ("lda", "qda", "kde"):
        fit_all(method=method)

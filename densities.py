"""Class-conditional score densities for ISDB annotations on MassSpecGym.

Estimates P(scores | correct) and P(scores | incorrect) for the pair
(entropy_similarity, ModifiedCosineGreedy), conditioned on precursor m/z and
instrument.

Three things drive the design:

* The two ISDB scores correlate at r = 0.81, so they are modelled jointly. Treating
  them as independent evidence and multiplying their likelihood ratios would roughly
  double-count the same fragment matches.
* Score quality degrades strongly with molecular size -- within the train fold alone,
  per-row AUC falls from 0.80 below 400 Da to 0.57 in the 700-900 Da range. Precursor
  m/z is available at inference time, so it is a conditioning axis rather than a
  nuisance to correct for afterwards.
* A score of exactly 0 means "no shared fragments", not "very dissimilar". Those rows
  are *more* likely to be correct than genuinely-low-scoring ones, so the value 0 gets
  its own indicator instead of sitting at the bottom of the scale.

The fit is on the likelihood-ratio scale: a monotone gradient-boosted model gives
LR = f1/f0, f0 comes from a histogram of the negatives (abundant), and f1 = LR * f0.
Estimating f1 directly would rest on ~36k positives spread over the whole grid.

What the conditioning does and does not buy, measured on held-out folds:

* It clearly improves P(truth not in library) -- Brier 0.150 against a 0.191
  constant-prior baseline on Orbitrap test, AUC 0.764 for flagging features whose
  answer is absent. That is the state a consensus model most needs calibrated.
* It does not improve ranking candidates within a feature. m/z is identical across a
  feature's candidates so it cannot contribute there, and the joint of the two scores
  does not beat entropy_similarity alone (Orbitrap top-1 0.308 vs 0.314). Pooled
  per-row AUC looks far better than either score (0.807 vs 0.501) but that gap is
  almost entirely between-feature and should not be read as ranking skill.
"""

# %%
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

ENT, COS = "entropy_similarity", "ModifiedCosineGreedy"
FEATURES = [ENT, COS, "log_mz", "ent_zero", "cos_zero"]
# Monotone in both similarity scores; m/z and the zero indicators are unconstrained so
# the model can lift the "no shared fragments" rows back above the low-score region.
MONOTONIC = [1, 1, 0, 0, 0]
# m/z band edges, selected on the val fold. CFM-ID quality degrades with molecular
# size, but candidate rows are far too sparse above 600 Da for data-driven edges to
# land there, so the cuts are placed by where the effect is rather than by row counts.
MZ_CUTS = (500.0, 700.0, 900.0)


# %%
def load_labeled(path: str = "cfmid_scores.parquet") -> pl.DataFrame:
    """Load the ISDB run, label rows, and derive precursor m/z.

    Precursor m/z is not stored directly but follows from the two difference columns:
    ppm = diff / mz * 1e6. The recovered value matches exact mass + adduct to a few mDa.
    """
    df = pl.read_parquet(path).with_columns(
        (pl.col("inchikey_isdb") == pl.col("inchikey_msg"))
        .cast(pl.Int8)
        .alias("is_correct")
    )
    return df.with_columns(
        pl.when(pl.col("ppm_precursor_mz_diff").abs() > 1e-9)
        .then(pl.col("abs_precursor_mz_diff") / pl.col("ppm_precursor_mz_diff") * 1e6)
        .otherwise(None)
        .alias("precursor_mz"),
        pl.len().over("feature_id").alias("n_cand"),
        pl.col("is_correct").sum().over("feature_id").alias("truth_present"),
    )


def design_matrix(df: pl.DataFrame) -> np.ndarray:
    ent = df[ENT].to_numpy().astype(float)
    cos = df[COS].to_numpy().astype(float)
    mz = df["precursor_mz"].to_numpy().astype(float)
    return np.column_stack(
        [ent, cos, np.log(mz), (ent <= 0).astype(float), (cos <= 0).astype(float)]
    )


# %%
@dataclass
class ConditionalDensities:
    """Joint class-conditional densities over (entropy, cosine) given precursor m/z.

    `pdf` returns a density in score-units^-2 where both scores are positive, and a
    probability mass where either is exactly 0 -- the distribution is a mixture of a
    continuous part and atoms on the score==0 faces, so a single number cannot be a
    density everywhere. `likelihood_ratio` is dimensionless and always comparable.
    """

    n_score_bins: int = 16
    mz_cuts: tuple[float, ...] = MZ_CUTS
    min_cell: int = 25
    model: HistGradientBoostingClassifier | None = None
    prior: float = 0.0
    p_absent: float = 0.0
    prior_by_band: np.ndarray = field(default_factory=lambda: np.array([]))
    ent_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    cos_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    mz_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    hist0: np.ndarray = field(default_factory=lambda: np.array([]))
    lr_grid: np.ndarray = field(default_factory=lambda: np.array([]))

    # -- fitting ----------------------------------------------------------------
    def fit(self, df: pl.DataFrame) -> "ConditionalDensities":
        df = df.drop_nulls("precursor_mz")
        X, y = design_matrix(df), df["is_correct"].to_numpy()
        self.prior = float(y.mean())

        self.model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=200,
            l2_regularization=1.0,
            monotonic_cst=MONOTONIC,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=0,
        ).fit(X, y)

        self._build_grid(df.filter(pl.col("is_correct") == 0))

        # The ratio is normalised inside each m/z band, so turning it into a posterior
        # needs the prior from that same band. Pooling would be wrong: the per-row rate
        # runs from 0.012 below 500 Da to 0.052 above 900 Da.
        _, _, im = self._cells(df)
        nb = len(self.mz_edges) - 1
        tot = np.bincount(im, minlength=nb)
        self.prior_by_band = np.where(
            tot > 0, np.bincount(im, weights=y.astype(float), minlength=nb) / np.maximum(tot, 1), self.prior
        )
        # Fraction of features whose true molecule is absent from the library entirely.
        # Fitted for convenience, but it is the least transferable quantity in the model
        # -- override it on any population that is not distributed like the train fold.
        self.p_absent = 1.0 - float(
            df.group_by("feature_id").agg(pl.col("truth_present").first())["truth_present"].mean()
        )
        return self

    def _build_grid(self, neg: pl.DataFrame) -> None:
        """Discretise both densities onto a shared grid so they stay mutually consistent.

        The model gives a smooth pointwise ratio, but f0 is a histogram. Mixing the two
        resolutions makes f1 = LR * f0 integrate to something other than 1, so the ratio
        is aggregated to the same cells as the histogram: within a cell it is the mean
        model ratio over the negatives that landed there, which is exactly the cell's
        share of the f1 mass divided by its share of the f0 mass.
        """
        qs = np.linspace(0, 1, self.n_score_bins + 1)[1:-1]
        self.ent_edges = np.unique(
            np.r_[
                0.0,
                np.quantile(
                    neg.filter(pl.col(ENT) > 0)[ENT].to_numpy().astype(float), qs
                ),
                1.0,
            ]
        )
        self.cos_edges = np.unique(
            np.r_[
                0.0,
                np.quantile(
                    neg.filter(pl.col(COS) > 0)[COS].to_numpy().astype(float), qs
                ),
                1.0,
            ]
        )
        # A few wide m/z bands, not quantile bins. Equal-count bins put almost every edge
        # below 460 Da, which splits the dense low-mass region into noise while leaving
        # the whole high-mass range -- where score quality actually collapses -- in one
        # bin. On the val fold, fine quantile binning was the worst of every option
        # tried; these cuts were the best. Cuts with too few rows above them are dropped.
        mz = neg["precursor_mz"].to_numpy().astype(float)
        cuts = [
            c for c in sorted(self.mz_cuts) if (mz >= c).sum() >= self.min_cell * 20
        ]
        self.mz_edges = np.unique(np.r_[0.0, cuts, np.inf])

        ie, ic, im = self._cells(neg)
        shape = (len(self.mz_edges) - 1, len(self.ent_edges), len(self.cos_edges))
        flat_idx = (im * shape[1] + ie) * shape[2] + ic
        size = int(np.prod(shape))
        n0 = np.bincount(flat_idx, minlength=size).astype(float).reshape(shape)
        sum_lr = np.bincount(
            flat_idx, weights=self._raw_lr(design_matrix(neg)), minlength=size
        ).reshape(shape)

        # Sparse cells fall back to the model evaluated at the cell centre; the histogram
        # has too few negatives there for an average to mean anything.
        with np.errstate(invalid="ignore", divide="ignore"):
            cell_lr = np.where(n0 >= self.min_cell, sum_lr / np.maximum(n0, 1), np.nan)
        centre_lr = self._raw_lr(self._centre_design()).reshape(shape)
        cell_lr = np.where(np.isnan(cell_lr), centre_lr, cell_lr)

        # Laplace smoothing keeps empty cells from producing an infinite ratio.
        h = n0 + 0.5
        # Condition on m/z: each m/z slice is its own distribution over the score grid.
        self.hist0 = h / h.sum(axis=(1, 2), keepdims=True)

        # Averaging within cells mixes cells whose internal score distributions differ,
        # which can invert the ordering the underlying model guarantees. Project back
        # onto the monotone cone before normalising.
        cell_lr = self._monotone_2d(cell_lr, h)

        # Rescale per m/z slice so that sum(LR * f0) = 1, i.e. f1 is a proper density.
        z = (cell_lr * self.hist0).sum(axis=(1, 2), keepdims=True)
        self.lr_grid = cell_lr / np.where(z > 0, z, 1.0)

    @staticmethod
    def _monotone_2d(lr: np.ndarray, w: np.ndarray, n_sweeps: int = 20) -> np.ndarray:
        """Make the ratio non-decreasing in both scores by alternating weighted PAVA.

        Row index 0 on each score axis is the score==0 atom, which is a separate
        category rather than the bottom of the scale, so it is left out of the ordering.
        """
        out = lr.copy()
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        for _ in range(n_sweeps):
            prev = out.copy()
            for m in range(out.shape[0]):
                blk, wb = out[m, 1:, 1:], w[m, 1:, 1:]
                x = np.arange(blk.shape[0])
                for j in range(blk.shape[1]):
                    blk[:, j] = iso.fit_transform(x, blk[:, j], sample_weight=wb[:, j])
                x = np.arange(blk.shape[1])
                for i in range(blk.shape[0]):
                    blk[i, :] = iso.fit_transform(x, blk[i, :], sample_weight=wb[i, :])
            if np.max(np.abs(out - prev)) < 1e-9:
                break
        return out

    def _centre_design(self) -> np.ndarray:
        """Design matrix for every grid cell centre, in the same order as the grid."""
        ent_c = np.r_[0.0, 0.5 * (self.ent_edges[:-1] + self.ent_edges[1:])]
        cos_c = np.r_[0.0, 0.5 * (self.cos_edges[:-1] + self.cos_edges[1:])]
        hi = np.where(
            np.isfinite(self.mz_edges[1:]), self.mz_edges[1:], self.mz_edges[:-1] * 1.5
        )
        mz_c = np.clip(0.5 * (self.mz_edges[:-1] + hi), 50.0, None)
        m, e, c = np.meshgrid(mz_c, ent_c, cos_c, indexing="ij")
        e, c, m = e.ravel(), c.ravel(), m.ravel()
        return np.column_stack(
            [e, c, np.log(m), (e <= 0).astype(float), (c <= 0).astype(float)]
        )

    # -- lookup helpers ---------------------------------------------------------
    def _cells(self, df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._cells_raw(
            df[ENT].to_numpy().astype(float),
            df[COS].to_numpy().astype(float),
            df["precursor_mz"].to_numpy().astype(float),
        )

    def _cells_raw(self, ent, cos, mz):
        # Index 0 of each score axis is the score==0 atom; continuous bins start at 1.
        ie = np.where(ent <= 0, 0, np.digitize(ent, self.ent_edges[1:-1]) + 1)
        ic = np.where(cos <= 0, 0, np.digitize(cos, self.cos_edges[1:-1]) + 1)
        im = np.clip(np.digitize(mz, self.mz_edges[1:-1]), 0, len(self.mz_edges) - 2)
        return (
            np.clip(ie, 0, len(self.ent_edges) - 1),
            np.clip(ic, 0, len(self.cos_edges) - 1),
            im,
        )

    def _raw_lr(self, X: np.ndarray) -> np.ndarray:
        p = np.clip(self.model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return (p / (1 - p)) / (self.prior / (1 - self.prior))

    def _widths(self, ie: np.ndarray, ic: np.ndarray) -> np.ndarray:
        """Cell area for continuous cells; 1.0 on the atom faces (mass, not density)."""
        we = np.where(ie == 0, 1.0, np.diff(self.ent_edges)[np.clip(ie - 1, 0, None)])
        wc = np.where(ic == 0, 1.0, np.diff(self.cos_edges)[np.clip(ic - 1, 0, None)])
        return we * wc

    # -- public API -------------------------------------------------------------
    def likelihood_ratio(self, ent, cos, mz) -> np.ndarray:
        """LR = P(scores | correct) / P(scores | incorrect). Dimensionless."""
        ie, ic, im = self._cells_raw(
            *(np.atleast_1d(np.asarray(v, float)) for v in (ent, cos, mz))
        )
        return self.lr_grid[im, ie, ic]

    def pdf(self, ent, cos, mz, cls: int) -> np.ndarray:
        """P(scores | cls). Density where both scores > 0, probability mass otherwise."""
        ie, ic, im = self._cells_raw(
            *(np.atleast_1d(np.asarray(v, float)) for v in (ent, cos, mz))
        )
        f0 = self.hist0[im, ie, ic] / self._widths(ie, ic)
        return f0 if cls == 0 else f0 * self.lr_grid[im, ie, ic]

    def band_prior(self, mz) -> np.ndarray:
        """Fitted per-row P(correct) in the m/z band each query falls into."""
        _, _, im = self._cells_raw(*(np.atleast_1d(np.asarray(v, float)) for v in (0.0, 0.0, mz)))
        return self.prior_by_band[im]

    def posterior(self, ent, cos, mz, prior=None) -> np.ndarray:
        """P(correct | scores) for one candidate considered on its own.

        This is the marginal answer: the chance that a candidate row drawn at random
        with these scores is the correct one. It does not know how many rivals the
        candidate has, nor that at most one of them can be correct -- for that use
        `posterior_over_candidates`, which conditions on the whole candidate list and
        will differ, often substantially, for features with many or few candidates.

        `prior` overrides the fitted per-row rate. Supply one whenever the target data
        is not distributed like the train fold: the ratio transfers across populations,
        the prior does not (train p_absent 0.645 against 0.774 on held-out data).
        """
        lr = self.likelihood_ratio(ent, cos, mz)
        p = self.band_prior(mz) if prior is None else np.full(lr.shape, float(prior))
        odds = lr * (p / (1.0 - p))
        return odds / (1.0 + odds)

    def is_atom(self, ent, cos) -> np.ndarray:
        """True where the returned pdf value is a probability mass rather than a density."""
        return (np.atleast_1d(np.asarray(ent, float)) <= 0) | (
            np.atleast_1d(np.asarray(cos, float)) <= 0
        )

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "ConditionalDensities":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# %%
def posterior_over_candidates(lr: np.ndarray, p_absent: float) -> np.ndarray:
    """Posterior that each candidate is correct, given at most one can be.

    With Z the identity of the correct candidate and f0 shared by every candidate,
    P(Z=i | scores) is proportional to P(Z=i) * LR_i and the f0 terms cancel -- which
    is why only the ratio, never the individual densities, reaches the MRF.

    Returns an array of length len(lr) + 1; the last entry is P(truth not in library).
    """
    lr = np.asarray(lr, dtype=float)
    w = np.concatenate([(1.0 - p_absent) / len(lr) * lr, [p_absent]])
    return w / w.sum()


# %%
def cluster_bootstrap_auc(
    df: pl.DataFrame, lr: np.ndarray, n_boot: int = 200, seed: int = 0
) -> tuple[float, float, float]:
    """AUC with a 95% CI, resampling whole molecules rather than rows.

    MassSpecGym carries many replicate spectra per molecule, so rows are far from
    independent: the test fold's correct Orbitrap rows come from 234 molecules with a
    Kish effective sample size near 38. A row-level interval understates the spread by
    more than an order of magnitude.
    """
    y = df["is_correct"].to_numpy()
    codes = df["inchikey_msg"].to_physical().rank("dense").to_numpy() - 1
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    groups = np.split(order, starts[1:])

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate(
            [groups[i] for i in rng.integers(0, len(groups), len(groups))]
        )
        yb = y[idx]
        if 0 < yb.sum() < len(yb):
            boots.append(roc_auc_score(yb, lr[idx]))
    return (
        roc_auc_score(y, lr),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


# %%
def _top1(scores: list, labels: list) -> float:
    return float(
        np.mean(
            [
                int(np.argmax(np.asarray(s)) == np.argmax(np.asarray(y)))
                for s, y in zip(scores, labels)
            ]
        )
    )


def report(
    name: str, d: ConditionalDensities, tr: pl.DataFrame, te: pl.DataFrame
) -> None:
    """Held-out evaluation, all intervals bootstrapped over molecules rather than rows.

    Pass the val fold while choosing settings and the test fold only to report the
    result. Reading both together, then adjusting, is selection on the test set -- and
    with a molecule-level effective sample size near 38 the noise is large enough to
    make that look like real signal.
    """
    lr = d.likelihood_ratio(
        te[ENT].to_numpy(), te[COS].to_numpy(), te["precursor_mz"].to_numpy()
    )
    t = te.with_columns(pl.Series("lr", lr))
    y = te["is_correct"].to_numpy()
    print(f"\n{'=' * 66}\n{name}\n{'=' * 66}")
    print(
        f"held-out rows={len(y):,}  correct={y.sum():,}  "
        f"distinct correct molecules={te.filter(pl.col('is_correct') == 1)['inchikey_msg'].n_unique():,}"
    )

    # 1. Ranking candidates within a feature -- the task the unary potential does.
    #    m/z is constant across a feature's candidates, so it cannot contribute here.
    g = (
        t.filter(pl.col("truth_present") == 1)
        .group_by("feature_id")
        .agg(pl.col("lr"), pl.col(ENT), pl.col(COS), pl.col("is_correct"))
    )
    lab = g["is_correct"].to_list()
    print(f"\nwithin-feature top-1 accuracy (truth present, n={g.height:,}):")
    print(f"  {ENT:22s} {_top1(g[ENT].to_list(), lab):.3f}")
    print(f"  {COS:22s} {_top1(g[COS].to_list(), lab):.3f}")
    print(f"  {'joint LR':22s} {_top1(g['lr'].to_list(), lab):.3f}")

    # 2. Deciding whether the truth is in the library at all -- where m/z pays off.
    p_absent = (
        1
        - tr.group_by("feature_id")
        .agg(pl.col("truth_present").first())["truth_present"]
        .mean()
    )
    gf = t.group_by("feature_id").agg(pl.col("lr"), pl.col("truth_present").first())
    p_none = np.array(
        [
            posterior_over_candidates(np.asarray(x), p_absent)[-1]
            for x in gf["lr"].to_list()
        ]
    )
    absent = 1 - gf["truth_present"].to_numpy()
    print(
        f"\nP(truth not in library):  train prior={p_absent:.3f}  held-out actual={absent.mean():.3f}"
    )
    print(f"  AUC(P(none) -> absent) = {roc_auc_score(absent, p_none):.3f}")
    print(
        f"  Brier = {np.mean((p_none - absent) ** 2):.4f}   "
        f"constant-prior baseline = {np.mean((p_absent - absent) ** 2):.4f}"
    )

    # 3. Pooled per-row AUC. Reported last and with a caveat: it mixes within-feature
    #    discrimination with the model's ability to tell easy features from hard ones,
    #    so it overstates what the unary potential contributes to ranking.
    auc, lo, hi = cluster_bootstrap_auc(te, lr)
    print(f"\npooled per-row AUC (between- and within-feature combined):")
    print(f"  {'joint LR':22s} {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    for s in (ENT, COS):
        a, l, h = cluster_bootstrap_auc(te, te[s].to_numpy().astype(float))
        print(f"  {s:22s} {a:.3f}  95% CI [{l:.3f}, {h:.3f}]")


def fit_all(
    path: str = "cfmid_scores.parquet",
    out_dir: str = ".",
    quiet: bool = False,
    eval_fold: str = "test",
    mz_cuts: tuple[float, ...] = MZ_CUTS,
):
    """Fit and persist one model per instrument. Returns {instrument: path}."""
    df = load_labeled(path).drop_nulls("precursor_mz")
    # [M+Na]+ features have zero positives by construction (ISDB covers [M+H]+ only),
    # so they carry no f1 and would only distort f0.
    df = df.filter(pl.col("adduct") == "[M+H]+")

    written = {}
    for inst in ["Orbitrap", "QTOF"]:
        sub = df.filter(pl.col("instrument") == inst)
        tr = sub.filter(pl.col("fold") == "train")
        d = ConditionalDensities(mz_cuts=mz_cuts).fit(tr)
        if not quiet:
            bands = " | ".join(
                f"{lo:.0f}-{hi:.0f}" if np.isfinite(hi) else f"{lo:.0f}+"
                for lo, hi in zip(d.mz_edges[:-1], d.mz_edges[1:])
            )
            report(
                f"{inst}   m/z bands: {bands}   evaluated on: {eval_fold}",
                d,
                tr,
                sub.filter(pl.col("fold") == eval_fold),
            )
        out = f"{out_dir.rstrip('/')}/model_{inst}.pkl"
        d.save(out)
        written[inst] = out
    return written


if __name__ == "__main__":
    for inst, p in fit_all().items():
        print(f"-> wrote {p}")

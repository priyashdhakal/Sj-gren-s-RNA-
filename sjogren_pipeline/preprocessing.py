"""Reusable preprocessing primitives (no I/O): normalization, filtering, probe
collapse, batch correction (limma-style removeBatchEffect and parametric ComBat)
and PCA-based diagnostics.  All matrices are *samples x features*.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
def log2_transform(expr: pd.DataFrame, offset: float = 1.0) -> pd.DataFrame:
    return np.log2(expr.clip(lower=0) + offset)


def quantile_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Classical quantile normalization across samples (rows), ties get average rank."""
    values = df.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("quantile normalization needs finite values")
    reference = np.sort(values, axis=1).mean(axis=0)
    grid = np.arange(values.shape[1])
    out = np.empty_like(values)
    for i in range(values.shape[0]):
        out[i] = np.interp(rankdata(values[i], method="average") - 1, grid, reference)
    return pd.DataFrame(out, index=df.index, columns=df.columns)


# --------------------------------------------------------------------------
# filtering and probe -> gene collapse
# --------------------------------------------------------------------------
def drop_control_probes(df: pd.DataFrame) -> pd.DataFrame:
    keep = ~df.columns.astype(str).str.startswith("AFFX-")
    return df.loc[:, keep]


def expression_threshold(df: pd.DataFrame, method: str = "gmm", quantile: float = 0.25,
                         seed: int = 42) -> Tuple[float, str]:
    """Return a log2 'expressed' threshold.

    gmm: two-component Gaussian mixture on probe means; threshold = point where the
    posterior of the high component is 0.5.  Falls back to a quantile if the
    components are not separated (< 1 log2 unit).
    """
    means = df.mean(axis=0).to_numpy().reshape(-1, 1)
    if method == "gmm":
        gm = GaussianMixture(2, random_state=seed).fit(means)
        lo, hi = np.argsort(gm.means_.ravel())
        if gm.means_[hi, 0] - gm.means_[lo, 0] >= 1.0:
            grid = np.linspace(gm.means_[lo, 0], gm.means_[hi, 0], 400).reshape(-1, 1)
            post_hi = gm.predict_proba(grid)[:, hi]
            thr = float(grid[np.argmax(post_hi >= 0.5), 0])
            return thr, "gmm"
    return float(np.quantile(means, quantile)), f"quantile_{quantile}"


def expression_filter(df: pd.DataFrame, min_samples: int, thr: float) -> pd.DataFrame:
    keep = (df > thr).sum(axis=0) >= min_samples
    return df.loc[:, keep]


def collapse_to_genes(df: pd.DataFrame, annotation: pd.DataFrame, multi_gene: str = "drop",
                      method: str = "max_mean") -> Tuple[list, pd.DataFrame]:
    """Pick one probe per gene (highest mean / IQR, *not* the most significant probe).

    Returns (selected_probe_ids, probe_map) where probe_map has gene, probe, n_probes.
    """
    ann = annotation[["ID", "Gene Symbol"]].rename(columns={"ID": "probe", "Gene Symbol": "gene"}).copy()
    ann["probe"] = ann["probe"].astype(str)
    ann["gene"] = ann["gene"].astype("string").str.strip()
    ann = ann[ann["probe"].isin(df.columns.astype(str))]
    ann = ann[ann["gene"].notna() & (ann["gene"] != "") & (ann["gene"] != "---")]
    multi = ann["gene"].str.contains("///", regex=False)
    if multi_gene == "first":
        ann.loc[multi, "gene"] = ann.loc[multi, "gene"].str.split("///").str[0].str.strip()
    else:
        ann = ann[~multi]
    cols = df.columns.astype(str)
    stat = df.mean(axis=0) if method == "max_mean" else (df.quantile(0.75) - df.quantile(0.25))
    stat.index = cols
    ann["score"] = ann["probe"].map(stat)
    n_probes = ann.groupby("gene")["probe"].transform("count")
    ann["n_probes"] = n_probes
    best = ann.loc[ann.groupby("gene")["score"].idxmax()].sort_values("gene").reset_index(drop=True)
    return best["probe"].tolist(), best[["gene", "probe", "n_probes", "score"]]


# --------------------------------------------------------------------------
# batch correction
# --------------------------------------------------------------------------
def _sum_to_zero(labels: Sequence) -> Tuple[np.ndarray, list]:
    lab = np.asarray(labels).astype(str)
    levels = sorted(set(lab))
    C = np.zeros((len(lab), len(levels) - 1))
    for j, lv in enumerate(levels[:-1]):
        C[lab == lv, j] = 1.0
    C[lab == levels[-1], :] = -1.0
    return C, levels


def remove_batch_effect(df: pd.DataFrame, batch: Sequence,
                        covariates: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """limma::removeBatchEffect: fit y = mu + covariates + batch(sum-to-zero), subtract batch part."""
    Y = df.to_numpy(float)
    C, _ = _sum_to_zero(batch)
    parts = [np.ones((Y.shape[0], 1))]
    if covariates is not None and covariates.shape[1] > 0:
        parts.append(pd.get_dummies(covariates, drop_first=True, dtype=float).to_numpy())
    parts.append(C)
    X = np.hstack(parts)
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    corrected = Y - C @ coef[-C.shape[1]:, :]
    return pd.DataFrame(corrected, index=df.index, columns=df.columns)


def combat(df: pd.DataFrame, batch: Sequence, covariates: Optional[pd.DataFrame] = None,
           mean_only: bool = False, max_iter: int = 200, tol: float = 1e-4) -> pd.DataFrame:
    """Parametric empirical-Bayes ComBat (Johnson et al. 2007; follows sva::ComBat)."""
    dat = df.to_numpy(float).T                      # genes x samples
    n_genes, n_samp = dat.shape
    batch = np.asarray(batch).astype(str)
    levels = sorted(set(batch))
    k = len(levels)
    idx = [np.where(batch == lv)[0] for lv in levels]
    n_b = np.array([len(i) for i in idx], dtype=float)

    batch_design = np.stack([(batch == lv).astype(float) for lv in levels], axis=1)
    if covariates is not None and covariates.shape[1] > 0:
        cov = pd.get_dummies(covariates, drop_first=True, dtype=float).to_numpy()
        design = np.hstack([batch_design, cov])
    else:
        cov, design = None, batch_design

    B_hat = np.linalg.solve(design.T @ design, design.T @ dat.T)      # (p x genes)
    grand_mean = (n_b / n_samp) @ B_hat[:k, :]                         # genes
    resid = dat - (design @ B_hat).T
    var_pooled = (resid ** 2).sum(axis=1) / n_samp
    var_pooled[var_pooled == 0] = np.median(var_pooled[var_pooled > 0])
    stand_mean = np.tile(grand_mean[:, None], (1, n_samp))
    if cov is not None:
        stand_mean = stand_mean + (cov @ B_hat[k:, :]).T
    s_data = (dat - stand_mean) / np.sqrt(var_pooled)[:, None]

    gamma_hat = np.linalg.solve(batch_design.T @ batch_design, batch_design.T @ s_data.T)   # k x genes
    if mean_only:
        delta_hat = np.ones((k, n_genes))
    else:
        delta_hat = np.stack([s_data[:, i].var(axis=1, ddof=1) for i in idx], axis=0)

    gamma_bar = gamma_hat.mean(axis=1)
    t2 = gamma_hat.var(axis=1, ddof=1)
    if mean_only:
        a_prior = b_prior = np.ones(k)
    else:
        m = delta_hat.mean(axis=1)
        s2 = delta_hat.var(axis=1, ddof=1)
        a_prior = (2 * s2 + m ** 2) / s2
        b_prior = (m * s2 + m ** 3) / s2

    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.ones_like(delta_hat)
    for i in range(k):
        n = n_b[i]
        g_old, d_old = gamma_hat[i].copy(), delta_hat[i].copy()
        for _ in range(max_iter):
            g_new = (t2[i] * n * gamma_hat[i] + d_old * gamma_bar[i]) / (t2[i] * n + d_old)
            if mean_only:
                d_new = np.ones(n_genes)
            else:
                sum2 = ((s_data[:, idx[i]] - g_new[:, None]) ** 2).sum(axis=1)
                d_new = (0.5 * sum2 + b_prior[i]) / (n / 2.0 + a_prior[i] - 1.0)
            change = max(np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-12)),
                         np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-12)))
            g_old, d_old = g_new, d_new
            if change < tol:
                break
        gamma_star[i], delta_star[i] = g_old, d_old

    out = s_data.copy()
    for i in range(k):
        out[:, idx[i]] = (s_data[:, idx[i]] - gamma_star[i][:, None]) / np.sqrt(delta_star[i])[:, None]
    out = out * np.sqrt(var_pooled)[:, None] + stand_mean
    return pd.DataFrame(out.T, index=df.index, columns=df.columns)


def batches_confounded(batch: Sequence, group: Sequence) -> bool:
    """True when some batch contains a single group only (batch removal would erase biology)."""
    tab = pd.crosstab(pd.Series(batch, name="b"), pd.Series(group, name="g"))
    return bool(((tab > 0).sum(axis=1) < 2).any())


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def anova_r2(y: np.ndarray, labels: Sequence) -> float:
    lab = pd.Series(np.asarray(labels).astype(str))
    y = pd.Series(np.asarray(y, float))
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    ss_b = sum(len(v) * (v.mean() - y.mean()) ** 2 for _, v in y.groupby(lab))
    return float(ss_b / ss_tot)


def gene_variance_fraction(df: pd.DataFrame, labels: Sequence) -> float:
    """Mean over genes of the one-way ANOVA R2 of a label (equal gene weight = what a
    standardised model sees).  Under the null it is about (levels - 1) / (n - 1)."""
    X = df.to_numpy(float)
    lab = np.asarray(labels).astype(str)
    mean = X.mean(axis=0)
    sst = ((X - mean) ** 2).sum(axis=0)
    ssb = np.zeros(X.shape[1])
    for level in set(lab):
        m = lab == level
        ssb += m.sum() * (X[m].mean(axis=0) - mean) ** 2
    ok = sst > 0
    return float(np.mean(ssb[ok] / sst[ok]))


def pca_diagnostics(df: pd.DataFrame, group: Sequence, batch: Sequence, top_var: int = 2000,
                    n_pcs: int = 5) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """PC table (variance, R2 with batch/group) and summary scores.

    summary: batch_R2_w / group_R2_w are variance-weighted over the top PCs;
    silhouette values are computed on the same PC space.
    """
    top = df.var(axis=0).nlargest(min(top_var, df.shape[1])).index
    Z = df[top] - df[top].mean(axis=0)
    n_pcs = min(n_pcs, Z.shape[0] - 1)
    pca = PCA(n_components=n_pcs, random_state=0)
    scores = pca.fit_transform(Z)
    rows = []
    for i in range(n_pcs):
        rows.append({"PC": i + 1, "var_explained": pca.explained_variance_ratio_[i],
                     "R2_batch": anova_r2(scores[:, i], batch), "R2_group": anova_r2(scores[:, i], group)})
    tab = pd.DataFrame(rows)
    w = tab["var_explained"] / tab["var_explained"].sum()
    summ = {"batch_R2_w": float((w * tab["R2_batch"]).sum()), "group_R2_w": float((w * tab["R2_group"]).sum()),
            "group_R2_max": float(tab["R2_group"].max()), "batch_R2_pc1": float(tab.loc[0, "R2_batch"]),
            "pc1_var": float(tab.loc[0, "var_explained"]),
            "gene_R2_batch": gene_variance_fraction(df, batch), "gene_R2_group": gene_variance_fraction(df, group)}
    for name, lab in (("group", group), ("batch", batch)):
        lab = np.asarray(lab).astype(str)
        counts = pd.Series(lab).value_counts()
        summ[f"silhouette_{name}"] = (float(silhouette_score(scores, lab))
                                      if len(counts) >= 2 and counts.min() >= 2 else float("nan"))
    return tab, summ, pd.DataFrame(scores, index=df.index, columns=[f"PC{i + 1}" for i in range(n_pcs)])


def batch_predictability(df: pd.DataFrame, batch: Sequence, k: int = 200, seed: int = 0) -> float:
    """3-fold CV accuracy of predicting the batch from expression (chance ~ majority share).
    High values mean batch is still readable from the features."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    lab = np.asarray(batch, dtype=object).astype(str)
    if pd.Series(lab).value_counts().min() < 3:
        return float("nan")
    pipe = Pipeline([("s", SelectKBest(f_classif, k=min(k, df.shape[1]))), ("sc", StandardScaler()),
                     ("m", LogisticRegression(C=0.1, max_iter=3000))])
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    return float(cross_val_score(pipe, df.to_numpy(float), lab, cv=cv, scoring="accuracy").mean())


def rle_iqr_within_strata(df: pd.DataFrame, batch: Sequence, group: Sequence) -> pd.Series:
    """Relative-log-expression spread per array: IQR of (gene - median over the arrays of the same
    batch x group stratum).  Using strata keeps biology (diagnosis) from looking like noise.
    Strata with < 3 arrays fall back to the whole batch."""
    batch = pd.Series(np.asarray(batch, dtype=object).astype(str), index=df.index)
    group = pd.Series(np.asarray(group, dtype=object).astype(str), index=df.index)
    out = pd.Series(index=df.index, dtype=float)
    for (b, g), ids in df.groupby([batch, group]).groups.items():
        ref = ids if len(ids) >= 3 else batch.index[batch == b]
        R = df.loc[ids] - df.loc[ref].median(axis=0)
        out.loc[ids] = (R.quantile(0.75, axis=1) - R.quantile(0.25, axis=1)).values
    return out


def split_noisy_subbatch(batch: Sequence, noisy: Sequence, min_size: int = 3) -> np.ndarray:
    """Split a batch into '<b>a' (regular) and '<b>b' (noisy) when both parts have >= min_size arrays."""
    batch = np.asarray(batch, dtype=object).astype(str)
    noisy = np.asarray(noisy, dtype=bool)
    out = batch.astype(object)          # object dtype: fixed-width '<U1' would truncate '1a' to '1'
    for b in sorted(set(batch)):
        in_b = batch == b
        if (in_b & noisy).sum() >= min_size and (in_b & ~noisy).sum() >= min_size:
            out[in_b & ~noisy] = b + "a"
            out[in_b & noisy] = b + "b"
    return out

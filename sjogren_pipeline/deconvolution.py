"""Reference-based deconvolution of bulk salivary-gland expression.

Signature: HPA salivary-gland single-cell reference (cell type x gene CPM). Markers per cell type are genes
expressed >= ``min_specificity`` x higher than in any other cell type (top ``markers_per_type`` by CPM).
Each sample is decomposed by non-negative least squares on the linear scale (2^log2 - 1), with every
marker gene scaled to equal weight; coefficients are normalised to proportions.

The immune fraction (sum over immune cell types) replaces the earlier single mean-marker score as the
infiltration covariate. Its accuracy is checked against histology (%CD45) in GSE173808 by stage 3.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from .external_data import IMMUNE, EPITHELIAL


def select_markers(ref: pd.DataFrame, genes: List[str], n_per_type: int, min_spec: float) -> Dict[str, List[str]]:
    r = ref.reindex([g for g in genes if g in ref.index])
    out = {}
    for ct in r.columns:
        other = r.drop(columns=ct).max(axis=1)
        ok = r[ct] >= min_spec * other.clip(lower=1.0)
        cand = r.loc[ok & (r[ct] >= 10), ct].sort_values(ascending=False)
        if len(cand) >= 5:
            out[ct] = cand.index[:n_per_type].tolist()
    return out


def deconvolve(expr_log2: pd.DataFrame, ref: pd.DataFrame, n_per_type: int = 50, min_spec: float = 3.0) -> pd.DataFrame:
    """expr_log2: samples x genes (log2 scale). Returns samples x cell-type proportions + compartment sums."""
    markers = select_markers(ref, list(expr_log2.columns), n_per_type, min_spec)
    genes = sorted({g for v in markers.values() for g in v})
    S = ref.loc[genes, list(markers)].to_numpy(float)
    B = (2.0 ** expr_log2[genes].to_numpy(float) - 1.0).clip(min=0).T          # genes x samples
    scale = S.mean(axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    S_s, B_s = S / scale, B / scale
    # per-sample gene scaling so array and RNA-seq intensities are on the reference's scale
    B_s = B_s / np.median(B_s, axis=0, keepdims=True).clip(min=1e-9) * np.median(S_s.sum(axis=1))
    P = np.zeros((B.shape[1], S.shape[1]))
    for j in range(B.shape[1]):
        coef, _ = nnls(S_s, B_s[:, j])
        P[j] = coef / coef.sum() if coef.sum() > 0 else coef
    out = pd.DataFrame(P, index=expr_log2.index, columns=list(markers))
    out["immune_fraction"] = out[[c for c in out.columns if c in IMMUNE]].sum(axis=1)
    out["epithelial_fraction"] = out[[c for c in out.columns if c in EPITHELIAL]].sum(axis=1)
    out.attrs["n_markers"] = {k: len(v) for k, v in markers.items()}
    return out

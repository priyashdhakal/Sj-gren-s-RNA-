"""Single-cell helpers for the salivary-gland reference GSE272409 (minor labial gland, 7 SjD vs 6 non-SjD sicca).

Used by stage 8 (independent of the ranking):
  * per-sample 10x matrices (``data/raw/scrna/<GSE>/GSM*_{matrix.mtx,features.tsv,barcodes.tsv}.gz``)
  * QC: >= ``min_genes`` detected genes and < ``max_mito`` mitochondrial fraction per cell
  * cell-type calls from HPA salivary-gland markers (mean log-CP10K of markers, z-scored over cells; argmax
    when the best z > 0.5, else "unassigned"), grouped into immune / epithelial / stromal compartments
  * per cell type: mean log-CP10K and % expressing cells for genes of interest
  * pseudobulk counts per sample x compartment for limma-voom (SjD vs sicca)
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread

from .deconvolution import select_markers
from .external_data import compartment_of


def sample_files(folder: Path) -> Dict[str, Tuple[Path, Path, Path]]:
    """GSM -> (matrix, features, barcodes)."""
    out = {}
    for mtx in sorted(folder.glob("*matrix.mtx*")):
        stem = mtx.name.split("_matrix")[0]
        feat = next(iter(folder.glob(f"{stem}_features.tsv*")), None) or next(iter(folder.glob(f"{stem}_genes.tsv*")), None)
        bar = next(iter(folder.glob(f"{stem}_barcodes.tsv*")), None)
        if feat is not None and bar is not None:
            out[stem.split("_")[0]] = (mtx, feat, bar)
    return out


def _read_tsv(path: Path) -> pd.DataFrame:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        return pd.read_csv(fh, sep="\t", header=None)


def load_sample(mtx: Path, feat: Path, min_genes: int = 500, max_mito: float = 0.2) -> Tuple[sp.csr_matrix, np.ndarray]:
    """Return (genes x cells counts after QC, gene symbols)."""
    X = sp.csr_matrix(mmread(str(mtx)))
    f = _read_tsv(feat)
    genes = f.iloc[:, 1 if f.shape[1] > 1 else 0].astype(str).to_numpy()
    if X.shape[0] != len(genes):
        X = X.T.tocsr()
    n_genes = np.asarray((X > 0).sum(axis=0)).ravel()
    lib = np.asarray(X.sum(axis=0)).ravel()
    mito = np.asarray(X[np.char.startswith(genes.astype(str), "MT-")].sum(axis=0)).ravel() / np.maximum(lib, 1)
    keep = (n_genes >= min_genes) & (mito < max_mito)
    return X[:, keep].tocsr(), genes


def lognorm(X: sp.csr_matrix) -> sp.csr_matrix:
    lib = np.asarray(X.sum(axis=0)).ravel()
    lib[lib == 0] = 1
    Y = X.multiply(1e4 / lib).tocsr()
    Y.data = np.log1p(Y.data)
    return Y


def annotate(Y: sp.csr_matrix, genes: np.ndarray, ref: pd.DataFrame, n_markers: int = 30, min_spec: float = 3.0) -> pd.Series:
    """Cell type per cell from HPA salivary-gland markers (Y: genes x cells, log-normalised)."""
    gidx = {g: i for i, g in enumerate(genes)}
    markers = select_markers(ref, list(gidx), n_markers, min_spec)
    scores = {}
    for ct, mk in markers.items():
        idx = [gidx[g] for g in mk if g in gidx]
        s = np.asarray(Y[idx].mean(axis=0)).ravel()
        scores[ct] = (s - s.mean()) / (s.std() or 1.0)
    S = pd.DataFrame(scores)
    return S.idxmax(axis=1).where(S.max(axis=1) > 0.5, "unassigned")


def pseudobulk(X: sp.csr_matrix, genes: np.ndarray, mask: np.ndarray) -> pd.Series:
    return pd.Series(np.asarray(X[:, mask].sum(axis=1)).ravel(), index=genes).groupby(level=0).sum()


def gene_celltype_table(Y: sp.csr_matrix, genes: np.ndarray, celltype: np.ndarray, targets: List[str]) -> pd.DataFrame:
    """Long table: gene, cell_type, mean log-CP10K, % expressing (pooled over the given cells)."""
    gidx = {g: i for i, g in enumerate(genes)}
    rows = []
    for ct in sorted(set(celltype)):
        m = celltype == ct
        if m.sum() < 20:
            continue
        sub = Y[:, np.where(m)[0]]
        for g in targets:
            if g not in gidx:
                continue
            v = sub[gidx[g]]
            rows.append({"gene": g, "cell_type": ct, "compartment": compartment_of(ct) if ct != "unassigned" else "unassigned",
                         "n_cells": int(m.sum()), "mean_logcp10k": float(v.sum() / m.sum()),
                         "pct_expressing": float(100.0 * v.getnnz() / m.sum())})
    return pd.DataFrame(rows)

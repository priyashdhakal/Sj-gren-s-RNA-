"""Pooled multi-cohort dataset, fold-wise feature processing (DeepMOLFT dataloader.py equivalent).

Per cohort: case-vs-control samples of the main pipeline's stage-2 gene matrices, genes z-scored within
the cohort (label-free; removes platform and scale differences). Per outer fold, fitted on the TRAINING
cohorts only: top-variable pre-filter, branch assignment, correlation elimination (|r| > 0.85, DeepMOLFT
rule) and StandardScaler.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler


@dataclass
class Pooled:
    X: pd.DataFrame          # samples x genes, z-scored within cohort
    y: np.ndarray            # 1 = Sjogren's, 0 = control
    cohort: np.ndarray
    group: np.ndarray        # patient id (paired glands share one)
    cohort_var: pd.DataFrame  # cohort x gene, within-cohort variance / cohort median variance
    branch: pd.Series        # gene -> immune | gland | broad


def load_pooled(cfg: dict, root: Path) -> Pooled:
    main_cfg = yaml.safe_load(open(root / cfg["inputs"]["main_config"], encoding="utf-8"))
    s2 = root / cfg["inputs"]["stage2"]
    mats, ys, cohorts, groups, var = [], [], [], [], {}
    for acc in cfg["data"]["cohorts"]:
        d = main_cfg["datasets"][acc]
        meta = pd.read_csv(s2 / f"{acc}_samples_final.csv", index_col=0)
        meta = meta[meta["group"].isin([d["case"], d["control"]])]
        ex = pd.read_parquet(s2 / f"{acc}_gene_expr.parquet").loc[meta.index]
        v = ex.var(axis=0)
        var[acc] = v / v.median()
        if cfg["data"]["within_cohort_zscore"]:
            ex = (ex - ex.mean(axis=0)) / ex.std(axis=0, ddof=0).replace(0, np.nan)
        mats.append(ex)
        ys.append((meta["group"] == d["case"]).astype(int).to_numpy())
        cohorts.append(np.full(len(meta), acc))
        pid = meta["char_sjnumber"].astype(str) if "char_sjnumber" in meta else pd.Series(meta.index, index=meta.index)
        groups.append((acc + ":" + pid).to_numpy())
    counts = pd.Series([g for m in mats for g in m.columns]).value_counts()
    genes = sorted(counts.index[counts >= cfg["data"]["min_cohorts_per_gene"]])
    X = pd.concat([m.reindex(columns=genes) for m in mats]).fillna(0.0)
    return Pooled(X=X, y=np.concatenate(ys), cohort=np.concatenate(cohorts), group=np.concatenate(groups),
                  cohort_var=pd.DataFrame(var).T.reindex(columns=genes), branch=assign_branches(root, cfg, genes))


def assign_branches(root: Path, cfg: dict, genes: List[str]) -> pd.Series:
    """Cell of origin from the HPA salivary-gland reference (main pipeline's external_data module)."""
    sys.path.insert(0, str((root / cfg["inputs"]["pipeline_root"]).resolve()))
    from sjogren_pipeline.context import load_context
    from sjogren_pipeline.external_data import celltype_profile, hpa_salivary_reference

    ctx = load_context(str(root / cfg["inputs"]["main_config"]))
    prof = celltype_profile(hpa_salivary_reference(ctx), genes)
    comp = prof["top_compartment"].fillna("broad")
    return comp.map(lambda c: "immune" if c == "immune" else ("gland" if c in ("epithelial", "stromal") else "broad"))


def correlated_to_drop(X: np.ndarray, threshold: float) -> np.ndarray:
    """DeepMOLFT correlation(): for i, for j < i, |r_ij| > thr -> drop column i (vectorised)."""
    if X.shape[1] < 2:
        return np.zeros(X.shape[1], dtype=bool)
    r = np.abs(np.corrcoef(X, rowvar=False))
    r = np.nan_to_num(r)
    lower = np.tril(r, k=-1)
    return (lower > threshold).any(axis=1)


@dataclass
class FoldData:
    branches: List[str]
    genes: Dict[str, List[str]]
    train: Dict[str, np.ndarray]
    test: Dict[str, np.ndarray]


def build_fold(p: Pooled, cfg: dict, train_idx: np.ndarray, test_idx: np.ndarray) -> FoldData:
    train_cohorts = sorted(set(p.cohort[train_idx]))
    mv = p.cohort_var.loc[train_cohorts].mean(axis=0)
    top = mv.nlargest(cfg["data"]["top_variable_genes"]).index
    genes, tr, te = {}, {}, {}
    for b in cfg["branches"]:
        g = [x for x in top if p.branch.get(x) == b]
        Xtr = p.X.iloc[train_idx][g].to_numpy(float)
        drop = correlated_to_drop(Xtr, cfg["data"]["correlation_threshold"])
        g = [x for x, d in zip(g, drop) if not d]
        sc = StandardScaler().fit(p.X.iloc[train_idx][g].to_numpy(float))
        genes[b] = g
        tr[b] = sc.transform(p.X.iloc[train_idx][g].to_numpy(float)).astype(np.float32)
        te[b] = sc.transform(p.X.iloc[test_idx][g].to_numpy(float)).astype(np.float32)
    return FoldData(branches=list(cfg["branches"]), genes=genes, train=tr, test=te)


def subset(fd: FoldData, idx: np.ndarray, which: str = "train") -> Dict[str, np.ndarray]:
    src = fd.train if which == "train" else fd.test
    return {b: src[b][idx] for b in fd.branches}

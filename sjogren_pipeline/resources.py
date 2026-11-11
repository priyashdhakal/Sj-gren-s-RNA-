"""Loaders for optional external resources (see data/external/README.md)."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .context import Context


def load_markers(ctx: Context) -> pd.DataFrame:
    path = ctx.external / ctx.cfg["de"]["infiltration_markers_file"]
    if not path.exists():
        raise FileNotFoundError(f"marker panel not found: {path}")
    return pd.read_csv(path)


def marker_sets(ctx: Context, lineage: Optional[str] = None) -> Dict[str, List[str]]:
    m = load_markers(ctx)
    if lineage is not None:
        m = m[m["lineage"] == lineage]
    return {k: sorted(v["gene"].unique()) for k, v in m.groupby("cell_type")}


def zscore_columns(df: pd.DataFrame) -> pd.DataFrame:
    sd = df.std(axis=0, ddof=0).replace(0, np.nan)
    return ((df - df.mean(axis=0)) / sd).fillna(0.0)


def infiltration_score(ctx: Context, expr: pd.DataFrame) -> pd.Series:
    """Mean z-score of immune-lineage marker genes per sample (expr: samples x genes).

    Used as a covariate to separate infiltrate-driven expression from gland-intrinsic changes.
    """
    genes = sorted(set(load_markers(ctx).query("lineage == 'immune'")["gene"]) & set(expr.columns))
    if len(genes) < 5:
        raise ValueError("fewer than 5 immune marker genes found in the expression matrix")
    return zscore_columns(expr[genes]).mean(axis=1).rename("infiltration_score")

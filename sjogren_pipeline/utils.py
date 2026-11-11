"""Small shared helpers (I/O, seeds, multiple testing)."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is optional for stages 1-4
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:  # pragma: no cover
        pass


def write_table(df: pd.DataFrame, path: PathLike, index: bool = True) -> Path:
    """Write parquet / csv / csv.gz depending on the suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        out = df.copy()
        out.columns = out.columns.astype(str)
        out.to_parquet(path, index=index)
    else:
        df.to_csv(path, index=index)
    return path


def read_table(path: PathLike, index_col: Any = 0) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=index_col)


def dump_json(obj: Any, path: PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: PathLike) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not serialisable: {type(o)}")


def bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN-safe, same as R p.adjust('BH'))."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.minimum(ranked, 1.0)
    out[ok] = adj
    return out


def find_rscript(setting: str = "auto") -> str:
    """Locate Rscript: explicit path, PATH, then the newest C:/Program Files/R installation."""
    import glob
    import shutil

    if setting and setting != "auto":
        return setting
    hit = shutil.which("Rscript")
    if hit:
        return hit
    cands = sorted(glob.glob("C:/Program Files/R/R-*/bin/x64/Rscript.exe") + glob.glob("/usr/lib/R/bin/Rscript"))
    if not cands:
        raise FileNotFoundError("Rscript not found; set project.rscript in config.yaml")
    return cands[-1]


def run_r(script: PathLike, args: list, rscript: str = "auto", log=None) -> str:
    """Run an R script and return stdout; raise with stderr on failure."""
    import subprocess

    cmd = [find_rscript(rscript), "--vanilla", str(script)] + [str(a) for a in args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if log is not None and res.stdout.strip():
        for line in res.stdout.strip().splitlines():
            log.info("  [R] %s", line)
    if res.returncode != 0:
        raise RuntimeError(f"R failed ({Path(script).name}):\n{res.stderr[-3000:]}")
    return res.stdout


def rank_normalize(x: pd.Series, ascending: bool = True) -> pd.Series:
    """Map values to (0, 1] by average rank (higher = better when ascending)."""
    r = x.rank(method="average", ascending=ascending, na_option="keep")
    return r / r.notna().sum()

"""Stage 1: download expression data and clinical metadata from GEO.

Every cohort's GSM metadata comes from GEOparse (a local ``*_family.soft.gz`` in ``data/raw`` is reused).
Expression by dataset type:

* ``microarray_cel``     ``<GSE>_RAW.tar`` is downloaded and extracted to ``data/raw/cel/<GSE>``, then
                         RMA-normalised in R (``R/rma_cel.R``; cached as ``data/raw/cel/<GSE>_rma.csv.gz``)
* ``microarray_matrix``  VALUE columns of the series matrix (as deposited)
* ``rnaseq_counts`` /    supplementary count / TPM matrix (``data/raw/counts``); columns are matched to GSMs
  ``rnaseq_tpm``         by title, title suffix or title prefix; Ensembl IDs mapped to symbols with HPA
* ``scrna_10x``          ``<GSE>_RAW.tar`` extracted to ``data/raw/scrna/<GSE>``

Sample labels (group / batch / severity) come from the regex rules in ``config.yaml``.

Outputs (``outputs/stage01_download``):
    <GSE>_expression_raw.parquet   samples x features (probes or genes)
    <GSE>_samples.csv              tidy clinical table incl. group / batch and parsed characteristics
    <GSE>_platform.csv.gz          probe annotation (arrays)
    download_summary.csv
"""
from __future__ import annotations

import logging
import re
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .context import Context
from .utils import run_r, write_table

STAGE = "stage01_download"
GEO_SUPPL = "https://ftp.ncbi.nlm.nih.gov/geo/series/{stub}nnn/{acc}/suppl/{name}"
R_DIR = Path(__file__).resolve().parents[1] / "R"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _parse_characteristics(chars: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in chars:
        if ":" in item:
            k, v = item.split(":", 1)
            out[k.strip().lower().replace(" ", "_")] = v.strip()
    return out


def sample_table(gse) -> pd.DataFrame:
    rows = []
    for gsm_id, gsm in gse.gsms.items():
        chars = gsm.metadata.get("characteristics_ch1", [])
        row = {
            "gsm": gsm_id,
            "title": "; ".join(gsm.metadata.get("title", [])),
            "source": "; ".join(gsm.metadata.get("source_name_ch1", [])),
            "characteristics": "; ".join(chars),
        }
        row.update({f"char_{k}": v for k, v in _parse_characteristics(chars).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def assign_labels(meta: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """Apply regex rules from the config; fail loudly if a sample is unlabeled."""
    meta = meta.copy()

    def first_match(row) -> str:
        for rule in rules["group_rules"]:
            if re.search(rule["pattern"], str(row[rule["field"]]), flags=re.I):
                return rule["label"]
        return "Unknown"

    meta["group"] = meta.apply(first_match, axis=1)
    if (meta["group"] == "Unknown").any():
        bad = meta.loc[meta["group"] == "Unknown", "gsm"].tolist()
        raise ValueError(f"unlabeled samples (extend group_rules): {bad}")

    for key in ("batch", "severity"):
        spec = rules.get(key)
        if spec:
            ext = meta[spec["field"]].astype(str).str.extract(spec["pattern"], flags=re.I, expand=False)
            meta[key] = ext.str.lower() if key == "severity" else ext
    if "batch" in meta and meta["batch"].isna().any():
        raise ValueError("batch could not be parsed for some samples")
    if "batch" not in meta:
        meta["batch"] = "1"
    return meta


def expression_matrix(gse) -> pd.DataFrame:
    cols = {}
    for gsm_id, gsm in gse.gsms.items():
        t = gsm.table
        if t is None or not {"ID_REF", "VALUE"}.issubset(t.columns):
            raise ValueError(f"{gsm_id} has no VALUE table (series may only ship raw files)")
        v = pd.to_numeric(t.set_index("ID_REF")["VALUE"], errors="coerce")
        cols[gsm_id] = v[~v.index.duplicated(keep="first")]
    return pd.DataFrame(cols).T


def platform_table(gse) -> pd.DataFrame:
    gpl = next(iter(gse.gpls.values()))
    keep = [c for c in gpl.table.columns if c in (
        "ID", "Gene Symbol", "GENE_SYMBOL", "GENE_NAME", "Gene Title", "ENTREZ_GENE_ID", "GB_ACC", "Representative Public ID",
        "Gene Ontology Biological Process", "Gene Ontology Cellular Component",
        "Gene Ontology Molecular Function", "Annotation Date", "CONTROL_TYPE")]
    return gpl.table[keep].copy()


def _download(url: str, dest: Path, log: logging.Logger) -> bool:
    url = url.replace("ftp://", "https://")
    try:
        log.info("  downloading %s", url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)  # nosec - fixed GEO host
        return True
    except Exception as exc:  # network blocked / file missing
        log.warning("  could not download %s (%s)", url.split("/")[-1], exc.__class__.__name__)
        return False


def _suppl_url(acc: str, name: str) -> str:
    return GEO_SUPPL.format(stub=acc[:-3], acc=acc, name=name)


def _fetch_tar(ctx: Context, acc: str, folder: Path) -> None:
    """Download and extract <GSE>_RAW.tar into folder (skipped when folder already has files)."""
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        return
    tar = folder.parent / f"{acc}_RAW.tar"
    if not tar.exists() and not _download(_suppl_url(acc, tar.name), tar, ctx.log):
        raise FileNotFoundError(f"{tar.name} missing; place it in {folder.parent}")
    with tarfile.open(tar) as t:
        t.extractall(folder)


def ensembl_to_symbol(ctx: Context) -> Dict[str, str]:
    """Ensembl gene ID -> HGNC symbol from the HPA single-cell table (downloaded on first use)."""
    path = ctx.external / "hpa" / "rna_single_cell_type.tsv.zip"
    if not path.exists():
        _download("https://www.proteinatlas.org/download/tsv/rna_single_cell_type.tsv.zip", path, ctx.log)
    with zipfile.ZipFile(path) as z:
        with z.open(z.namelist()[0]) as fh:
            m = pd.read_csv(fh, sep="\t", usecols=["Gene", "Gene name"]).drop_duplicates()
    return dict(zip(m["Gene"], m["Gene name"]))


def _match_columns(cols: List[str], meta: pd.DataFrame, key: str) -> Dict[str, str]:
    """count-matrix column -> GSM."""
    titles = dict(zip(meta["title"].astype(str), meta["gsm"]))
    out = {}
    for c in cols:
        if key == "title" and c in titles:
            out[c] = titles[c]
        elif key == "title_suffix":
            hit = [g for t, g in titles.items() if t.endswith(c)]
            if len(hit) == 1:
                out[c] = hit[0]
        elif key == "title_prefix":
            p = c.split("_")[0]
            hit = [g for t, g in titles.items() if t.split("_")[0] == p]
            if len(hit) == 1:
                out[c] = hit[0]
    return out


def read_count_table(ctx: Context, acc: str, dcfg: dict, meta: pd.DataFrame) -> pd.DataFrame:
    """Supplementary counts / TPM -> samples x gene symbols (duplicated symbols summed)."""
    path = ctx.raw / "counts" / dcfg["counts_file"]
    if not path.exists() and not _download(_suppl_url(acc, dcfg["counts_file"]), path, ctx.log):
        raise FileNotFoundError(f"{path} missing")
    df = pd.read_csv(path)
    gid = dcfg["gene_id"]
    if gid == "symbol_column":
        df = df[df.iloc[:, 0].astype(str) != "Id"]             # second header row
        genes = df["Symbol"].astype(str)
        df = df.drop(columns=[c for c in df.columns[:3]])
    else:
        genes = df.iloc[:, 0].astype(str)
        df = df.iloc[:, 1:]
        if gid == "ensembl":
            mp = ensembl_to_symbol(ctx)
            genes = genes.str.split(".").str[0].map(mp)
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed") and str(c).strip()]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index = genes.to_numpy()
    df = df[df.index.notna() & (df.index.astype(str) != "nan")]
    df = df.groupby(level=0).sum()
    mp = _match_columns(list(df.columns), meta, dcfg["sample_key"])
    missing = set(meta["gsm"]) - set(mp.values())
    if missing:
        raise ValueError(f"{acc}: {len(missing)} GSMs not found in {path.name} (e.g. {sorted(missing)[:3]})")
    df = df[list(mp)].rename(columns=mp)
    return df.T


# --------------------------------------------------------------------------
# per-dataset handler
# --------------------------------------------------------------------------
def _dataset(ctx: Context, acc: str, dcfg: dict, out: Path) -> dict:
    import GEOparse

    logging.getLogger("GEOparse").setLevel(logging.WARNING)
    ctx.log.info("[%s] %s: loading series metadata", acc, dcfg["type"])
    gse = GEOparse.get_GEO(geo=acc, destdir=str(ctx.raw), silent=True, annotate_gpl=False)
    meta = assign_labels(sample_table(gse), dcfg["labels"])
    meta.to_csv(out / f"{acc}_samples.csv", index=False)
    kind = dcfg["type"]

    if kind == "scrna_10x":
        _fetch_tar(ctx, acc, ctx.raw / "scrna" / acc)
        n = len(list((ctx.raw / "scrna" / acc).glob("*matrix.mtx*")))
        ctx.log.info("[%s] single-cell reference: %d samples, %d matrices", acc, len(meta), n)
        return {"dataset": acc, "type": kind, "samples": len(meta), "features": float("nan"),
                "groups": str(meta["group"].value_counts().to_dict()), "status": "ok" if n else "no matrices"}

    if kind == "microarray_cel":
        cel_dir = ctx.raw / "cel" / acc
        _fetch_tar(ctx, acc, cel_dir)
        rma = ctx.raw / "cel" / f"{acc}_rma.csv.gz"
        if not rma.exists():
            ctx.log.info("[%s] RMA on %d CEL files (R/affy)", acc, len(list(cel_dir.glob("*"))))
            run_r(R_DIR / "rma_cel.R", [cel_dir, rma], ctx.cfg["project"].get("rscript", "auto"), ctx.log)
        expr = pd.read_csv(rma, index_col=0).T
        expr = expr.loc[meta["gsm"]]
        platform_table(gse).to_csv(out / f"{acc}_platform.csv.gz", index=False)
    elif kind == "microarray_matrix":
        expr = expression_matrix(gse).loc[meta["gsm"]]
        platform_table(gse).to_csv(out / f"{acc}_platform.csv.gz", index=False)
    elif kind in ("rnaseq_counts", "rnaseq_tpm"):
        expr = read_count_table(ctx, acc, dcfg, meta).loc[meta["gsm"]]
    else:
        raise ValueError(f"unknown dataset type {kind}")

    write_table(expr, out / f"{acc}_expression_raw.parquet")
    ctx.log.info("[%s] %d samples x %d features | groups: %s | batches: %s", acc, expr.shape[0], expr.shape[1],
                 meta["group"].value_counts().to_dict(), meta["batch"].value_counts().sort_index().to_dict())
    return {"dataset": acc, "type": kind, "samples": expr.shape[0], "features": expr.shape[1],
            "groups": str(meta["group"].value_counts().to_dict()), "status": "ok"}


def run(ctx: Context) -> None:
    out = ctx.stage_dir(STAGE)
    summary = []
    for acc, dcfg in ctx.datasets():
        try:
            summary.append(_dataset(ctx, acc, dcfg, out))
        except Exception as exc:  # keep going so one failure does not block other cohorts
            ctx.log.error("[%s] failed: %s", acc, exc)
            summary.append({"dataset": acc, "type": dcfg["type"], "status": f"FAILED: {exc}"})
    pd.DataFrame(summary).to_csv(out / "download_summary.csv", index=False)
    ctx.log.info("stage 1 done -> %s", out)

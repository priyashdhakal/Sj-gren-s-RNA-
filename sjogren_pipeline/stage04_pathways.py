"""Stage 4: pathway analysis -- over-representation (ORA) and ranked GSEA.

Gene-set libraries (cache -> online -> offline fallback):
  * ``GO_BP_GPL570``    built offline from the GPL570 annotation (always available)
  * ``CellTypeMarkers`` the curated marker panel (cell-type enrichment)
  * Enrichr libraries listed in ``pathways.online_libraries`` (GO 2023, Reactome 2022, KEGG 2021,
    MSigDB Hallmark): downloaded with gseapy and cached as ``data/external/genesets/<name>.gmt``;
    skipped with a warning when the network is unavailable.

ORA uses a hypergeometric test against the *tested* gene universe, separately for up-regulated,
down-regulated and all significant genes.  GSEA (``gseapy.prerank``) ranks genes by the moderated
meta z statistic.  Both use the stage-3 random-effects meta-analysis (primary and deconvolution-adjusted).
Significant terms are collapsed into non-redundant representatives (Jaccard overlap).
"""
from __future__ import annotations

import socket
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .context import Context
from .resources import marker_sets
from .utils import bh_adjust

STAGE = "stage04_pathways"
S1, S3 = "stage01_download", "stage03_diffexpr"


# --------------------------------------------------------------------------
# gene-set libraries
# --------------------------------------------------------------------------
def build_go_bp(platform: pd.DataFrame, universe: set) -> Dict[str, List[str]]:
    """GO biological-process sets from the array annotation (single-gene probes only)."""
    p = platform[["Gene Symbol", "Gene Ontology Biological Process"]].dropna()
    p = p[~p["Gene Symbol"].str.contains("///", regex=False)]
    sets: Dict[str, set] = {}
    for gene, go in zip(p["Gene Symbol"], p["Gene Ontology Biological Process"]):
        if gene not in universe:
            continue
        for item in go.split("///"):
            parts = [x.strip() for x in item.split("//")]
            if len(parts) >= 2 and parts[0].isdigit():
                sets.setdefault(f"{parts[1]} (GO:{parts[0]})", set()).add(gene)
    return {k: sorted(v) for k, v in sets.items()}


def read_gmt(path: Path) -> Dict[str, List[str]]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) > 2:
                out[f[0]] = [g for g in f[2:] if g]
    return out


def write_gmt(sets: Dict[str, List[str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for k, v in sets.items():
            fh.write(k + "\tNA\t" + "\t".join(v) + "\n")


def load_libraries(ctx: Context, platform: pd.DataFrame, universe: set) -> Dict[str, Dict[str, List[str]]]:
    libs: Dict[str, Dict[str, List[str]]] = {"GO_BP_GPL570": build_go_bp(platform, universe),
                                              "CellTypeMarkers": marker_sets(ctx)}
    gdir = ctx.external / "genesets"
    gdir.mkdir(parents=True, exist_ok=True)
    socket.setdefaulttimeout(20)
    online_ok = True                      # flips to False after the first connection failure
    for name in ctx.cfg["pathways"]["online_libraries"]:
        gmt = gdir / f"{name}.gmt"
        if gmt.exists():
            libs[name] = read_gmt(gmt)
            continue
        if not online_ok:
            ctx.log.warning("  library %s not cached and network unavailable -> skipped", name)
            continue
        try:
            import gseapy

            lib = gseapy.get_library(name=name, organism="Human")
            if not lib:
                raise RuntimeError("empty library")
            write_gmt(lib, gmt)
            libs[name] = lib
            ctx.log.info("  downloaded %s (%d sets)", name, len(lib))
        except Exception as exc:
            online_ok = online_ok and not any(w in exc.__class__.__name__ for w in ("Proxy", "Connection", "Timeout", "URLError"))
            ctx.log.warning("  library %s unavailable (%s) -> skipped; run once with internet to cache it",
                            name, exc.__class__.__name__)
    for name, lib in libs.items():
        ctx.log.info("  library %-18s %5d sets", name, len(lib))
    return libs


# --------------------------------------------------------------------------
# ORA / GSEA
# --------------------------------------------------------------------------
def ora(genes: List[str], universe: set, lib: Dict[str, List[str]], min_size: int, max_size: int) -> pd.DataFrame:
    genes = set(genes) & universe
    M, N = len(universe), len(genes)
    rows = []
    for term, members in lib.items():
        mem = set(members) & universe
        n = len(mem)
        if n < min_size or n > max_size:
            continue
        k = len(genes & mem)
        if k == 0:
            continue
        p = stats.hypergeom.sf(k - 1, M, n, N)
        odds = ((k + 0.5) / (N - k + 0.5)) / ((n - k + 0.5) / (M - n - N + k + 0.5))
        rows.append({"term": term, "size": n, "overlap": k, "list_size": N, "odds_ratio": odds, "P": p,
                     "genes": ";".join(sorted(genes & mem))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["FDR"] = bh_adjust(df["P"].to_numpy())
    return df.sort_values("P").reset_index(drop=True)


def collapse_redundant(df: pd.DataFrame, lib: Dict[str, List[str]], universe: set, jaccard: float,
                       fdr: float) -> pd.DataFrame:
    """Greedy representatives: the best-ranked term of each family of overlapping significant terms."""
    df = df.copy()
    df["representative"] = ""
    kept: List[str] = []
    for i, r in df[df["FDR"] < fdr].iterrows():
        a = set(lib[r["term"]]) & universe
        rep = None
        for k in kept:
            b = set(lib[k]) & universe
            if len(a & b) / max(len(a | b), 1) >= jaccard:
                rep = k
                break
        if rep is None:
            kept.append(r["term"])
            df.at[i, "representative"] = r["term"]
        else:
            df.at[i, "representative"] = rep
    return df


def run_gsea(rank: pd.Series, lib: Dict[str, List[str]], min_size: int, max_size: int, nperm: int, seed: int,
             threads: int) -> pd.DataFrame:
    import gseapy

    pre = gseapy.prerank(rnk=rank.sort_values(ascending=False), gene_sets=lib, min_size=min_size,
                         max_size=max_size, permutation_num=nperm, seed=seed, threads=threads, no_plot=True,
                         verbose=False)
    res = pre.res2d.copy()
    res = res.rename(columns={"Term": "term", "NES": "NES", "NOM p-val": "P", "FDR q-val": "FDR",
                              "Lead_genes": "leading_edge", "Gene %": "gene_pct"})
    for c in ("ES", "NES", "P", "FDR"):
        res[c] = pd.to_numeric(res[c], errors="coerce")
    return res.sort_values("FDR").reset_index(drop=True)


def _barplot(df: pd.DataFrame, path, title: str, n: int = 15) -> None:
    d = df[df["representative"] == df["term"]].head(n) if "representative" in df else df.head(n)
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.2))
    ax.barh(range(len(d)), -np.log10(d["FDR"].clip(lower=1e-300)), color="#4c72b0")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([t[:58] for t in d["term"]], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("-log10 FDR")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(ctx: Context) -> None:
    cfg = ctx.cfg["pathways"]
    out, s1, s3 = ctx.stage_dir(STAGE), ctx.stage_dir(S1), ctx.stage_dir(S3)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    an, de = ctx.cfg["analysis"], ctx.cfg["de"]
    d_acc = an["discovery_dataset"]
    platform = pd.read_csv(s1 / f"{d_acc}_platform.csv.gz", dtype=str)

    # random-effects meta-analysis across all gland cohorts (stage 3): primary and deconvolution-adjusted
    allg = pd.read_csv(s3 / "de_all_genes.csv", index_col=0)
    adj = allg.dropna(subset=["adj_meta_logFC"])
    models = {"primary": pd.DataFrame({"gene": allg.index, "logFC": allg["meta_logFC"].to_numpy(),
                                       "adj.P.Val": allg["meta_FDR"].to_numpy(), "t": allg["meta_z"].to_numpy()}),
              "infiltration_adjusted": pd.DataFrame({"gene": adj.index, "logFC": adj["adj_meta_logFC"].to_numpy(),
                                                     "adj.P.Val": adj["adj_meta_FDR"].to_numpy(),
                                                     "t": (adj["adj_meta_logFC"] / adj["adj_meta_SE"]).to_numpy()})}
    universe = set(models["primary"]["gene"])
    libs = load_libraries(ctx, platform, universe)

    ora_all, gsea_all = [], []
    for mname, tab in models.items():
        thr = de["relaxed_abs_lfc"]
        lists = {k: tab[(tab["adj.P.Val"] < cfg["fdr"]) & ((tab["logFC"] >= thr) if k == "up" else (tab["logFC"] <= -thr))]["gene"].tolist()
                 for k in ("up", "down")}
        lists["all"] = lists["up"] + lists["down"]
        ctx.log.info("[%s] list sizes: up %d | down %d", mname, len(lists["up"]), len(lists["down"]))
        for lname, genes in lists.items():
            for libname, lib in libs.items():
                ms = 5 if libname == "CellTypeMarkers" else cfg["min_size"]
                res = ora(genes, universe, lib, ms, cfg["max_size"])
                if res.empty:
                    continue
                res = collapse_redundant(res, lib, universe, cfg["redundancy_jaccard"], cfg["fdr"])
                res.insert(0, "list", lname)
                res.insert(0, "library", libname)
                res.insert(0, "model", mname)
                ora_all.append(res)
        rank = tab.set_index("gene")["t"]
        for libname, lib in libs.items():
            ms = 5 if libname == "CellTypeMarkers" else cfg["min_size"]
            try:
                g = run_gsea(rank, lib, ms, cfg["max_size"], cfg["gsea_permutations"], ctx.seed, ctx.n_jobs)
            except Exception as exc:
                ctx.log.warning("  GSEA failed for %s/%s: %s", mname, libname, exc)
                continue
            g = collapse_redundant(g.assign(P=g["P"]), lib, universe, cfg["redundancy_jaccard"], cfg["fdr"]) \
                if "FDR" in g else g
            g.insert(0, "library", libname)
            g.insert(0, "model", mname)
            gsea_all.append(g)
            ctx.log.info("  GSEA %-22s %-18s %3d sets FDR<%.2f (%d up, %d down)", mname, libname,
                         int((g["FDR"] < cfg["fdr"]).sum()), cfg["fdr"],
                         int(((g["FDR"] < cfg["fdr"]) & (g["NES"] > 0)).sum()),
                         int(((g["FDR"] < cfg["fdr"]) & (g["NES"] < 0)).sum()))

    ora_df = pd.concat(ora_all, ignore_index=True)
    ora_df.to_csv(out / "ora_results.csv", index=False)
    gsea_df = pd.concat(gsea_all, ignore_index=True)
    gsea_df.to_csv(out / "gsea_results.csv", index=False)
    sig = ora_df[ora_df["FDR"] < cfg["fdr"]]
    ctx.log.info("ORA: %d significant term hits (%d representatives)", len(sig),
                 int((sig["representative"] == sig["term"]).sum()))
    for (m, lib, lst), d in ora_df.groupby(["model", "library", "list"]):
        if lst == "all":
            _barplot(d, fig_dir / f"ORA_{m}_{lib}.png", f"ORA ({lst} DE genes) | {m} | {lib}")
    for (m, lib), d in gsea_df.groupby(["model", "library"]):
        _barplot(d[d["FDR"] < cfg["fdr"]], fig_dir / f"GSEA_{m}_{lib}.png", f"GSEA | {m} | {lib}")
    ctx.log.info("stage 4 done -> %s", out)

"""Stage 3: differential expression in every gland cohort + random-effects meta-analysis.

Per cohort (R limma via ``R/limma_jobs.R``; batch / tissue are covariates, nothing is batch-corrected):
  arrays & TPM   lmFit -> eBayes(trend = TRUE, robust = TRUE)  (arrayWeights optional: de.array_weights)
  counts         TMM -> voomWithQualityWeights -> lmFit -> eBayes(robust = TRUE)
  paired glands  duplicateCorrelation on the patient (GSE173808)

Variants
  primary                 ~ 0 + group (+ batch) (+ covariates)
  infiltration_adjusted   + immune fraction from deconvolution (HPA salivary-gland reference)
  cd45_adjusted           + histological %CD45 (GSE173808 only; checks the deconvolution adjustment)
Discovery cohort also: pSS vs Sicca, within-batch label-permutation calibration.

Meta-analysis: DerSimonian-Laird random effects over the case-vs-control contrast of all cohorts. Each
cohort's moderated t is converted to an equivalent z (accounts for small residual df), SE = |logFC| / |z|.
Robustness: leave-one-cohort-out maximum P and a replication-only meta (discovery cohort excluded).

Outputs (``outputs/stage03_diffexpr``): de_all_genes.csv (one row per gene), candidate_genes.csv,
<GSE>__<contrast>__<variant>.csv, deconvolution_<GSE>.csv, deconvolution_validation.csv,
calibration_permutation.csv, de_summary.csv, figures/.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .context import Context
from .deconvolution import deconvolve
from .external_data import hpa_salivary_reference
from .preprocessing import batches_confounded
from .resources import infiltration_score
from .utils import bh_adjust, dump_json, read_table, run_r

warnings.filterwarnings("ignore", category=RuntimeWarning)
STAGE = "stage03_diffexpr"
S2 = "stage02_preprocess"
R_DIR = Path(__file__).resolve().parents[1] / "R"


# --------------------------------------------------------------------------
# meta-analysis
# --------------------------------------------------------------------------
def t_to_se(tab: pd.DataFrame) -> pd.Series:
    """Equivalent-z standard error: z = Phi^-1(P_t), SE = |logFC| / |z| (falls back to limma SE)."""
    p = tab["P.Value"].clip(lower=1e-300)
    z = stats.norm.isf(p / 2.0)
    se = tab["logFC"].abs() / np.where(z > 1e-8, z, np.nan)
    return se.where(np.isfinite(se) & (se > 0), tab["SE"])


def dl_meta(b: pd.DataFrame, se: pd.DataFrame, min_k: int) -> pd.DataFrame:
    """DerSimonian-Laird random-effects meta-analysis, genes x cohorts (NaN = not measured)."""
    B, S = b.to_numpy(float), se.to_numpy(float)
    ok = np.isfinite(B) & np.isfinite(S) & (S > 0)
    W = np.where(ok, 1.0 / np.where(ok, S, 1.0) ** 2, 0.0)
    Bz = np.where(ok, B, 0.0)
    k = ok.sum(axis=1)
    sw = W.sum(axis=1)
    fe = (W * Bz).sum(axis=1) / np.where(sw > 0, sw, np.nan)
    Q = (W * (Bz - fe[:, None]) ** 2 * ok).sum(axis=1)
    c = sw - (W ** 2).sum(axis=1) / np.where(sw > 0, sw, np.nan)
    tau2 = np.maximum(0.0, (Q - (k - 1)) / np.where(c > 0, c, np.nan))
    Wr = np.where(ok, 1.0 / (np.where(ok, S, 1.0) ** 2 + tau2[:, None]), 0.0)
    beta = (Wr * Bz).sum(axis=1) / Wr.sum(axis=1)
    se_m = 1.0 / np.sqrt(Wr.sum(axis=1))
    z = beta / se_m
    sign_ok = (np.sign(Bz) == np.sign(beta)[:, None]) & ok
    out = pd.DataFrame({"meta_logFC": beta, "meta_SE": se_m, "meta_z": z, "meta_P": 2 * stats.norm.sf(np.abs(z)),
                        "tau2": tau2, "Q_P": stats.chi2.sf(Q, np.maximum(k - 1, 1)),
                        "I2": np.where(Q > 0, np.maximum(0.0, (Q - (k - 1)) / np.where(Q > 0, Q, 1.0)), 0.0),
                        "n_cohorts": k, "n_same_direction": sign_ok.sum(axis=1)}, index=b.index)
    out = out[out["n_cohorts"] >= min_k]
    out["meta_FDR"] = bh_adjust(out["meta_P"].to_numpy())
    return out


def leave_one_out(b: pd.DataFrame, se: pd.DataFrame, min_k: int) -> pd.Series:
    """Max meta P over leave-one-cohort-out analyses (genes with < min_k - 1 cohorts left -> NaN)."""
    ps = []
    for c in b.columns:
        m = dl_meta(b.drop(columns=c), se.drop(columns=c), max(min_k - 1, 2))
        ps.append(m["meta_P"].reindex(b.index))
    return pd.concat(ps, axis=1).max(axis=1)


# --------------------------------------------------------------------------
def _volcano(tab: pd.DataFrame, title: str, path, fdr: float, lfc: float) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    y = -np.log10(tab["meta_FDR"].clip(lower=1e-300))
    sig = (tab["meta_FDR"] < fdr) & (tab["meta_logFC"].abs() >= lfc)
    ax.scatter(tab["meta_logFC"][~sig], y[~sig], s=5, c="#bbbbbb", alpha=0.6, lw=0)
    ax.scatter(tab["meta_logFC"][sig & (tab["meta_logFC"] > 0)], y[sig & (tab["meta_logFC"] > 0)], s=7, c="#d62728", lw=0)
    ax.scatter(tab["meta_logFC"][sig & (tab["meta_logFC"] < 0)], y[sig & (tab["meta_logFC"] < 0)], s=7, c="#1f77b4", lw=0)
    ax.axhline(-np.log10(fdr), ls="--", c="k", lw=0.7)
    for _, r in tab.assign(gene=tab.index).nsmallest(12, "meta_P").iterrows():
        ax.annotate(r["gene"], (r["meta_logFC"], -np.log10(max(r["meta_FDR"], 1e-300))), fontsize=6)
    ax.set_xlabel("random-effects meta log2 fold change")
    ax.set_ylabel("-log10 FDR")
    ax.set_title(f"{title}\n{int(sig.sum())} genes (FDR<{fdr}, |lfc|>={lfc})", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _zs(x: pd.Series) -> pd.Series:
    return (x - x.mean()) / (x.std(ddof=0) or 1.0)


def run(ctx: Context) -> None:
    cfg = ctx.cfg["de"]
    out, s2 = ctx.stage_dir(STAGE), ctx.stage_dir(S2)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    rin = out / "_r_inputs"
    rin.mkdir(exist_ok=True)
    disc = ctx.cfg["analysis"]["discovery_dataset"]
    ref = hpa_salivary_reference(ctx)
    dc_cfg = ctx.cfg["deconvolution"]
    jobs, cohorts, deconv_val = [], {}, []

    for acc, d in ctx.datasets():
        if d["type"] == "scrna_10x":
            continue
        meta = pd.read_csv(s2 / f"{acc}_samples_final.csv", index_col=0)
        meta["batch"] = meta["batch"].astype(str)
        expr = read_table(s2 / f"{acc}_gene_expr.parquet").loc[meta.index]
        case, ctrl = d["case"], d["control"]

        # ---- deconvolution (log_ratio arrays cannot be deconvolved -> marker score) ----------
        if d.get("scale") == "log_ratio":
            meta["immune"] = infiltration_score(ctx, expr)
            src = "marker score (log ratios)"
        else:
            dec = deconvolve(expr, ref, dc_cfg["markers_per_type"], dc_cfg["min_specificity"])
            dec.to_csv(out / f"deconvolution_{acc}.csv")
            meta["immune"] = dec["immune_fraction"]
            src = "NNLS deconvolution"
            ms = infiltration_score(ctx, expr)
            if d.get("measured_infiltration") == "cd45":
                tis = meta["char_tissue"].astype(str)
                cd45 = pd.to_numeric(np.where(tis.str.contains("parotid"), meta["char_parotid_%cd45"],
                                              meta["char_labial_%cd45"]), errors="coerce")
                meta["cd45"] = cd45
                for t in sorted(tis.unique()):
                    k = (tis == t).to_numpy() & np.isfinite(cd45)
                    deconv_val.append({"dataset": acc, "tissue": t, "n": int(k.sum()),
                                       "spearman_deconv_vs_CD45": stats.spearmanr(dec["immune_fraction"][k], cd45[k])[0],
                                       "spearman_marker_score_vs_CD45": stats.spearmanr(ms[k], cd45[k])[0]})
        meta["immune_z"] = _zs(meta["immune"])
        ctx.log.info("[%s] immune fraction (%s) by group: %s", acc, src,
                     meta.groupby("group")["immune"].mean().round(3).to_dict())

        # ---- design ------------------------------------------------------------------
        levels = [ctrl, case] + [g for g in sorted(meta["group"].unique()) if g not in (ctrl, case)]
        terms = ["0", "group"]
        if meta["batch"].nunique() > 1:
            terms.append("batch")
        terms += d.get("covariates", [])
        if "tissue" in d.get("covariates", []):
            meta["tissue"] = meta["char_tissue"].astype(str).str.replace(" ", "_")
        contrasts = {f"{case}_vs_{ctrl}": [case, ctrl]}
        contrasts.update(d.get("extra_contrasts", {}))
        kind = "counts" if d["type"] == "rnaseq_counts" else ("array" if d["type"].startswith("microarray") else "logexpr")
        mat = read_table(s2 / f"{acc}_gene_counts.parquet").loc[meta.index] if kind == "counts" else expr
        ep, mp = rin / f"{acc}_expr.csv.gz", rin / f"{acc}_meta.csv"
        mat.T.to_csv(ep)
        meta.to_csv(mp)
        variants = {"primary": terms, "infiltration_adjusted": terms + ["immune_z"]}
        if "cd45" in meta:
            meta["cd45_z"] = _zs(np.log1p(meta["cd45"].fillna(meta["cd45"].median())))
            meta.to_csv(mp)
            variants["cd45_adjusted"] = terms + ["cd45_z"]
        for vname, tt in variants.items():
            job = {"name": f"{acc}__{vname}", "kind": kind, "expr": str(ep), "meta": str(mp),
                   "formula": "~" + " + ".join(tt), "group_col": "group", "levels": levels, "contrasts": contrasts,
                   "block": d.get("block") and f"char_{d['block']}", "array_weights": bool(cfg["array_weights"]) and kind != "counts",
                   "out_dir": str(out)}
            if acc == disc and vname == "primary":
                job["permutations"] = {"n": int(cfg["n_permutations"]), "seed": ctx.seed, "within": "batch"}
            jobs.append(job)
        cohorts[acc] = {"case": case, "ctrl": ctrl, "contrast": f"{case}_vs_{ctrl}", "n_case": int((meta.group == case).sum()),
                        "n_ctrl": int((meta.group == ctrl).sum()), "tissue": d["tissue"], "kind": kind,
                        "confounded": batches_confounded(meta["batch"], meta["group"]) if meta["batch"].nunique() > 1 else False}

    jf = rin / "jobs.json"
    jf.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    ctx.log.info("running %d limma jobs in R", len(jobs))
    run_r(R_DIR / "limma_jobs.R", [jf], ctx.cfg["project"].get("rscript", "auto"), ctx.log)

    # ---- collect ----------------------------------------------------------------------
    tabs: Dict = {}
    summary = []
    for f in sorted(out.glob("*__*__*.csv")):
        acc, variant, contrast = f.stem.split("__")
        if contrast == "permutations":
            continue
        t = pd.read_csv(f).set_index("gene")
        tabs[(acc, variant, contrast)] = t
        summary.append({"dataset": acc, "variant": variant, "contrast": contrast, "genes_tested": len(t),
                        "fdr<0.05": int((t["adj.P.Val"] < cfg["fdr"]).sum()),
                        "fdr<0.05_|lfc|>=1": int(((t["adj.P.Val"] < cfg["fdr"]) & (t["logFC"].abs() >= 1)).sum()),
                        "frac_p<0.05": float((t["P.Value"] < 0.05).mean())})
    summ = pd.DataFrame(summary)
    summ.to_csv(out / "de_summary.csv", index=False)
    ctx.log.info("DE summary:\n%s", summ.to_string(index=False))
    perm = out / f"{disc}__primary__permutations.csv"
    if perm.exists():
        cal = pd.read_csv(perm)
        cal.to_csv(out / "calibration_permutation.csv", index=False)
        ctx.log.info("permutation calibration (%s, %d perms): mean genes FDR<0.05 = %.2f, mean frac p<0.05 = %.3f (expect 0.05)",
                     disc, len(cal), cal["n_fdr"].mean(), cal["frac_p05"].mean())
    if deconv_val:
        dv = pd.DataFrame(deconv_val)
        dv.to_csv(out / "deconvolution_validation.csv", index=False)
        ctx.log.info("deconvolution vs histology (%%CD45):\n%s", dv.round(3).to_string(index=False))

    # ---- meta-analysis -------------------------------------------------------------------
    def stack(variant: str, col: str) -> pd.DataFrame:
        cols = {}
        for acc, c in cohorts.items():
            t = tabs.get((acc, variant, c["contrast"]))
            if t is not None:
                cols[acc] = t[col] if col != "SE_eq" else t_to_se(t)
        return pd.DataFrame(cols)

    k_min = int(cfg["meta_min_cohorts"])
    b, se = stack("primary", "logFC"), stack("primary", "SE_eq")
    meta_tab = dl_meta(b, se, k_min)
    meta_tab["loo_max_P"] = leave_one_out(b, se, k_min).reindex(meta_tab.index)
    rep = dl_meta(b.drop(columns=disc), se.drop(columns=disc), k_min - 1)
    meta_tab = meta_tab.join(rep[["meta_logFC", "meta_P", "meta_FDR", "n_cohorts"]].add_prefix("rep_"), how="left")
    pv = stack("primary", "P.Value")
    aligned = (np.sign(b).eq(np.sign(meta_tab["meta_logFC"]).reindex(b.index), axis=0)) & (pv < 0.05)
    meta_tab["n_nominal_same_direction"] = aligned.sum(axis=1).reindex(meta_tab.index)
    ba, sa = stack("infiltration_adjusted", "logFC"), stack("infiltration_adjusted", "SE_eq")
    adj = dl_meta(ba, sa, k_min)
    meta_tab = meta_tab.join(adj[["meta_logFC", "meta_SE", "meta_P", "meta_FDR", "n_same_direction"]].add_prefix("adj_"), how="left")

    allg = meta_tab.join(b.add_prefix("logFC_")).join(pv.add_prefix("P_"))
    d_con = cohorts[disc]["contrast"]
    dt = tabs[(disc, "primary", d_con)]
    allg = allg.join(dt[["logFC", "SE", "t", "P.Value", "adj.P.Val", "AveExpr"]].add_prefix("disc_"), how="left")
    for (acc, variant, contrast), t in tabs.items():
        if acc == disc and variant == "primary" and contrast != d_con:
            allg = allg.join(t[["logFC", "adj.P.Val"]].add_prefix(f"{contrast}_"), how="left")
        if variant == "cd45_adjusted":
            allg = allg.join(t[["logFC", "P.Value"]].add_prefix(f"cd45adj_{acc}_"), how="left")
    allg = allg.sort_values("meta_P")
    allg.index.name = "gene"
    allg.to_csv(out / "de_all_genes.csv")

    fdr, lfc = cfg["candidate_meta_fdr"], cfg["relaxed_abs_lfc"]
    cand = allg[(allg["meta_FDR"] < fdr) & (allg["meta_logFC"].abs() >= lfc) & (allg["loo_max_P"] < 0.05)]
    cand.to_csv(out / "candidate_genes.csv")
    _volcano(allg, f"Random-effects meta-analysis, {len(cohorts)} gland cohorts", fig_dir / "meta_volcano.png", fdr, lfc)
    n_adj = int(((cand["adj_meta_FDR"] < fdr) & (np.sign(cand["adj_meta_logFC"]) == np.sign(cand["meta_logFC"]))).sum())
    ctx.log.info("meta-analysis: %d genes in >= %d cohorts; FDR<%.2f: %d; candidates (|lfc|>=%.2f, robust to leaving "
                 "any cohort out): %d (%d up, %d down); replication-only meta FDR<0.05 among candidates: %d; "
                 "still significant after deconvolution adjustment: %d", len(allg), k_min, fdr,
                 int((allg["meta_FDR"] < fdr).sum()), lfc, len(cand), int((cand["meta_logFC"] > 0).sum()),
                 int((cand["meta_logFC"] < 0).sum()), int((cand["rep_meta_FDR"] < 0.05).sum()), n_adj)
    dump_json({"cohorts": cohorts, "meta_genes": int(len(allg)), "meta_fdr05": int((allg["meta_FDR"] < fdr).sum()),
               "candidates": int(len(cand)), "candidates_infiltration_robust": n_adj}, out / "stage3_summary.json")
    ctx.log.info("stage 3 done -> %s", out)

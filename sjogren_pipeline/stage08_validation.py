"""Stage 8: validation of the leading targets and RT-qPCR panel design.

  1. Cohort-level replication of the top-N targets: direction and nominal significance in each of the gland
     cohorts, replication-only meta-analysis (discovery cohort excluded), and a permutation test of the mean
     sign-aligned replication-only z of the top-N against random measured genes.
  2. Independent single-cell validation in salivary glands (GSE272409, 7 SjD vs 6 sicca; never used for the
     ranking): which cell types express each target (mean log-CP10K, % expressing) and pseudobulk limma-voom
     SjD vs sicca within the immune, epithelial and stromal compartments -- does the target change *inside*
     the compartment that expresses it (not only through cell abundance)?
  3. RT-qPCR panel: top targets (half gland-intrinsic when available) + positive / negative controls + geNorm
     reference-gene screen on the discovery data; per-group n for the requested power from the conservative
     replication-only effect size.

Outputs (``outputs/stage08_validation``): replication_topN.csv, replication_tests.csv,
scrna_celltype_expression.csv, scrna_celltype_counts.csv, scrna_pseudobulk_<compartment>.csv,
scrna_target_summary.csv, qpcr_panel.csv, reference_gene_stability.csv.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower

from . import scrna
from .context import Context
from .external_data import hpa_salivary_reference
from .preprocessing import remove_batch_effect
from .utils import read_table, run_r

warnings.filterwarnings("ignore")
STAGE, S1, S2, S3, S7 = "stage08_validation", "stage01_download", "stage02_preprocess", "stage03_diffexpr", "stage07_ranking"
R_DIR = Path(__file__).resolve().parents[1] / "R"


def genorm(expr: pd.DataFrame, genes: list) -> pd.DataFrame:
    """geNorm stability (M): iteratively drop the gene with the largest mean SD of pairwise log-ratios."""
    cur = list(genes)
    hist = []
    while len(cur) > 2:
        sub = expr[cur]
        M = {g: np.mean([np.std(sub[g] - sub[h], ddof=1) for h in cur if h != g]) for g in cur}
        worst = max(M, key=M.get)
        hist.append({"gene": worst, "M": M[worst], "rank_from_worst": len(hist) + 1})
        cur.remove(worst)
    for g in cur:
        hist.append({"gene": g, "M": np.mean([np.std(expr[g] - expr[h], ddof=1) for h in cur if h != g]),
                     "rank_from_worst": len(hist) + 1})
    df = pd.DataFrame(hist)
    df["stability_rank"] = df["rank_from_worst"].max() + 1 - df["rank_from_worst"]
    return df.sort_values("stability_rank")


def single_cell(ctx: Context, out: Path, targets: list) -> pd.DataFrame:
    acc = next((a for a, d in ctx.datasets("scrna_10x")), None)
    if acc is None:
        return pd.DataFrame()
    files = scrna.sample_files(ctx.raw / "scrna" / acc)
    if not files:
        ctx.log.warning("no single-cell matrices in %s -> skipped", ctx.raw / "scrna" / acc)
        return pd.DataFrame()
    dcfg = ctx.cfg["datasets"][acc]
    smeta = pd.read_csv(ctx.stage_dir(S1) / f"{acc}_samples.csv").set_index("gsm")
    ref = hpa_salivary_reference(ctx)
    expr_rows, count_rows, pb = [], [], {"immune": {}, "epithelial": {}, "stromal": {}}
    for gsm, (mtx, feat, _) in files.items():
        X, genes = scrna.load_sample(mtx, feat)
        Y = scrna.lognorm(X)
        ct = scrna.annotate(Y, genes, ref).to_numpy()
        comp = np.array([scrna.compartment_of(c) if c != "unassigned" else "unassigned" for c in ct])
        t = scrna.gene_celltype_table(Y, genes, ct, targets)
        t.insert(0, "sample", gsm)
        expr_rows.append(t)
        vc = pd.Series(ct).value_counts()
        count_rows.append(pd.DataFrame({"sample": gsm, "group": smeta.loc[gsm, "group"], "cell_type": vc.index, "n": vc.values}))
        for c in pb:
            m = comp == c
            if m.sum() >= 20:
                pb[c][gsm] = scrna.pseudobulk(X, genes, m)
        ctx.log.info("  [%s] %s: %d cells after QC (%s)", acc, gsm, X.shape[1],
                     pd.Series(comp).value_counts().to_dict())
    counts = pd.concat(count_rows)
    counts.to_csv(out / "scrna_celltype_counts.csv", index=False)
    E = pd.concat(expr_rows)
    # pooled over samples, weighted by cells
    E["w_mean"] = E["mean_logcp10k"] * E["n_cells"]
    E["w_pct"] = E["pct_expressing"] * E["n_cells"]
    g = E.groupby(["gene", "cell_type", "compartment"], as_index=False)[["w_mean", "w_pct", "n_cells"]].sum()
    g["mean_logcp10k"] = g["w_mean"] / g["n_cells"]
    g["pct_expressing"] = g["w_pct"] / g["n_cells"]
    g = g.drop(columns=["w_mean", "w_pct"])
    g.to_csv(out / "scrna_celltype_expression.csv", index=False)

    # pseudobulk limma-voom per compartment (R)
    rin = out / "_r_inputs"
    rin.mkdir(exist_ok=True)
    jobs = []
    for c, cols in pb.items():
        if len(cols) < 6:
            continue
        P = pd.DataFrame(cols).fillna(0)
        meta = smeta.loc[P.columns, ["group"]]
        if meta["group"].value_counts().min() < 3:
            continue
        cpm = P / P.sum(axis=0) * 1e6
        P = P[(cpm >= 1).sum(axis=1) >= 3]
        P.to_csv(rin / f"pb_{c}.csv.gz")
        meta.to_csv(rin / f"pb_{c}_meta.csv")
        jobs.append({"name": f"pseudobulk_{c}", "kind": "counts", "expr": str(rin / f"pb_{c}.csv.gz"),
                     "meta": str(rin / f"pb_{c}_meta.csv"), "formula": "~0 + group", "group_col": "group",
                     "levels": [dcfg["control"], dcfg["case"]], "contrasts": {"SjD_vs_Sicca": [dcfg["case"], dcfg["control"]]},
                     "block": None, "array_weights": False, "out_dir": str(out)})
    if jobs:
        (rin / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
        run_r(R_DIR / "limma_jobs.R", [rin / "jobs.json"], ctx.cfg["project"].get("rscript", "auto"), ctx.log)
    # per-target summary: top expressing cell type and within-compartment change
    rows = []
    for gene, d in g[g["cell_type"] != "unassigned"].groupby("gene"):
        top = d.sort_values("mean_logcp10k", ascending=False).iloc[0]
        row = {"gene": gene, "scrna_top_cell_type": top["cell_type"], "scrna_top_compartment": top["compartment"],
               "scrna_top_mean_logcp10k": top["mean_logcp10k"], "scrna_top_pct": top["pct_expressing"]}
        for c in pb:
            f = out / f"pseudobulk_{c}__SjD_vs_Sicca.csv"
            if f.exists():
                t = pd.read_csv(f).set_index("gene")
                if gene in t.index:
                    row[f"{c}_logFC"], row[f"{c}_P"] = t.loc[gene, "logFC"], t.loc[gene, "P.Value"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("gene") if rows else pd.DataFrame()


def run(ctx: Context) -> None:
    cfg = ctx.cfg["validation"]
    out = ctx.stage_dir(STAGE)
    s2, s3, s7 = ctx.stage_dir(S2), ctx.stage_dir(S3), ctx.stage_dir(S7)
    d_acc = ctx.cfg["analysis"]["discovery_dataset"]
    rng = np.random.default_rng(ctx.seed)
    N = cfg["top_n"]

    rank = pd.read_csv(s7 / "target_ranking_all.csv", index_col=0)
    de = pd.read_csv(s3 / "de_all_genes.csv", index_col=0)
    top = rank.head(N)

    # ---- 1. replication -----------------------------------------------------------------------
    cohorts = [c[len("logFC_"):] for c in de.columns if c.startswith("logFC_")]
    rep = top[["priority_score", "meta_logFC", "meta_FDR", "rep_meta_FDR", "n_cohorts", "n_same_direction",
               "n_nominal_same_direction", "infiltration_robust", "top_cell_type", "clinical_sjogren_target"]].copy()
    for c in cohorts:
        rep[f"{c}_logFC"] = de[f"logFC_{c}"].reindex(rep.index)
        rep[f"{c}_P"] = de[f"P_{c}"].reindex(rep.index)
    rep.to_csv(out / "replication_topN.csv")
    rz = np.sign(de["meta_logFC"]) * stats.norm.isf(de["rep_meta_P"].clip(lower=1e-300) / 2) * \
        np.sign(de["rep_meta_logFC"]) * np.sign(de["meta_logFC"])
    rz = (np.sign(de["rep_meta_logFC"]) * np.sign(de["meta_logFC"]) * stats.norm.isf(de["rep_meta_P"].clip(lower=1e-300) / 2)).dropna()
    obs = rz.reindex(top.index).mean()
    null = np.array([rz.sample(N, random_state=int(s)).mean() for s in rng.integers(0, 1e9, 5000)])
    tests = pd.DataFrame([
        {"test": f"mean sign-aligned replication-only meta z, top-{N} vs random genes", "observed": obs,
         "null_mean": null.mean(), "null_95th": np.percentile(null, 95), "p": (1 + (null >= obs).sum()) / (1 + len(null))},
        {"test": f"top-{N}: cohorts in the meta direction (mean fraction)",
         "observed": float((top["n_same_direction"] / top["n_cohorts"]).mean()), "null_mean": 0.5, "null_95th": np.nan, "p": np.nan},
        {"test": f"top-{N}: replication-only meta FDR<0.05 (count)", "observed": int((top["rep_meta_FDR"] < 0.05).sum()),
         "null_mean": np.nan, "null_95th": np.nan, "p": np.nan}])
    tests.to_csv(out / "replication_tests.csv", index=False)
    ctx.log.info("replication of top-%d:\n%s", N, tests.round(4).to_string(index=False))

    # ---- 2. independent single-cell validation ---------------------------------------------------
    sc_targets = rank.head(max(N, 50)).index.tolist()
    try:
        sc = single_cell(ctx, out, sc_targets)
    except Exception as exc:
        ctx.log.warning("single-cell validation failed: %s", exc)
        sc = pd.DataFrame()
    if len(sc):
        summ = rank.head(50)[["priority_score", "meta_logFC", "top_cell_type", "top_compartment", "infiltration_robust"]].join(sc)
        comp = summ["scrna_top_compartment"]
        within_p = pd.Series([summ.loc[g, f"{c}_P"] if isinstance(c, str) and f"{c}_P" in summ else np.nan
                              for g, c in comp.items()], index=summ.index)
        within_lfc = pd.Series([summ.loc[g, f"{c}_logFC"] if isinstance(c, str) and f"{c}_logFC" in summ else np.nan
                                for g, c in comp.items()], index=summ.index)
        summ["within_compartment_logFC"] = within_lfc
        summ["within_compartment_P"] = within_p
        summ["within_compartment_supports"] = (np.sign(within_lfc) == np.sign(summ["meta_logFC"])) & (within_p < 0.05)
        summ["hpa_vs_scrna_compartment_agree"] = summ["top_compartment"] == summ["scrna_top_compartment"]
        summ.to_csv(out / "scrna_target_summary.csv")
        t20 = summ.head(N)
        ctx.log.info("single-cell (GSE272409) for top-%d: compartment agrees with HPA for %d/%d; changed in the same "
                     "direction (P<0.05) inside the expressing compartment, SjD vs sicca: %d/%d", N,
                     int(t20["hpa_vs_scrna_compartment_agree"].sum()), int(t20["scrna_top_compartment"].notna().sum()),
                     int(t20["within_compartment_supports"].sum()), int(t20["within_compartment_P"].notna().sum()))

    # ---- 3. qPCR panel ---------------------------------------------------------------------------
    meta_d = pd.read_csv(s2 / f"{d_acc}_samples_final.csv", index_col=0)
    meta_d["batch"] = meta_d["batch"].astype(str)
    ex = read_table(s2 / f"{d_acc}_gene_expr.parquet").loc[meta_d.index]
    bc = remove_batch_effect(ex, meta_d["batch"].to_numpy(), covariates=meta_d[["group"]])
    refs = [g for g in cfg["qpcr_reference_genes_to_screen"] if g in bc.columns and g in de.index]
    stable = [g for g in refs if de.loc[g, "meta_P"] > 0.1 and de.loc[g, "disc_AveExpr"] >= de["disc_AveExpr"].median()]
    ref_tab = pd.DataFrame()
    if len(stable) >= 3:
        ref_tab = genorm(bc, stable).merge(de.loc[stable, ["meta_P", "meta_logFC", "disc_AveExpr"]], left_on="gene", right_index=True)
        ref_tab["pass_screen"] = ref_tab["stability_rank"] <= 3
        ref_tab.to_csv(out / "reference_gene_stability.csv", index=False)
    chosen_refs = ref_tab.head(3)["gene"].tolist() if len(ref_tab) else []

    n_panel = cfg["qpcr_panel_size"]
    pool = rank[~rank.index.isin(refs)]
    gland = pool[pool["infiltration_robust"] & (pool["top_compartment"] != "immune")].head(n_panel // 2)
    rest = pool[~pool.index.isin(gland.index)].head(n_panel - len(gland))
    panel = pd.concat([gland, rest])
    pos = rank[rank["clinical_sjogren_target"] & ~rank.index.isin(panel.index)].head(2)
    mid = de[(de["meta_P"] > 0.5) & de["disc_AveExpr"].notna()]
    neg = mid.iloc[(mid["disc_AveExpr"] - de.loc[panel.index, "disc_AveExpr"].median()).abs().argsort()[:3]]
    case = ctx.cfg["datasets"][d_acc]["case"]
    n_case, n_ctrl = int((meta_d["group"] == case).sum()), int((meta_d["group"] == ctx.cfg["datasets"][d_acc]["control"]).sum())
    alpha = cfg["alpha"] / max(len(panel), 1)
    rows = []
    for role, tab in [("candidate", panel), ("positive_control", pos)]:
        for g, r in tab.iterrows():
            sd = de.loc[g, "disc_SE"] / np.sqrt(1 / n_case + 1 / n_ctrl) if np.isfinite(de.loc[g, "disc_SE"]) else np.nan
            eff = de.loc[g, "rep_meta_logFC"] if np.isfinite(de.loc[g, "rep_meta_logFC"]) else de.loc[g, "meta_logFC"]
            n_req = (TTestIndPower().solve_power(effect_size=abs(eff) / max(sd, 1e-6), alpha=alpha, power=cfg["power"])
                     if np.isfinite(sd) and abs(eff) > 0 else np.nan)
            rows.append({"role": role, "gene": g, "priority_rank": int(r["rank"]), "expected_log2FC": eff,
                         "expected_dCt_shift": -eff, "sd_log2": sd,
                         "n_per_group_for_power": int(np.ceil(n_req)) if np.isfinite(n_req) else np.nan,
                         "top_cell_type": r.get("top_cell_type"), "compartment": r.get("top_compartment"),
                         "infiltration_robust": bool(r["infiltration_robust"]),
                         "clinical_sjogren_target": bool(r["clinical_sjogren_target"])})
    for g, r in neg.iterrows():
        rows.append({"role": "negative_control", "gene": g, "expected_log2FC": r["meta_logFC"], "expected_dCt_shift": -r["meta_logFC"],
                     "top_cell_type": "", "compartment": "", "note": "non-DE in meta-analysis (P>0.5), similar expression"})
    for g in chosen_refs:
        rows.append({"role": "reference_gene", "gene": g, "expected_log2FC": de.loc[g, "meta_logFC"],
                     "expected_dCt_shift": -de.loc[g, "meta_logFC"], "note": "geNorm-ranked stable gene (non-DE, high expression)"})
    q = pd.DataFrame(rows)
    q["suggested_material"] = np.where(q["compartment"].eq("immune"),
                                       "gland biopsy RNA + sorted gland-infiltrating lymphocytes",
                                       "gland biopsy RNA; primary salivary gland epithelial cells (SGEC)")
    q["perturbation_suggestion"] = np.where(q["compartment"].eq("immune"),
                                            "blocking antibody / knockdown in primary B or T cells; read out activation & IFN genes",
                                            "siRNA/CRISPR in SGECs +/- IFN-alpha/gamma; read out secretory (AQP5) and IFN genes")
    q.to_csv(out / "qpcr_panel.csv", index=False)
    ctx.log.info("qPCR panel: %d candidates (%d gland-intrinsic), %d positive, %d negative controls, references %s; "
                 "median n per group for %.0f%% power (Bonferroni alpha %.4f, replication-only effect): %s",
                 len(panel), len(gland), len(pos), len(neg), chosen_refs, 100 * cfg["power"], alpha,
                 q.loc[q["role"] == "candidate", "n_per_group_for_power"].median())
    ctx.log.info("stage 8 done -> %s", out)

"""Stage 9: publication figures for results and model evaluation (reads stages 2-8, recomputes nothing heavy).

Results
  fig01_cohorts                 cohorts, tissue, case / control sizes
  fig02_de_quality              permutation calibration, deconvolution vs histology, DE genes per cohort
  fig03_meta_volcano            random-effects meta-analysis volcano
  fig04_cross_cohort_heatmap    log2FC of the top-30 targets in every cohort
  fig05_forest_top_targets      per-cohort effects + random-effects estimate for the top 6 targets
  fig06_pathways                Hallmark GSEA, primary vs deconvolution-adjusted
  fig07_ranking_contributions   top-30 priority score split by evidence layer
  fig08_ranking_robustness      held-out clinical-target AUROC per layer; weight-sensitivity rank intervals
  fig09_single_cell             gland scRNA cell-of-origin dot plot; within-compartment SjD vs sicca change
Models
  fig10_model_metrics           nested-CV AUROC, AUPRC, balanced accuracy, sensitivity, specificity, F1, MCC, Brier
  fig11_roc_pr                  out-of-fold ROC and precision-recall curves
  fig12_model_robustness        permutation null, leave-one-batch-out, preprocessing screen, confusion matrix
  fig13_external_validation     AUROC of every model in every external cohort (+ 95% CI of the best model)
  fig14_explainability          deletion-curve faithfulness and top consensus genes
  fig15_consensus_targets       agreement of the multi-evidence ranking with DeepMOLFT SHAP; tier-1 consensus genes

Every figure is written as PNG (300 dpi) and PDF; the numbers behind the model figures are in
model_metrics.csv / model_metrics_by_repeat.csv (table view for each chart).
Palette: validated categorical slots (adjacent pairs; first three all-pairs), blue sequential, blue-gray-red diverging.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)

from .context import Context

warnings.filterwarnings("ignore")
STAGE = "figures"

# ---- palette (validated: scripts/validate_palette.js, light surface) ------------------------------------
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
NEUTRAL = "#c9c8c2"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#6da7ec", "#f0efec", "#ef8a7e", "#b8322f"])
UP, DOWN = "#c9403f", "#256abf"
MODELS = ["elasticnet", "l2_logreg", "linear_svm", "random_forest", "grad_boosting", "ae_mlp"]
MODEL_LABEL = {"elasticnet": "Elastic net", "l2_logreg": "L2 logistic", "linear_svm": "Linear SVM",
               "random_forest": "Random forest", "grad_boosting": "Gradient boosting", "ae_mlp": "Supervised autoencoder",
               "dummy": "Prior (baseline)"}
MODEL_COLOR = dict(zip(MODELS, SLOT))
PROBA_MODELS = {"random_forest", "dummy"}          # score_vec used predict_proba; others decision_function / logit
LAYERS = ["differential_expression", "gland_intrinsic", "genetics", "tractability", "network", "model_importance", "safety"]
LAYER_LABEL = {"differential_expression": "Meta DE", "gland_intrinsic": "Gland-intrinsic", "genetics": "Genetics",
               "tractability": "Tractability", "network": "PPI network", "model_importance": "DeepMOLFT SHAP",
               "safety": "Safety", "model_importance_dm": "DeepMOLFT SHAP", "cell_type_specificity": "Cell-type specificity",
               "priority_score": "Priority score (all layers)",
               "priority_without_tractability": "Priority without tractability"}


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 8, "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "legend.fontsize": 7, "lines.linewidth": 1.5, "text.color": INK,
        "pdf.fonttype": 42})


def _save(fig, out, name: str) -> None:
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _panel(ax, letter: str) -> None:
    ax.text(-0.12, 1.06, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", color=INK)


def _bars_gap():
    return {"edgecolor": SURFACE, "linewidth": 1.0}


def _se_eq(lfc: pd.Series, p: pd.Series) -> pd.Series:
    z = stats.norm.isf(p.clip(lower=1e-300) / 2.0)
    return lfc.abs() / np.where(z > 1e-8, z, np.nan)


# ======================================================================================================
# results
# ======================================================================================================
def fig_cohorts(ctx, o, s3) -> None:
    summ = pd.read_json(s3 / "stage3_summary.json", typ="series")["cohorts"]
    rows = [{"cohort": k, "tissue": v["tissue"], "case": v["n_case"], "control": v["n_ctrl"], "kind": v["kind"],
             "control_label": ctx.cfg["datasets"][k]["control"]} for k, v in summ.items()]
    d = pd.DataFrame(rows).iloc[::-1]
    kind = {"array": "microarray", "counts": "RNA-seq counts", "logexpr": "RNA-seq TPM"}
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(d))
    ax.barh(y, d["case"], color=SLOT[0], height=0.62, label="pSS / SS", **_bars_gap())
    ax.barh(y, d["control"], left=d["case"], color=SLOT[1], height=0.62, label="Control (healthy or sicca)", **_bars_gap())
    for yi, r in zip(y, d.itertuples()):
        ax.text(r.case + r.control + 1.5, yi, f"{r.case} vs {r.control} {r.control_label.lower()}  ·  {r.tissue}  ·  {kind[r.kind]}",
                va="center", fontsize=7, color=INK2)
    ax.set_yticks(y, d["cohort"])
    ax.set_xlabel("samples in the case-vs-control contrast")
    ax.set_xlim(0, d[["case", "control"]].sum(axis=1).max() * 2.25)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper right", ncol=1)
    ax.set_title("Seven salivary-gland cohorts in the meta-analysis (+ GSE272409 scRNA-seq for validation)")
    _save(fig, o, "fig01_cohorts")


def fig_de_quality(ctx, o, s3) -> None:
    cal = pd.read_csv(s3 / "calibration_permutation.csv")
    summ = pd.read_csv(s3 / "de_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), gridspec_kw={"width_ratios": [1, 1.1, 1.4]})
    ax = axes[0]
    ax.hist(cal["frac_p05"], bins=20, color=SLOT[0], **_bars_gap())
    ax.axvline(0.05, color=INK, ls="--", lw=1)
    ax.axvline(cal["frac_p05"].mean(), color=SLOT[1], lw=1.5)
    ax.text(0.05, ax.get_ylim()[1] * 0.95, " expected 0.05", fontsize=7, va="top", color=INK)
    ax.text(cal["frac_p05"].mean(), ax.get_ylim()[1] * 0.8, f" observed mean {cal['frac_p05'].mean():.3f}", fontsize=7,
            va="top", color=INK)
    ax.set_xlabel("fraction of genes with p < 0.05")
    ax.set_ylabel("label permutations")
    ax.set_title("Label-permutation calibration")
    _panel(ax, "a")

    ax = axes[1]
    dv = s3 / "deconvolution_GSE173808.csv"
    if dv.exists():
        dec = pd.read_csv(dv, index_col=0)
        meta = pd.read_csv(ctx.stage_dir("stage02_preprocess") / "GSE173808_samples_final.csv", index_col=0)
        tis = meta["char_tissue"].astype(str)
        cd45 = pd.to_numeric(pd.Series(np.where(tis.str.contains("parotid"), meta["char_parotid_%cd45"],
                                                meta["char_labial_%cd45"]), index=meta.index), errors="coerce")
        for i, t in enumerate(["parotid gland", "labial gland"]):
            k = (tis == t) & cd45.notna()
            rho = stats.spearmanr(dec.loc[k[k].index, "immune_fraction"], cd45[k])[0]
            ax.scatter(cd45[k], dec.loc[k[k].index, "immune_fraction"], s=16, color=SLOT[i], edgecolor=SURFACE,
                       linewidth=0.6, label=f"{t.split()[0]} (Spearman ρ = {rho:.2f})")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("histological %CD45+ cells")
        ax.set_ylabel("deconvolved immune fraction")
        ax.legend(loc="lower right")
        ax.set_title("Deconvolution vs %CD45 (GSE173808)")
    _panel(ax, "b")

    ax = axes[2]
    p = summ[summ["variant"].isin(["primary", "infiltration_adjusted"])]
    p = p[p["contrast"].str.endswith(("Control", "Sicca"))]
    first = p.groupby("dataset").head(10)
    first = first[~((first["dataset"] == ctx.cfg["analysis"]["discovery_dataset"]) & (first["contrast"] == "pSS_vs_Sicca"))]
    w = first.pivot_table(index="dataset", columns="variant", values="fdr<0.05")
    w = w.loc[list(summ_order(ctx))].fillna(0)
    x = np.arange(len(w))
    ax.bar(x - 0.2, w["primary"], 0.4, color=SLOT[0], label="primary", **_bars_gap())
    ax.bar(x + 0.2, w["infiltration_adjusted"], 0.4, color=SLOT[1], label="+ immune fraction", **_bars_gap())
    ax.set_xticks(x, w.index, rotation=35, ha="right")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("genes at FDR < 0.05")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    for xi, (a, b) in enumerate(zip(w["primary"], w["infiltration_adjusted"])):
        for dx, v in ((-0.2, a), (0.2, b)):
            if v == 0:
                ax.text(xi + dx, 0.15, "0", ha="center", fontsize=6.5, color=INK2)
    ax.set_title("Genes at FDR < 0.05 per cohort")
    _panel(ax, "c")
    fig.tight_layout(w_pad=2.5)
    _save(fig, o, "fig02_de_quality")


def summ_order(ctx):
    return [a for a, d in ctx.datasets() if d["type"] != "scrna_10x"]


def fig_volcano(ctx, o, de, rank) -> None:
    fdr, lfc = ctx.cfg["de"]["candidate_meta_fdr"], ctx.cfg["de"]["relaxed_abs_lfc"]
    y = -np.log10(de["meta_P"].clip(lower=1e-300))
    sig = (de["meta_FDR"] < fdr) & (de["meta_logFC"].abs() >= lfc)
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(de.loc[~sig, "meta_logFC"], y[~sig], s=4, color=NEUTRAL, lw=0, rasterized=True, label="not significant")
    up, dn = sig & (de["meta_logFC"] > 0), sig & (de["meta_logFC"] < 0)
    ax.scatter(de.loc[up, "meta_logFC"], y[up], s=6, color=UP, lw=0, rasterized=True, label=f"up in pSS ({int(up.sum())})")
    ax.scatter(de.loc[dn, "meta_logFC"], y[dn], s=6, color=DOWN, lw=0, rasterized=True, label=f"down in pSS ({int(dn.sum())})")
    thr = y[de["meta_FDR"] < fdr].min()
    ax.axhline(thr, color=MUTED, ls="--", lw=0.8)
    for x0 in (-lfc, lfc):
        ax.axvline(x0, color=MUTED, ls="--", lw=0.8)
    lab = [g for g in rank.head(12).index if g in de.index]
    for g in lab:
        ax.annotate(g, (de.loc[g, "meta_logFC"], y[g]), xytext=(4, 2), textcoords="offset points", fontsize=6.5, color=INK)
        ax.scatter(de.loc[g, "meta_logFC"], y[g], s=14, facecolor="none", edgecolor=INK, lw=0.7)
    ax.set_xlabel("random-effects meta log2 fold change (pSS vs control)")
    ax.set_ylabel("−log10 meta P")
    ax.legend(loc="upper left", markerscale=2)
    ax.set_title(f"Meta-analysis of 7 gland cohorts: {int(sig.sum())} genes (FDR < {fdr}, |log2FC| ≥ {lfc})")
    _save(fig, o, "fig03_meta_volcano")


def fig_heatmap(ctx, o, de, rank) -> None:
    top = rank.head(30).index
    cohorts = summ_order(ctx)
    M = de.loc[top, [f"logFC_{c}" for c in cohorts]]
    M.columns = cohorts
    P = de.loc[top, [f"P_{c}" for c in cohorts]]
    M["meta"] = de.loc[top, "meta_logFC"]
    fig, ax = plt.subplots(figsize=(6.2, 8.2))
    v = np.nanpercentile(np.abs(M.to_numpy()), 98)
    im = ax.imshow(M.to_numpy(), cmap=DIV, vmin=-v, vmax=v, aspect="auto")
    for i in range(len(top)):
        for j, c in enumerate(cohorts):
            p = P.iloc[i, j]
            if np.isfinite(p) and p < 0.05:
                ax.text(j, i, "•" if p >= 0.001 else "••", ha="center", va="center", fontsize=6, color=INK)
    ax.axvline(len(cohorts) - 0.5, color=SURFACE, lw=3)
    ax.set_xticks(range(M.shape[1]), list(M.columns), rotation=40, ha="right")
    clin = rank.loc[top, "clinical_sjogren_target"]
    ax.set_yticks(range(len(top)), [f"{g} *" if c else g for g, c in zip(top, clin)])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.45, pad=0.02)
    cb.set_label("log2 fold change (pSS vs control)")
    cb.outline.set_visible(False)
    ax.set_title("Top-30 targets in every cohort (• P < 0.05, •• P < 0.001; grey = not measured)")
    ax.set_facecolor(GRID)
    _save(fig, o, "fig04_cross_cohort_heatmap")


def fig_forest(ctx, o, de, rank) -> None:
    genes = rank.head(6).index
    cohorts = summ_order(ctx)
    fig, axes = plt.subplots(2, 3, figsize=(10, 5.6), sharex=False)
    for ax, g in zip(axes.ravel(), genes):
        b = pd.Series({c: de.loc[g, f"logFC_{c}"] for c in cohorts})
        p = pd.Series({c: de.loc[g, f"P_{c}"] for c in cohorts})
        se = _se_eq(b, p)
        ok = b.notna()
        y = np.arange(ok.sum())[::-1] + 1
        ax.errorbar(b[ok], y, xerr=1.96 * se[ok], fmt="o", ms=4.5, color=SLOT[0], ecolor=SLOT[0], elinewidth=1.2, capsize=0)
        m, s = de.loc[g, "meta_logFC"], de.loc[g, "meta_SE"]
        ax.fill([m - 1.96 * s, m, m + 1.96 * s, m], [0, 0.28, 0, -0.28], color=SLOT[1])
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_yticks(list(y) + [0], list(b[ok].index) + ["Random effects"])
        ax.set_title(f"{g}  (I² = {de.loc[g, 'I2']:.2f}, meta FDR = {de.loc[g, 'meta_FDR']:.1e})", fontsize=8)
        ax.set_xlabel("log2FC ± 95% CI")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Per-cohort effects and random-effects estimate for the top-6 ranked targets", x=0.01, ha="left",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, o, "fig05_forest_top_targets")


def fig_pathways(ctx, o, s4) -> None:
    g = pd.read_csv(s4 / "gsea_results.csv")
    h = g[g["library"] == "MSigDB_Hallmark_2020"]
    w = h.pivot_table(index="term", columns="model", values="NES")
    f = h.pivot_table(index="term", columns="model", values="FDR")
    keep = f.index[(f < 0.05).any(axis=1)]
    w = w.loc[keep].sort_values("primary")
    f = f.loc[w.index]
    fig, ax = plt.subplots(figsize=(6.4, 0.24 * len(w) + 1.4))
    y = np.arange(len(w))
    for i, (m, lab) in enumerate([("primary", "primary meta-analysis"), ("infiltration_adjusted", "+ immune-fraction adjustment")]):
        sig = f[m] < 0.05
        ax.scatter(w.loc[sig, m], y[sig.to_numpy()], s=34, color=SLOT[i], edgecolor=SURFACE, lw=0.8, zorder=3, label=f"{lab} (FDR < 0.05)")
        ax.scatter(w.loc[~sig, m], y[~sig.to_numpy()], s=34, facecolor=SURFACE, edgecolor=SLOT[i], lw=1.2, zorder=3,
                   label=f"{lab} (n.s.)")
    for yi, (a, b) in enumerate(zip(w["primary"], w["infiltration_adjusted"])):
        ax.plot([a, b], [yi, yi], color=GRID, lw=1.5, zorder=1)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, w.index)
    ax.set_xlabel("normalised enrichment score (GSEA prerank on meta z)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=6.5)
    ax.set_title("MSigDB Hallmark pathways (FDR < 0.05 in either analysis)")
    _save(fig, o, "fig06_pathways")


def fig_ranking(ctx, o, rank) -> None:
    mode = ctx.cfg["ranking"]["mode"]
    w = pd.Series(ctx.cfg["ranking"]["modes"][mode]["weights"], dtype=float)
    w = w[w > 0] / w[w > 0].sum()
    lay = [l for l in LAYERS if l in w.index]
    top = rank.head(30)
    pen = top["priority_score"] / (top[lay] * w[lay]).sum(axis=1)
    C = (top[lay] * w[lay]).mul(pen, axis=0).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    left = np.zeros(len(C))
    for i, l in enumerate(lay):
        ax.barh(range(len(C)), C[l], left=left, color=SLOT[i], height=0.72, label=LAYER_LABEL[l], **_bars_gap())
        left += C[l].to_numpy()
    comp = top["top_compartment"].iloc[::-1].fillna("")
    for yi, (tot, c) in enumerate(zip(left, comp)):
        ax.text(tot + 0.006, yi, f"{tot:.2f}  {c.replace('_', ' ')}", va="center", fontsize=6.5, color=INK2)
    clin = top["clinical_sjogren_target"].iloc[::-1]
    ax.set_yticks(range(len(C)), [f"{g} *" if c else g for g, c in zip(C.index, clin)])
    ax.set_xlim(0, left.max() * 1.3)
    ax.set_xlabel("weighted priority score (after abundance-marker penalty)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=7)
    ax.set_title(f"Top-30 targets, {mode} mode: contribution of each evidence layer\n"
                 "(right: compartment of origin in the HPA salivary-gland reference; * clinical-stage Sjögren target)",
                 pad=34)
    _save(fig, o, "fig07_ranking_contributions")


def fig_ranking_robustness(ctx, o, s7, rank) -> None:
    ev = pd.read_csv(s7 / "ranking_evaluation.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    ev = ev.sort_values("AUROC_clinical_targets")
    y = np.arange(len(ev))
    col = [SLOT[1] if r.startswith("priority") else SLOT[0] for r in ev["ranking"]]
    ax.barh(y, ev["AUROC_clinical_targets"] - 0.5, left=0.5, color=col, height=0.62, **_bars_gap())
    for yi, (a, k) in enumerate(zip(ev["AUROC_clinical_targets"], ev.iloc[:, 2])):
        ax.text(max(a, 0.5) + 0.01, yi, f"{a:.2f}  ({k} in top 50)", va="center", fontsize=6.5, color=INK2)
    ax.axvline(0.5, color=INK, lw=0.8)
    ax.set_yticks(y, [LAYER_LABEL.get(r, r) for r in ev["ranking"]])
    ax.set_xlim(0.2, 1.0)
    ax.set_xlabel("AUROC for recovering held-out clinical-stage Sjögren targets")
    ax.grid(axis="y", visible=False)
    n = int(ev["clinical_targets_in_universe"].iloc[0])
    ax.set_title(f"Held-out benchmark: {n} clinical targets\namong {int(ev['universe'].iloc[0])} candidates")
    _panel(ax, "a")

    ax = axes[1]
    t = rank.head(30).iloc[::-1]
    lo = (t["rank_median_sens"] - t["rank_iqr_sens"] / 2).clip(lower=1)
    hi = t["rank_median_sens"] + t["rank_iqr_sens"] / 2
    y = np.arange(len(t))
    ax.hlines(y, lo, hi, color=SLOT[0], lw=2.5)
    ax.scatter(t["rank"], y, s=22, color=SLOT[1], zorder=3, label="rank with configured weights")
    ax.scatter(t["rank_median_sens"], y, s=22, facecolor=SURFACE, edgecolor=SLOT[0], lw=1.2, zorder=3,
               label="median rank over 1,000 weight draws (bar: IQR)")
    for yi, p in enumerate(t["P_top20_sens"]):
        ax.text(hi.max() + 2, yi, f"P(top 20) {p:.2f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, t.index)
    ax.set_xlabel("rank")
    ax.set_xlim(0, hi.max() + 16)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.09), ncol=2, fontsize=6.5)
    ax.set_title("Weight sensitivity of the top 30\n(1,000 Dirichlet-perturbed weight vectors)")
    _panel(ax, "b")
    fig.tight_layout()
    _save(fig, o, "fig08_ranking_robustness")


def fig_single_cell(ctx, o, s8, rank) -> None:
    ex_p, sm_p = s8 / "scrna_celltype_expression.csv", s8 / "scrna_target_summary.csv"
    if not (ex_p.exists() and sm_p.exists()):
        return
    ex = pd.read_csv(ex_p)
    sm = pd.read_csv(sm_p, index_col=0)
    genes = [g for g in rank.head(20).index if g in set(ex["gene"])]
    ex = ex[(ex["gene"].isin(genes)) & (ex["cell_type"] != "unassigned")]
    order_ct = (ex.groupby(["cell_type", "compartment"])["n_cells"].first().reset_index()
                .sort_values(["compartment", "n_cells"], ascending=[True, False]))
    order_ct = order_ct[order_ct["n_cells"] >= 100]["cell_type"].tolist()
    ex = ex[ex["cell_type"].isin(order_ct)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2), gridspec_kw={"width_ratios": [2.3, 1]})
    ax = axes[0]
    xi = {c: i for i, c in enumerate(order_ct)}
    yi = {g: i for i, g in enumerate(genes[::-1])}
    sc = ax.scatter(ex["cell_type"].map(xi), ex["gene"].map(yi), s=ex["pct_expressing"].clip(upper=100) * 1.6 + 2,
                    c=ex["mean_logcp10k"], cmap=SEQ, vmin=0, vmax=np.nanpercentile(ex["mean_logcp10k"], 98),
                    edgecolor=MUTED, linewidth=0.3)
    ax.set_xticks(range(len(order_ct)), order_ct, rotation=50, ha="right")
    ax.set_yticks(range(len(genes)), genes[::-1])
    ax.set_xlim(-0.6, len(order_ct) - 0.4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.4, pad=0.01)
    cb.set_label("mean log CP10K")
    cb.outline.set_visible(False)
    for p in (10, 50, 90):
        ax.scatter([], [], s=p * 1.6 + 2, color=NEUTRAL, edgecolor=MUTED, lw=0.3, label=f"{p}% cells")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), title="expressing", title_fontsize=7, labelspacing=1.1)
    ax.set_title("Cell of origin of the top-20 targets in salivary glands (GSE272409, SjD + sicca)")
    _panel(ax, "a")

    ax = axes[1]
    comps = [c for c in ["epithelial", "stromal", "immune"] if f"{c}_logFC" in sm]
    L = sm.reindex(genes)[[f"{c}_logFC" for c in comps]].to_numpy(float)
    P = sm.reindex(genes)[[f"{c}_P" for c in comps]].to_numpy(float)
    v = np.nanpercentile(np.abs(L), 95)
    im = ax.imshow(L, cmap=DIV, vmin=-v, vmax=v, aspect="auto")
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            if np.isfinite(P[i, j]) and P[i, j] < 0.05:
                ax.text(j, i, "•" if P[i, j] >= 0.01 else "••", ha="center", va="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(comps)), comps)
    ax.set_yticks(range(len(genes)), genes)
    ax.grid(False)
    ax.set_facecolor(GRID)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.4, pad=0.02)
    cb.set_label("log2FC SjD vs sicca")
    cb.outline.set_visible(False)
    ax.set_title("Change inside each compartment\n(pseudobulk limma-voom; • P < 0.05, •• P < 0.01)")
    _panel(ax, "b")
    fig.tight_layout()
    _save(fig, o, "fig09_single_cell")


# ======================================================================================================
# models
# ======================================================================================================
def _to_prob(model: str, s: np.ndarray) -> np.ndarray:
    if model in PROBA_MODELS:
        return s
    if model == "linear_svm":
        return np.full_like(s, np.nan)                 # margins are not probabilities
    return 1.0 / (1.0 + np.exp(-s))                    # logistic regression, boosting, autoencoder: log-odds


def model_metrics(s5) -> tuple[pd.DataFrame, pd.DataFrame]:
    oof = pd.read_csv(s5 / "oof_scores.csv")
    lab = pd.read_csv(s5 / "model_input_labels.csv", index_col=0)
    rows = []
    for m, d in oof.groupby("model"):
        d = d.set_index("sample").loc[lab.index]
        y = lab["y"].to_numpy()
        for rcol in [c for c in d.columns if c.startswith("repeat_")]:
            s = d[rcol].to_numpy(float)
            thr = 0.5 if m in PROBA_MODELS else 0.0
            pred = (s > thr).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
            pr = _to_prob(m, s)
            rows.append({"model": m, "repeat": rcol, "AUROC": roc_auc_score(y, s), "AUPRC": average_precision_score(y, s),
                         "balanced_accuracy": balanced_accuracy_score(y, pred), "sensitivity": tp / (tp + fn),
                         "specificity": tn / (tn + fp), "F1": f1_score(y, pred, zero_division=0),
                         "MCC": matthews_corrcoef(y, pred), "Brier": brier_score_loss(y, pr) if np.isfinite(pr).all() else np.nan,
                         "TP": tp, "FP": fp, "TN": tn, "FN": fn})
    by_rep = pd.DataFrame(rows)
    metrics = ["AUROC", "AUPRC", "balanced_accuracy", "sensitivity", "specificity", "F1", "MCC", "Brier"]
    summ = by_rep.groupby("model")[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    return summ, by_rep


def fig_model_metrics(o, s5) -> pd.DataFrame:
    summ, by_rep = model_metrics(s5)
    summ.round(4).to_csv(o / "model_metrics.csv")
    by_rep.round(4).to_csv(o / "model_metrics_by_repeat.csv", index=False)
    metrics = [("AUROC", "AUROC"), ("AUPRC", "AUPRC"), ("balanced_accuracy", "Balanced accuracy"),
               ("sensitivity", "Sensitivity (recall)"), ("specificity", "Specificity"), ("F1", "F1 score"),
               ("MCC", "Matthews corr. coef."), ("Brier", "Brier score (lower = better)")]
    order = MODELS + ["dummy"]
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), sharey=True)
    for ax, (m, title) in zip(axes.ravel(), metrics):
        y = np.arange(len(order))[::-1]
        for yi, mod in zip(y, order):
            mu, sd = summ.loc[mod, f"{m}_mean"], summ.loc[mod, f"{m}_std"]
            if not np.isfinite(mu):
                ax.text(0.02, yi, "n/a (no probabilities)", transform=ax.get_yaxis_transform(), va="center", fontsize=6, color=MUTED)
                continue
            c = MODEL_COLOR.get(mod, MUTED)
            ax.errorbar(mu, yi, xerr=sd, fmt="o", ms=5.5, color=c, ecolor=c, elinewidth=1.4, capsize=0)
            ax.text(mu, yi + 0.32, f"{mu:.2f}", ha="center", fontsize=6, color=INK2)
        ax.set_title(title, fontsize=8)
        ax.set_yticks(y, [MODEL_LABEL[m_] for m_ in order])
        ax.grid(axis="y", visible=False)
        lo = 0 if m != "MCC" else min(-0.1, np.nanmin(summ[f"{m}_mean"]) - 0.1)
        ax.set_xlim(lo, 1.05 if m != "Brier" else max(0.3, np.nanmax(summ[f"{m}_mean"]) + 0.05))
    fig.suptitle("Nested cross-validation, GSE40611 pSS vs control (n = 35): out-of-fold metrics, mean ± SD over 5 repeats "
                 "of 5-fold CV\n(threshold: probability 0.5 or decision value 0; Brier only for models with probability-scale output)",
                 x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, o, "fig10_model_metrics")
    return summ


def fig_roc_pr(o, s5) -> None:
    oof = pd.read_csv(s5 / "oof_scores.csv")
    lab = pd.read_csv(s5 / "model_input_labels.csv", index_col=0)
    y = lab["y"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for m in MODELS + ["dummy"]:
        d = oof[oof["model"] == m].set_index("sample").loc[lab.index]
        reps = [c for c in d.columns if c.startswith("repeat_")]
        c = MODEL_COLOR.get(m, MUTED)
        ls = "--" if m == "dummy" else "-"
        grid = np.linspace(0, 1, 101)
        tprs = [np.interp(grid, *roc_curve(y, d[r])[:2]) for r in reps]
        auc = np.mean([roc_auc_score(y, d[r]) for r in reps])
        axes[0].plot(grid, np.mean(tprs, axis=0), color=c, ls=ls, label=f"{MODEL_LABEL[m]} ({auc:.3f})")
        precs = []
        for r in reps:
            p, rc, _ = precision_recall_curve(y, d[r])
            precs.append(np.interp(grid, rc[::-1], p[::-1]))
        ap = np.mean([average_precision_score(y, d[r]) for r in reps])
        axes[1].plot(grid, np.mean(precs, axis=0), color=c, ls=ls, label=f"{MODEL_LABEL[m]} ({ap:.3f})")
    axes[0].plot([0, 1], [0, 1], color=GRID, lw=1)
    axes[0].set_xlabel("false-positive rate (1 − specificity)")
    axes[0].set_ylabel("true-positive rate (sensitivity)")
    axes[0].set_title("ROC, out-of-fold (mean over 5 CV repeats), AUROC in legend")
    axes[1].axhline(y.mean(), color=GRID, lw=1)
    axes[1].set_xlabel("recall (sensitivity)")
    axes[1].set_ylabel("precision (PPV)")
    axes[1].set_title("Precision-recall, out-of-fold, AUPRC in legend")
    for ax, l in zip(axes, "ab"):
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.03)
        ax.legend(loc="lower right" if l == "a" else "lower left", fontsize=6.5)
        _panel(ax, l)
    fig.tight_layout()
    _save(fig, o, "fig11_roc_pr")


def fig_model_robustness(o, s5, summ) -> None:
    import json

    best = json.loads((s5 / "best_model.json").read_text())
    perm = pd.read_csv(s5 / "permutation_null.csv")
    lobo = pd.read_csv(s5 / "leave_one_batch_out.csv")
    scr = pd.read_csv(s5 / "preprocessing_screen.csv")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), gridspec_kw={"width_ratios": [1.2, 0.9, 1.2, 0.8]})
    ax = axes[0]
    ax.hist(perm["perm_AUROC"], bins=15, color=NEUTRAL, **_bars_gap(), label="shuffled labels (50)")
    ax.axvline(best["observed_perm_AUROC"], color=MODEL_COLOR[best["best_model"]], lw=2,
               label=f"observed {best['observed_perm_AUROC']:.3f}")
    ax.set_xlabel("AUROC (5-fold CV)")
    ax.set_ylabel("permutations")
    ax.set_xlim(0, 1.02)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2)
    ax.set_title(f"Permutation test, {MODEL_LABEL[best['best_model']]}\np = {best['permutation_p']:.3f}")
    _panel(ax, "a")

    ax = axes[1]
    x = np.arange(len(lobo))
    ax.bar(x, lobo["AUROC"], color=MODEL_COLOR[best["best_model"]], width=0.6, **_bars_gap())
    for xi, (a, n) in enumerate(zip(lobo["AUROC"], lobo["n_test"])):
        ax.text(xi, a + 0.02, f"{a:.2f}\nn={n}", ha="center", fontsize=6.5, color=INK2)
    ax.axhline(0.5, color=INK, ls="--", lw=0.8)
    ax.set_xticks(x, [f"batch {b}" for b in lobo["held_out_batch"]])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("AUROC on held-out batch")
    ax.grid(axis="x", visible=False)
    ax.set_title("Leave-one-batch-out")
    _panel(ax, "b")

    ax = axes[2]
    x = np.arange(len(scr))
    ax.bar(x - 0.2, scr["nested_cv_AUROC"], 0.4, color=SLOT[0], label="nested-CV AUROC", **_bars_gap())
    ax.bar(x + 0.2, scr["batch_pred_acc"], 0.4, color=SLOT[1], label="batch read-out accuracy", **_bars_gap())
    for xi, ch in zip(x, scr["chosen"]):
        if ch:
            ax.text(xi, 1.04, "chosen (label-free)", ha="center", fontsize=6.5, color=INK)
    ax.set_xticks(x, [m.replace("_", " ") for m in scr["matrix"]])
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=2, fontsize=6.5)
    ax.grid(axis="x", visible=False)
    ax.set_title("Batch handling for the ML matrix")
    _panel(ax, "c")

    ax = axes[3]
    _, by_rep = model_metrics(s5)
    r = by_rep[by_rep["model"] == best["best_model"]][["TN", "FP", "FN", "TP"]].sum()
    cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
    ax.imshow(cm, cmap=SEQ, vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm[i, j] / cm[i].sum():.0%})", ha="center", va="center", fontsize=7,
                    color=SURFACE if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_xticks([0, 1], ["pred. control", "pred. pSS"])
    ax.set_yticks([0, 1], ["control", "pSS"])
    ax.grid(False)
    ax.set_title("Confusion matrix (5 repeats pooled)")
    _panel(ax, "d")
    fig.tight_layout()
    _save(fig, o, "fig12_model_robustness")


def fig_external(ctx, o, s5) -> None:
    import json

    ext = pd.read_csv(s5 / "external_validation.csv")
    best = json.loads((s5 / "best_model.json").read_text())["best_model"]
    cohorts = [c for c in summ_order(ctx) if c in set(ext["cohort"])]
    A = ext.pivot_table(index="model", columns="cohort", values="AUROC").reindex(index=MODELS, columns=cohorts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    im = ax.imshow(A.to_numpy(), cmap=SEQ, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            v = A.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=SURFACE if v > 0.85 else INK)
    meta = ext.drop_duplicates("cohort").set_index("cohort")
    ax.set_xticks(range(len(cohorts)), [f"{c}\n{meta.loc[c, 'n_case']} vs {meta.loc[c, 'n_control']}" for c in cohorts], fontsize=6.5)
    ax.set_yticks(range(len(MODELS)), [MODEL_LABEL[m] for m in MODELS])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("AUROC (0.5 = chance)")
    cb.outline.set_visible(False)
    ax.set_title("External validation: trained on GSE40611, tested on each other cohort")
    _panel(ax, "a")

    ax = axes[1]
    b = ext[ext["model"] == best].set_index("cohort").loc[cohorts].iloc[::-1]
    y = np.arange(len(b))
    ax.errorbar(b["AUROC"], y, xerr=[b["AUROC"] - b["CI_low"], b["CI_high"] - b["AUROC"]], fmt="o", ms=5,
                color=MODEL_COLOR[best], elinewidth=1.4, capsize=0)
    ax.axvline(0.5, color=INK, ls="--", lw=0.8)
    notes = {"GSE23117": "batch confounded with diagnosis", "GSE40568": "n = 8", "GSE157159": "n = 17",
             "GSE173808": "controls = symptomatic sicca", "GSE127952": "low focus score; Agilent ratios"}
    ax.set_yticks(y, [f"{c}  ({b.loc[c, 'tissue']})" for c in b.index], fontsize=6.5)
    for yi, c in zip(y, b.index):
        if c in notes:
            ax.text(0.02, yi + 0.3, notes[c], transform=ax.get_yaxis_transform(), fontsize=6, color=MUTED)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Best model ({MODEL_LABEL[best]}) with uncertainty")
    _panel(ax, "b")
    fig.tight_layout()
    _save(fig, o, "fig13_external_validation")


THEME = {"Interferon response": ["IFI27", "MX1", "ISG15", "IFI6", "RSAD2", "BST2", "IFITM1", "GBP3", "IFI44L", "ISG20"],
         "Chemokines": ["CCL19", "CXCL9"], "Antigen presentation": ["HLA-A", "HLA-B", "HLA-F"],
         "B cells / dendritic cells": ["MS4A1", "LAMP3"], "Gland secretion (AQP5)": ["BMP6"]}


def fig_consensus(o, s7) -> None:
    """Agreement between the multi-evidence ranking and DeepMOLFT SHAP; tier-1 consensus genes by theme."""
    rk = pd.read_csv(s7 / "target_ranking_all.csv", index_col=0)
    cp = s7 / "target_consensus.csv"
    if not cp.exists() or "deepmolft_branch_rank" not in rk:
        return
    cons = pd.read_csv(cp, index_col=0)
    d = rk.dropna(subset=["deepmolft_branch_rank"])
    t1, t2 = cons.index[cons["consensus_tier"] == 1], cons.index[cons["consensus_tier"] == 2]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.axvspan(0.8, 100, ymin=0, ymax=1, color="#eef4fc", zorder=0)
    other = d.index.difference(t1.union(t2))
    ax.scatter(d.loc[other, "rank"], d.loc[other, "deepmolft_branch_rank"], s=10, color=NEUTRAL, lw=0, label="other candidates")
    ax.scatter(d.loc[t2, "rank"], d.loc[t2, "deepmolft_branch_rank"], s=30, color=SLOT[1], edgecolor=SURFACE, lw=0.6,
               label=f"consensus tier 2 ({len(t2)}): smaller effects")
    ax.scatter(d.loc[t1, "rank"], d.loc[t1, "deepmolft_branch_rank"], s=40, color=SLOT[0], edgecolor=SURFACE, lw=0.6,
               label=f"consensus tier 1 ({len(t1)}): robust, infiltration-independent")
    # hand-placed offsets for the crowded cluster around pipeline rank 30-60 (points)
    offs = {"IFITM1": (-34, -9), "HLA-B": (-6, 7), "HLA-F": (5, -8), "GBP3": (-30, 5), "HLA-A": (5, -9),
            "MS4A1": (6, 3), "IFI44L": (-38, 2), "LAMP3": (5, 2), "BST2": (5, -8), "ISG20": (5, 3)}
    for g in t1:
        ax.annotate(g, (d.loc[g, "rank"], d.loc[g, "deepmolft_branch_rank"]), xytext=offs.get(g, (4, 3)),
                    textcoords="offset points", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.axhline(50, color=MUTED, ls="--", lw=0.8)
    ax.axvline(100, color=MUTED, ls="--", lw=0.8)
    ax.set_xlabel("multi-evidence pipeline rank (log)")
    ax.set_ylabel("DeepMOLFT SHAP rank within its branch (log)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.32), ncol=1)
    ax.set_title("Two independent methods agree (shaded: pipeline top 100; dashed: DeepMOLFT top 50)")
    _panel(ax, "a")

    ax = axes[1]
    rows = [(g, th) for th, gs in THEME.items() for g in gs if g in set(t1)]
    rows += [(g, "Other") for g in t1 if g not in {r[0] for r in rows}]
    rows = rows[::-1]
    fc = [2 ** rk.loc[g, "meta_logFC"] for g, _ in rows]
    th_color = dict(zip(list(THEME) + ["Other"], SLOT[:len(THEME) + 1]))
    ax.barh(range(len(rows)), fc, color=[th_color[t] for _, t in rows], height=0.7, **_bars_gap())
    for i, ((g, _), v) in enumerate(zip(rows, fc)):
        ax.text(v + 0.1, i, f"{v:.1f}×  #{int(rk.loc[g, 'rank'])}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(rows)), [g for g, _ in rows])
    ax.set_xlim(0, max(fc) * 1.3)
    ax.set_xlabel("fold change in Sjögren’s glands (meta-analysis)   ·   #n = pipeline rank")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=th_color[t], lw=6, label=t) for t in THEME], loc="upper right", fontsize=6.5)
    ax.set_title(f"The {len(t1)} tier-1 consensus targets, by biological theme")
    _panel(ax, "b")
    fig.tight_layout()
    _save(fig, o, "fig15_consensus_targets")


def fig_explain(o, s6, de) -> None:
    dc = pd.read_csv(s6 / "deletion_curves.csv")
    models = list(dict.fromkeys(dc["model"]))
    fig, axes = plt.subplots(1, len(models) + 1, figsize=(4 * (len(models) + 1), 3.6))
    meth_color = {"attribution": SLOT[0], "f_score_train": SLOT[1], "random": MUTED}
    meth_label = {"attribution": "model attribution", "f_score_train": "training F-score", "random": "random genes"}
    for ax, m, l in zip(axes, models, "abcd"):
        d = dc[dc["model"] == m]
        for meth, dm in d.groupby("method"):
            key = meth if meth in meth_color else "attribution"
            lab = meth_label[key] + (f" ({meth.replace('_', ' ')})" if key == "attribution" else "")
            ax.plot(dm["k"], dm["AUROC"], marker="o", ms=4, color=meth_color[key],
                    ls="--" if key == "random" else "-", label=lab)
        ax.set_xlabel("top-k genes removed (set to training mean)")
        ax.set_ylabel("out-of-fold AUROC")
        ax.set_title(MODEL_LABEL.get(m, m), fontsize=8)
        ax.legend(loc="lower left", fontsize=6)
        _panel(ax, l)
    # most faithful explainer (largest drop vs random in the deletion test); consensus scores saturate near 1
    ds = pd.read_csv(s6 / "deletion_summary.csv")
    ds = ds[~ds["method"].isin(["random", "f_score_train"])]
    best = ds.sort_values("drop_vs_random_mean", ascending=False).iloc[0]["model"]
    gi = pd.read_csv(s6 / f"gene_importance_{best}.csv", index_col=0)
    ax = axes[-1]
    t = gi.head(20).iloc[::-1]
    col = [UP if v > 0 else DOWN for v in de["meta_logFC"].reindex(t.index).fillna(0)]
    ax.barh(range(len(t)), t["mean_abs_all_folds"], color=col, height=0.7, **_bars_gap())
    for yi, (v, f) in enumerate(zip(t["mean_abs_all_folds"], t["selection_freq"])):
        ax.text(v, yi, f"  {f:.0%}", va="center", fontsize=6, color=INK2)
    ax.set_yticks(range(len(t)), t.index)
    ax.set_xlim(0, t["mean_abs_all_folds"].max() * 1.25)
    ax.set_xlabel("mean |attribution| over 25 outer folds\n(label: % of folds in which the gene was selected)")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Line2D([], [], color=UP, lw=6, label="higher in pSS (meta)"),
                       Line2D([], [], color=DOWN, lw=6, label="lower in pSS (meta)")],
              loc="upper left", bbox_to_anchor=(0, -0.24), ncol=2, fontsize=6.5)
    ax.set_title(f"Top-20 genes, {MODEL_LABEL.get(best, best)} (most faithful)", fontsize=8)
    _panel(ax, "abcd"[len(models)])
    fig.suptitle("Explainability: a faithful attribution lowers AUROC faster than random gene removal", x=0.01, ha="left",
                 fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, o, "fig14_explainability")


# ======================================================================================================
def run(ctx: Context) -> None:
    _style()
    o = ctx.stage_dir(STAGE)
    s3, s4, s5, s6, s7, s8 = (ctx.stage_dir(s) for s in ("stage03_diffexpr", "stage04_pathways", "stage05_models",
                                                          "stage06_explain", "stage07_ranking", "stage08_validation"))
    de = pd.read_csv(s3 / "de_all_genes.csv", index_col=0)
    rank = pd.read_csv(s7 / "target_ranking_all.csv", index_col=0)
    steps = [("cohorts", lambda: fig_cohorts(ctx, o, s3)), ("DE quality", lambda: fig_de_quality(ctx, o, s3)),
             ("volcano", lambda: fig_volcano(ctx, o, de, rank)), ("heatmap", lambda: fig_heatmap(ctx, o, de, rank)),
             ("forest", lambda: fig_forest(ctx, o, de, rank)), ("pathways", lambda: fig_pathways(ctx, o, s4)),
             ("ranking", lambda: fig_ranking(ctx, o, rank)),
             ("ranking robustness", lambda: fig_ranking_robustness(ctx, o, s7, rank)),
             ("single cell", lambda: fig_single_cell(ctx, o, s8, rank)),
             ("consensus", lambda: fig_consensus(o, s7))]
    for name, fn in steps:
        fn()
        ctx.log.info("  figure: %s", name)
    summ = fig_model_metrics(o, s5)
    fig_roc_pr(o, s5)
    fig_model_robustness(o, s5, summ)
    fig_external(ctx, o, s5)
    fig_explain(o, s6, de)
    ctx.log.info("model metrics (out-of-fold, mean over repeats):\n%s",
                 summ[[c for c in summ.columns if c.endswith("_mean")]].round(3).to_string())
    ctx.log.info("stage 9 done -> %s (%d figures)", o, len(list(o.glob("*.png"))))

"""Figures for the DeepMOLFT task (validated palette; first three categorical slots are all-pairs safe)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

SLOT = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
MODEL_COLOR = {"DeepMOLFT": SLOT[0], "elasticnet": SLOT[1], "random_forest": SLOT[2]}
MODEL_LABEL = {"DeepMOLFT": "DeepMOLFT (late fusion + Optuna)", "elasticnet": "Elastic net", "random_forest": "Random forest"}
BRANCH_COLOR = {"immune": SLOT[1], "gland": SLOT[0], "broad": SLOT[2]}
BRANCH_LABEL = {"immune": "immune-cell genes", "gland": "gland (epithelial/stromal) genes", "broad": "broadly expressed genes"}


def _style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "font.size": 8, "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.5, "axes.axisbelow": True, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "legend.frameon": False, "legend.fontsize": 7,
                         "pdf.fonttype": 42})


def _save(fig, d, name):
    fig.savefig(d / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(d / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def make_figures(d, met, summ, cons, br, dl, pooled_shap, cfg):
    _style()
    cohorts = cfg["data"]["cohorts"]

    # 1. held-out AUROC per cohort and model
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    y = np.arange(len(cohorts))[::-1]
    for i, (m, off) in enumerate(zip(["DeepMOLFT", "elasticnet", "random_forest"], [0.22, 0, -0.22])):
        r = met[met["model"] == m].set_index("test_cohort").reindex(cohorts)
        wa = float(summ.set_index("model").loc[m, "weighted_AUROC"])
        ax.scatter(r["AUROC"], y + off, s=46, color=MODEL_COLOR[m], edgecolor=SURFACE, lw=0.8, zorder=3,
                   label=f"{MODEL_LABEL[m]} (n-weighted AUROC {wa:.3f})")
    n = met.drop_duplicates("test_cohort").set_index("test_cohort").reindex(cohorts)
    ax.set_yticks(y, [f"{c} (n={int(n.loc[c, 'n'])})" for c in cohorts])
    ax.axvline(0.5, color=INK, ls="--", lw=0.8)
    ax.set_xlim(0.2, 1.03)
    ax.set_xlabel("AUROC on the held-out cohort (never used for training, tuning or early stopping)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.42), ncol=1)
    ax.grid(axis="y", visible=False)
    ax.set_title("Leave-one-cohort-out validation")
    _save(fig, d, "deepmolft_fig1_loco_auroc")

    # 2. top-15 SHAP genes within each branch (late fusion: compare genes only within their branch)
    share = br.groupby("branch")["share_of_total_abs_shap"].mean()
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.2))
    for ax, b in zip(axes, ["gland", "immune", "broad"]):
        t = cons[cons["branch"] == b].sort_values("branch_rank").head(15).iloc[::-1]
        ax.barh(range(len(t)), t["within_branch_importance"], color=BRANCH_COLOR[b], height=0.72, edgecolor=SURFACE, lw=1)
        for i, (v, f, dirn, rk) in enumerate(zip(t["within_branch_importance"], t["folds_in_branch_top20"],
                                                 t["direction"], t["pipeline_rank"])):
            lab = f"  {'↑' if dirn == 'higher_in_SjS' else '↓'} {f}/{len(cohorts)}"
            lab += f" · #{int(rk)}" if pd.notna(rk) else ""
            ax.text(v, i, lab, va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(range(len(t)), t.index)
        ax.set_xlim(0, t["within_branch_importance"].max() * 1.75)
        ax.set_xlabel("share of the branch’s |SHAP| (mean over held-out cohorts)")
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{BRANCH_LABEL[b].capitalize()}\n(branch = {share.get(b, 0):.0%} of all |SHAP|)", fontsize=8.5)
    fig.suptitle("Top DeepMOLFT SHAP genes per input branch   (↑/↓ higher/lower in Sjögren’s · k/7 = folds in branch "
                 "top 20 · #n = main-pipeline target rank)", x=0.01, ha="left", fontsize=9, fontweight="bold")
    fig.tight_layout()
    _save(fig, d, "deepmolft_fig2_top_shap_genes")

    # 3. DeepMOLFT-style SHAP summary (beeswarm) on all held-out samples: top 7 genes of each branch
    genes = [g for b in ["gland", "immune", "broad"]
             for g in cons[cons["branch"] == b].sort_values("branch_rank").head(7).index]
    S = pd.DataFrame(0.0, index=[], columns=genes)
    V = pd.DataFrame(0.0, index=[], columns=genes)
    srows, vrows = [], []
    for test_cohort, branches, gdict, sv, Xte in pooled_shap:
        n_s = next(iter(sv.values())).shape[0]
        s_df = pd.DataFrame(np.nan, index=range(n_s), columns=genes)
        v_df = pd.DataFrame(np.nan, index=range(n_s), columns=genes)
        for b in branches:
            for j, g in enumerate(gdict[b]):
                if g in genes:
                    s_df[g] = sv[b][:, j]
                    v_df[g] = Xte[b][:, j]
        srows.append(s_df)
        vrows.append(v_df)
    S, V = pd.concat(srows, ignore_index=True).fillna(0.0), pd.concat(vrows, ignore_index=True).fillna(0.0)
    plt.figure()
    shap.summary_plot(S.to_numpy(), V.to_numpy(), feature_names=genes, show=False, max_display=20, plot_size=(6.4, 6.4))
    plt.title("SHAP summary on held-out cohorts (DeepMOLFT style)", fontsize=9, fontweight="bold", loc="left")
    _save(plt.gcf(), d, "deepmolft_fig3_shap_beeswarm")

    # 4. branch contribution + deletion test
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax = axes[0]
    w = br.pivot_table(index="test_cohort", columns="branch", values="share_of_total_abs_shap").reindex(cohorts)
    left = np.zeros(len(w))
    for b in ["gland", "immune", "broad"]:
        if b in w:
            ax.barh(range(len(w)), w[b], left=left, color=BRANCH_COLOR[b], height=0.65, edgecolor=SURFACE, lw=1,
                    label=BRANCH_LABEL[b])
            left += w[b].fillna(0).to_numpy()
    ax.set_yticks(range(len(w)), w.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of total |SHAP| per held-out cohort")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)
    ax.grid(axis="y", visible=False)
    ax.set_title("Which gene group does the model rely on?")
    ax = axes[1]
    for o, c, ls in [("SHAP", SLOT[0], "-"), ("random", MUTED, "--")]:
        m = dl[dl["ordering"] == o].groupby("k")["AUROC"].mean()
        ax.plot(m.index, m.values, marker="o", ms=4, color=c, ls=ls, label=f"remove top-k by {o}" if o == "SHAP" else "remove random k")
    ax.set_xlabel("k genes removed (set to training mean)")
    ax.set_ylabel("mean held-out AUROC (7 cohorts)")
    ax.legend(loc="lower left")
    ax.set_title("SHAP faithfulness (deletion test)")
    fig.tight_layout()
    _save(fig, d, "deepmolft_fig4_branches_deletion")

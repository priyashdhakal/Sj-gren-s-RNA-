"""Stage 6: which genes drive the models? SHAP, integrated gradients and ablation.

For every outer fold of stage 5 the tuned model is refitted on the training samples (hyper-parameters
are read from ``nested_cv_fold_results.csv``) and explained on the held-out samples:

  * linear models   closed-form linear SHAP  (coef * (x - background mean); identical to
                    ``shap.LinearExplainer`` with independent features)
  * tree models     ``shap.TreeExplainer``
  * autoencoder     Integrated Gradients (Captum) against the training-mean baseline

Aggregation across folds gives mean |attribution|, selection frequency and top-k stability per gene.

Validation of the explanations themselves
  * deletion curves: remove (set to the training mean) the top-k genes ranked by an attribution and
    follow the pooled out-of-fold AUROC; a faithful attribution drops faster than a random ordering
    and faster than a training-fold F-score ranking
  * single-gene ablation of the consensus top genes and group ablation (cell-type marker sets and
    co-expression clusters) with size-matched random-group nulls

Outputs (``outputs/stage06_explain``): gene_importance_consensus.csv, gene_importance_<model>.csv,
deletion_curves.csv, deletion_summary.csv, gene_ablation.csv, group_ablation.csv, figures/.
"""
from __future__ import annotations

import json
import warnings
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC

from .context import Context
from .deep_models import SupervisedAutoencoder
from .resources import marker_sets
from .stage05_models import model_zoo, score_vec
from .utils import load_json, rank_normalize, read_table

warnings.filterwarnings("ignore")
STAGE, S5, S3 = "stage06_explain", "stage05_models", "stage03_diffexpr"


# --------------------------------------------------------------------------
# attributions
# --------------------------------------------------------------------------
def attribute(clf, Xs_tr: np.ndarray, Xs_te: np.ndarray, ig_steps: int = 64):
    """Per-sample attributions in the scaled selected-feature space (positive = towards 'case')."""
    if isinstance(clf, SupervisedAutoencoder):
        import torch
        from captum.attr import IntegratedGradients

        ig = IntegratedGradients(lambda x: clf.net_.logit(x))
        x = torch.tensor(Xs_te, dtype=torch.float32)
        base = torch.zeros_like(x)                      # scaled training mean
        return ig.attribute(x, baselines=base, n_steps=ig_steps).detach().numpy(), "integrated_gradients"
    if isinstance(clf, LogisticRegression):
        return clf.coef_.ravel() * (Xs_te - Xs_tr.mean(axis=0)), "shap_linear"
    if isinstance(clf, SVC):
        return clf.coef_.ravel() * (Xs_te - Xs_tr.mean(axis=0)), "shap_linear"
    if isinstance(clf, (RandomForestClassifier, GradientBoostingClassifier)):
        import shap

        sv = shap.TreeExplainer(clf).shap_values(Xs_te)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        return sv, "shap_tree"
    raise TypeError(f"no explainer for {type(clf).__name__}")


def _auc(y, s):
    s = np.asarray(s, dtype=float)
    return roc_auc_score(y, s) if (len(set(y)) == 2 and not np.isnan(s).any()) else np.nan


def _pooled_auc(store: Dict, y: np.ndarray) -> Dict:
    """store[(model, method, k, repeat)] -> filled score vector -> AUROC."""
    return {key: _auc(y, v) for key, v in store.items()}


# --------------------------------------------------------------------------
def run(ctx: Context) -> None:
    cfg = ctx.cfg["explain"]
    out, s5, s3 = ctx.stage_dir(STAGE), ctx.stage_dir(S5), ctx.stage_dir(S3)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    seed, rng = ctx.seed, np.random.default_rng(ctx.seed)

    X_df = read_table(s5 / "model_input_matrix.parquet")
    lab = pd.read_csv(s5 / "model_input_labels.csv", index_col=0)
    X, y, ids, genes_all = X_df.to_numpy(), lab["y"].to_numpy(), lab.index.to_numpy(), np.asarray(X_df.columns)
    pos = {s: i for i, s in enumerate(ids)}
    folds = pd.read_csv(s5 / "nested_cv_fold_results.csv")
    summary = pd.read_csv(s5 / "nested_cv_summary.csv")
    best = load_json(s5 / "best_model.json")["best_model"]
    real = summary[summary["model"] != "dummy"]
    linear = real[real["model"].isin(["elasticnet", "l2_logreg", "linear_svm"])]
    to_explain = [best]
    if len(linear) and linear.iloc[0]["model"] not in to_explain:
        to_explain.append(linear.iloc[0]["model"])
    if "ae_mlp" in set(real["model"]) and "ae_mlp" not in to_explain:
        to_explain.append("ae_mlp")
    ctx.log.info("explaining models: %s", to_explain)

    zoo = model_zoo(seed, to_explain, X.shape[1])
    n_splits = ctx.cfg["models"]["outer_splits"]
    cache, attr_rows = [], []
    ks_all = [0, 1, 2, 3, 5, 8, 12, 20, 35, 50]
    del_store: Dict = {}
    n_rand = cfg["n_random_deletion"]

    # ---------------------------------------------------------------- pass 1
    for name in to_explain:
        pipe0, _ = zoo[name]
        fr = folds[folds["model"] == name]
        for _, r in fr.iterrows():
            tr = np.array([pos[s] for s in r["train"].split(";")])
            te = np.array([pos[s] for s in r["test"].split(";")])
            est = sklearn.clone(pipe0).set_params(**json.loads(r["best_params"])).fit(X[tr], y[tr])
            sel = est.named_steps["select"]
            mask = sel.get_support()
            genes = genes_all[mask]
            clf = est.named_steps["clf"]
            Xs_tr = est[:-1].transform(X[tr])
            Xs_te = est[:-1].transform(X[te])
            attr, method = attribute(clf, Xs_tr, Xs_te)
            imp = np.abs(attr).mean(axis=0)
            signed = attr.mean(axis=0)
            for g, a, sg in zip(genes, imp, signed):
                attr_rows.append({"model": name, "method": method, "fold": int(r["fold"]), "gene": g,
                                  "mean_abs": a, "mean_signed": sg})
            cache.append({"model": name, "method": method, "fold": int(r["fold"]), "repeat": int(r["repeat"]),
                          "clf": clf, "Xs_te": Xs_te, "te": te, "y_te": y[te], "genes": genes,
                          "importance": imp, "fscore": sel.scores_[mask]})

            # deletion curves: attribution vs training-fold F-score vs random order
            n_sel = len(genes)
            ks = [k for k in ks_all if k <= n_sel]
            orders = {method: np.argsort(-imp), "f_score_train": np.argsort(-np.nan_to_num(sel.scores_[mask]))}
            for k in ks:
                for mname, order in orders.items():
                    Z = Xs_te.copy()
                    Z[:, order[:k]] = 0.0
                    store = del_store.setdefault((name, mname, k, int(r["repeat"])), np.full(len(y), np.nan))
                    store[te] = score_vec(clf, Z)
                rs = np.zeros(len(te))
                for _ in range(n_rand):
                    Z = Xs_te.copy()
                    Z[:, rng.permutation(n_sel)[:k]] = 0.0
                    rs += score_vec(clf, Z) / n_rand
                store = del_store.setdefault((name, "random", k, int(r["repeat"])), np.full(len(y), np.nan))
                store[te] = rs
        ctx.log.info("  %-14s attributed over %d folds (%s)", name, len(fr), cache[-1]["method"])

    # ------------------------------------------------------- aggregate attributions
    attr_df = pd.DataFrame(attr_rows)
    attr_df.to_csv(out / "attributions_by_fold.csv", index=False)
    per_model = {}
    for name in to_explain:
        d = attr_df[attr_df["model"] == name]
        nf = d["fold"].nunique()
        top20 = d.groupby("fold", group_keys=False).apply(lambda x: x.nlargest(20, "mean_abs")).groupby("gene").size()
        g = d.groupby("gene").agg(mean_abs_selected=("mean_abs", "mean"), mean_signed=("mean_signed", "mean"),
                                  n_folds_selected=("fold", "nunique"))
        g["mean_abs_all_folds"] = d.groupby("gene")["mean_abs"].sum() / nf
        g["selection_freq"] = g["n_folds_selected"] / nf
        g["top20_freq"] = top20.reindex(g.index).fillna(0) / nf
        g["sign_consistency"] = d.groupby("gene")["mean_signed"].apply(lambda s: abs(np.sign(s).mean()))
        g["method"] = d["method"].iloc[0]
        g = g.sort_values("mean_abs_all_folds", ascending=False)
        g.to_csv(out / f"gene_importance_{name}.csv")
        per_model[name] = g

    # ---------------------------------------------------------------- deletion
    aucs = _pooled_auc(del_store, y)
    rows = [{"model": k[0], "method": k[1], "k": k[2], "repeat": k[3], "AUROC": v} for k, v in aucs.items()]
    dc = pd.DataFrame(rows).groupby(["model", "method", "k"], as_index=False)["AUROC"].mean()
    dc.to_csv(out / "deletion_curves.csv", index=False)
    ds = []
    for model, d in dc.groupby("model"):
        base = d[d["method"] == "random"].set_index("k")["AUROC"]
        for method, dm in d.groupby("method"):
            m = dm.set_index("k")["AUROC"]
            ds.append({"model": model, "method": method, "mean_AUROC_over_k": m.mean(),
                       "AUROC_at_k5": m.get(5, np.nan), "AUROC_at_k12": m.get(12, np.nan),
                       "drop_vs_random_mean": (base - m).mean()})
    dsum = pd.DataFrame(ds)
    dsum.to_csv(out / "deletion_summary.csv", index=False)
    ctx.log.info("deletion test (larger 'drop_vs_random' = more faithful):\n%s",
                 dsum.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, len(to_explain), figsize=(4.4 * len(to_explain), 3.6), squeeze=False)
    for ax, model in zip(axes[0], to_explain):
        for method, dm in dc[dc["model"] == model].groupby("method"):
            ax.plot(dm["k"], dm["AUROC"], marker="o", ms=3, label=method)
        ax.set_title(f"{model}: deletion of top-k genes", fontsize=9)
        ax.set_xlabel("k genes removed")
        ax.set_ylabel("pooled OOF AUROC")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "deletion_curves.png", dpi=150)
    plt.close(fig)

    # ----------------------------------------------------------------- consensus
    tab = pd.DataFrame(index=sorted(set().union(*[set(g.index) for g in per_model.values()])))
    for name, g in per_model.items():
        tab[f"{name}_{g['method'].iloc[0]}"] = g["mean_abs_all_folds"]
        tab[f"{name}_top20_freq"] = g["top20_freq"]
        tab[f"{name}_signed"] = g["mean_signed"]
    imp_cols = [c for c in tab.columns if not c.endswith(("_top20_freq", "_signed"))]
    for c in imp_cols:
        tab[c] = tab[c].fillna(0.0)
        tab[f"rank01_{c}"] = rank_normalize(tab[c])
    sign_cols = [c for c in tab.columns if c.endswith("_signed")]
    tab["direction"] = np.where(tab[sign_cols].fillna(0).mean(axis=1) >= 0, "up_in_case", "down_in_case")
    tab["sign_agreement"] = (np.sign(tab[sign_cols].replace(0, np.nan)).abs().mean(axis=1)
                             * (np.sign(tab[sign_cols]).replace(0, np.nan).nunique(axis=1) <= 1))
    tab["attribution_score"] = tab[[f"rank01_{c}" for c in imp_cols]].mean(axis=1)
    tab["stability"] = tab[[c for c in tab.columns if c.endswith("_top20_freq")]].fillna(0).mean(axis=1)

    # ------------------------------------------------------------ gene ablation
    top_genes = tab.sort_values("attribution_score", ascending=False).head(cfg["top_genes"]).index.tolist()

    def margin(entry, Z):
        return score_vec(entry["clf"], Z) * (2 * entry["y_te"] - 1)

    def ablate(gene_set, model, n_null=0):
        eff, null_eff, ab = [], [], {}
        for e in (c for c in cache if c["model"] == model):
            idx = [i for i, g in enumerate(e["genes"]) if g in gene_set]
            if not idx:
                continue
            base = margin(e, e["Xs_te"])
            Z = e["Xs_te"].copy()
            Z[:, idx] = 0.0
            eff.append(base - margin(e, Z))
            sc = ab.setdefault(e["repeat"], np.full(len(y), np.nan))
            sc[e["te"]] = score_vec(e["clf"], Z)
            for _ in range(n_null):
                j = rng.permutation(len(e["genes"]))[:len(idx)]
                Zr = e["Xs_te"].copy()
                Zr[:, j] = 0.0
                null_eff.append((base - margin(e, Zr)).mean())
        if not eff:
            return None
        allm = np.concatenate(eff)
        d_auc = np.nanmean([_auc(y, v) for v in ab.values()]) if ab else np.nan
        return {"delta_margin": float(allm.mean()), "n_folds_with_gene": len(eff),
                "null_mean": float(np.mean(null_eff)) if null_eff else np.nan,
                "null_sd": float(np.std(null_eff)) if null_eff else np.nan, "AUROC_after": float(d_auc)}

    g_rows = []
    for model in to_explain:
        for g in top_genes:
            r = ablate({g}, model)
            if r:
                g_rows.append({"model": model, "gene": g, **r})
    gab = pd.DataFrame(g_rows)
    gab.to_csv(out / "gene_ablation.csv", index=False)
    mean_abl = gab.groupby("gene")["delta_margin"].mean()
    tab["ablation_delta_margin"] = mean_abl.reindex(tab.index)
    tab["ablation_rank01"] = rank_normalize(tab["ablation_delta_margin"])

    # ----------------------------------------------------------- group ablation
    groups: Dict[str, List[str]] = {f"celltype:{k}": v for k, v in marker_sets(ctx).items()}
    sel_genes = tab.sort_values("attribution_score", ascending=False).head(150).index.tolist()
    sel_genes = [g for g in sel_genes if g in X_df.columns]
    corr = np.corrcoef(X_df[sel_genes].to_numpy().T)
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0)
    cl = fcluster(linkage(squareform(dist, checks=False), "average"), t=8, criterion="maxclust")
    for c in sorted(set(cl)):
        groups[f"coexpr:cluster{c}"] = [g for g, k in zip(sel_genes, cl) if k == c]
    tab["coexpr_cluster"] = pd.Series({g: f"cluster{k}" for g, k in zip(sel_genes, cl)})
    ga_rows = []
    for model in to_explain:
        for gname, members in groups.items():
            r = ablate(set(members), model, n_null=cfg.get("n_group_null", 10))
            if r and r["n_folds_with_gene"] >= 5:
                size = int(np.mean([sum(g in set(members) for g in e["genes"]) for e in cache if e["model"] == model]))
                r["z_vs_random"] = (r["delta_margin"] - r["null_mean"]) / (r["null_sd"] or np.nan)
                ga_rows.append({"model": model, "group": gname, "n_members_config": len(members),
                                "mean_members_selected": size, **r})
    ga = pd.DataFrame(ga_rows).sort_values(["model", "z_vs_random"], ascending=[True, False])
    ga.to_csv(out / "group_ablation.csv", index=False)
    ctx.log.info("group ablation (top by z vs size-matched random):\n%s",
                 ga.groupby("model").head(4)[["model", "group", "delta_margin", "null_mean", "z_vs_random"]]
                 .round(3).to_string(index=False))

    tab["consensus_score"] = tab[["attribution_score", "ablation_rank01"]].mean(axis=1, skipna=True)
    tab = tab.sort_values("consensus_score", ascending=False)
    tab.index.name = "gene"
    tab.to_csv(out / "gene_importance_consensus.csv")
    ctx.log.info("top consensus genes: %s", tab.head(25).index.tolist())

    t = tab.head(cfg["top_genes"])
    fig, ax = plt.subplots(figsize=(6, 0.28 * len(t) + 1.2))
    ax.barh(range(len(t)), t["consensus_score"], color=["#d62728" if d == "up_in_case" else "#1f77b4" for d in t["direction"]])
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("consensus attribution score (red: up in case)")
    fig.tight_layout()
    fig.savefig(fig_dir / "top_consensus_genes.png", dpi=150)
    plt.close(fig)
    ctx.log.info("stage 6 done -> %s", out)

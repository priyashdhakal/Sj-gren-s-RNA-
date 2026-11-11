"""Stage 5: baseline machine learning and deep learning with nested cross-validation.

Design
  * Cohort: case vs control of the discovery dataset (GSE40611 pSS vs Control, n = 35, RMA).
  * Phase A -- the input matrix (none / removeBatchEffect / ComBat on RMA data) is chosen in stage 2 WITHOUT
    labels (batch least readable from expression). A nested-CV + leave-one-batch-out screen of all three is
    reported for transparency but not used for the choice (choosing by AUROC would bias the estimate).
  * Phase B -- model zoo on the chosen matrix: dummy, elastic-net, L2 logistic, linear SVM,
    random forest, gradient boosting and a supervised autoencoder (torch).  Every model is wrapped
    in ``SelectKBest -> StandardScaler -> estimator`` so feature selection and scaling happen inside
    each training fold.  Hyper-parameters are tuned by an inner stratified CV (grid search) in every
    outer fold: outer = repeated stratified k-fold, inner = stratified k-fold.
  * The best model (outer-fold AUROC, ties by AUPRC) gets a label-permutation null and a
    leave-one-batch-out test; every model is tested externally on all other gland cohorts.
  * Models are supporting evidence for target ranking (low weight in stage 7).

Outputs (``outputs/stage05_models``): see the file list at the end of ``run``.
"""
from __future__ import annotations

import json
import warnings
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
                                     permutation_test_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .context import Context
from .deep_models import SupervisedAutoencoder
from .preprocessing import batch_predictability, combat, remove_batch_effect
from .utils import dump_json, read_table, write_table

STAGE = "stage05_models"
S2 = "stage02_preprocess"

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# model zoo
# --------------------------------------------------------------------------
def _logreg(seed: int, l1_ratio: float | None = None) -> LogisticRegression:
    """Version-aware elastic-net / L2 logistic regression (sklearn >= 1.8 deprecates `penalty`)."""
    if l1_ratio is None:
        return LogisticRegression(max_iter=5000, random_state=seed)
    ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if ver >= (1, 8):
        return LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=5000, random_state=seed)
    return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000, random_state=seed)


def _pipe(est) -> Pipeline:
    return Pipeline([("select", SelectKBest(f_classif, k=50)), ("scale", StandardScaler()), ("clf", est)])


def model_zoo(seed: int, names: List[str], n_features: int) -> Dict[str, Tuple[Pipeline, dict]]:
    ks = [k for k in (20, 50, 200, 1000) if k <= n_features]
    zoo = {
        "dummy": (_pipe(DummyClassifier(strategy="prior")), {}),
        "elasticnet": (_pipe(_logreg(seed, 0.5)),
                       {"select__k": ks, "clf__C": [0.01, 0.1, 1.0], "clf__l1_ratio": [0.2, 0.5, 0.8]}),
        "l2_logreg": (_pipe(_logreg(seed)), {"select__k": ks, "clf__C": [0.001, 0.01, 0.1, 1.0]}),
        "linear_svm": (_pipe(SVC(kernel="linear", random_state=seed)),
                       {"select__k": ks, "clf__C": [0.001, 0.01, 0.1, 1.0]}),
        "random_forest": (_pipe(RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=1)),
                          {"select__k": [k for k in ks if k <= 200], "clf__max_features": ["sqrt", 0.3],
                           "clf__min_samples_leaf": [1, 2]}),
        "grad_boosting": (_pipe(GradientBoostingClassifier(n_estimators=100, subsample=0.8, random_state=seed)),
                          {"select__k": [k for k in ks if k <= 200], "clf__max_depth": [1, 2],
                           "clf__learning_rate": [0.05, 0.1]}),
        "ae_mlp": (_pipe(SupervisedAutoencoder(seed=seed)),
                   {"select__k": [k for k in ks if k <= 200], "clf__latent": [4, 8],
                    "clf__recon_weight": [0.1, 1.0], "clf__weight_decay": [1e-3, 1e-2]}),
    }
    return {k: zoo[k] for k in names if k in zoo}


def score_vec(est, X) -> np.ndarray:
    return est.decision_function(X) if hasattr(est, "decision_function") else est.predict_proba(X)[:, 1]


def fit_tuned(pipe: Pipeline, grid: dict, X, y, inner: int, seed: int, n_jobs: int = 1):
    if not grid:
        return pipe.fit(X, y), {}
    gs = GridSearchCV(pipe, grid, cv=StratifiedKFold(inner, shuffle=True, random_state=seed), scoring="roc_auc",
                      n_jobs=n_jobs, refit=True, error_score=np.nan)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


# --------------------------------------------------------------------------
# evaluation helpers
# --------------------------------------------------------------------------
def nested_cv(name: str, pipe: Pipeline, grid: dict, X: np.ndarray, y: np.ndarray, ids: np.ndarray,
              outer: RepeatedStratifiedKFold, inner: int, seed: int, n_jobs: int):
    """Returns (fold_rows, oof dict {repeat: scores}) -- hyper-parameters are tuned inside each outer fold."""
    n_splits = outer.cvargs["n_splits"]
    rows, oof = [], {}
    for f, (tr, te) in enumerate(outer.split(X, y)):
        rep = f // n_splits
        est, best = fit_tuned(pipe, grid, X[tr], y[tr], inner, seed + f, n_jobs)
        s = score_vec(est, X[te])
        oof.setdefault(rep, np.full(len(y), np.nan))[te] = s
        rows.append({"model": name, "fold": f, "repeat": rep,
                     "AUROC": roc_auc_score(y[te], s), "AUPRC": average_precision_score(y[te], s),
                     "balanced_acc": balanced_accuracy_score(y[te], est.predict(X[te])),
                     "best_params": json.dumps(best, default=str),
                     "train": ";".join(ids[tr]), "test": ";".join(ids[te])})
    return rows, oof


def bootstrap_auc(y: np.ndarray, oof: Dict[int, np.ndarray], n_boot: int, seed: int) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    reps = list(oof.values())
    point = float(np.mean([roc_auc_score(y, s) for s in reps]))
    vals = []
    for _ in range(n_boot):
        b = rng.integers(0, len(y), len(y))
        if len(set(y[b])) < 2:
            continue
        vals.append(np.mean([roc_auc_score(y[b], s[b]) for s in reps]))
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def lobo(pipe: Pipeline, grid: dict, X: np.ndarray, y: np.ndarray, batch: np.ndarray, inner: int, seed: int) -> pd.DataFrame:
    rows = []
    for b in sorted(set(batch)):
        te = batch == b
        tr = ~te
        if len(set(y[te])) < 2 or min(np.bincount(y[tr])) < inner:
            continue
        est, _ = fit_tuned(pipe, grid, X[tr], y[tr], inner, seed)
        rows.append({"held_out_batch": b, "n_test": int(te.sum()), "AUROC": roc_auc_score(y[te], score_vec(est, X[te]))})
    return pd.DataFrame(rows)


def top_variable(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df[df.var(axis=0).nlargest(min(n, df.shape[1])).index]


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    sd = df.std(axis=0, ddof=0).replace(0, 1.0)
    return (df - df.mean(axis=0)) / sd


# --------------------------------------------------------------------------
def run(ctx: Context) -> None:
    cfg = ctx.cfg["models"]
    out, s2 = ctx.stage_dir(STAGE), ctx.stage_dir(S2)
    an = ctx.cfg["analysis"]
    acc = an["discovery_dataset"]
    seed, n_jobs = ctx.seed, ctx.n_jobs

    dcfg = ctx.cfg["datasets"][acc]
    case, ctrl = dcfg["case"], dcfg["control"]
    meta = pd.read_csv(s2 / f"{acc}_samples_final.csv", index_col=0)
    meta = meta[meta["group"].isin([case, ctrl])]
    ids = meta.index.to_numpy()
    y = (meta["group"] == case).astype(int).to_numpy()
    batch = meta["batch"].astype(str).to_numpy()
    expr = read_table(s2 / f"{acc}_gene_expr.parquet").loc[meta.index]
    ctx.log.info("cohort %s %s vs %s: n=%d (%d/%d), technical batches %s", acc, case, ctrl, len(y), y.sum(),
                 (1 - y).sum(), dict(Counter(batch)))

    # ------------------------------------------------------------------ phase A
    # The matrix is chosen label-free in stage 2 (batch least readable); the screen below is reported only.
    variants = {"none": expr, "remove_batch_effect": remove_batch_effect(expr, batch), "combat": combat(expr, batch)}
    chosen = pd.read_csv(s2 / f"{acc}_cohort_samples.csv", index_col=0)["batch_method"].iloc[0]
    outer_a = RepeatedStratifiedKFold(n_splits=cfg["outer_splits"], n_repeats=3, random_state=seed)
    screen = []
    for vname, mat in variants.items():
        Xv = top_variable(mat, cfg["top_variable_genes"])
        pipe, grid = model_zoo(seed, ["l2_logreg"], Xv.shape[1])["l2_logreg"]
        rows, oof = nested_cv("l2_logreg", pipe, grid, Xv.to_numpy(), y, ids, outer_a, cfg["inner_splits"], seed, n_jobs)
        lo = lobo(pipe, grid, Xv.to_numpy(), y, batch, cfg["inner_splits"], seed)
        auc = float(np.mean([r["AUROC"] for r in rows]))
        screen.append({"matrix": vname, "nested_cv_AUROC": auc,
                       "nested_cv_AUROC_sd": float(np.std([r["AUROC"] for r in rows])),
                       "LOBO_AUROC_mean": float(lo["AUROC"].mean()) if len(lo) else np.nan,
                       "LOBO_AUROC_min": float(lo["AUROC"].min()) if len(lo) else np.nan,
                       "batch_pred_acc": batch_predictability(Xv, batch), "n_genes": Xv.shape[1]})
        ctx.log.info("  screen %-20s nested AUROC %.3f | LOBO mean %.3f | batch readable (acc) %.2f", vname, auc,
                     screen[-1]["LOBO_AUROC_mean"], screen[-1]["batch_pred_acc"])
    sc = pd.DataFrame(screen)
    sc["chosen"] = sc["matrix"] == chosen
    sc.to_csv(out / "preprocessing_screen.csv", index=False)
    ctx.log.info("ML matrix: %s (chosen label-free in stage 2; screen AUROCs are not used for the choice)", chosen)

    X_df = top_variable(variants[chosen], cfg["top_variable_genes"])
    X = X_df.to_numpy()
    write_table(X_df, out / "model_input_matrix.parquet")
    pd.DataFrame({"y": y, "group": meta["group"].to_numpy(), "batch": batch}, index=ids).to_csv(out / "model_input_labels.csv")

    # ------------------------------------------------------------------ phase B
    outer = RepeatedStratifiedKFold(n_splits=cfg["outer_splits"], n_repeats=cfg["outer_repeats"], random_state=seed)
    zoo = model_zoo(seed, cfg["run"], X.shape[1])
    fold_rows, oofs, summ = [], {}, []
    for name, (pipe, grid) in zoo.items():
        rows, oof = nested_cv(name, pipe, grid, X, y, ids, outer, cfg["inner_splits"], seed, n_jobs)
        fold_rows += rows
        oofs[name] = oof
        d = pd.DataFrame(rows)
        pooled, lo, hi = bootstrap_auc(y, oof, cfg["bootstrap"], seed)
        summ.append({"model": name, "AUROC_mean": d["AUROC"].mean(), "AUROC_sd": d["AUROC"].std(),
                     "AUPRC_mean": d["AUPRC"].mean(), "balanced_acc_mean": d["balanced_acc"].mean(),
                     "pooled_oof_AUROC": pooled, "CI95_low": lo, "CI95_high": hi})
        ctx.log.info("  nested CV %-14s AUROC %.3f +- %.3f | AUPRC %.3f | pooled OOF %.3f [%.3f, %.3f]", name,
                     summ[-1]["AUROC_mean"], summ[-1]["AUROC_sd"], summ[-1]["AUPRC_mean"], pooled, lo, hi)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / "nested_cv_fold_results.csv", index=False)
    summary = pd.DataFrame(summ).sort_values(["AUROC_mean", "AUPRC_mean"], ascending=False).reset_index(drop=True)
    summary.to_csv(out / "nested_cv_summary.csv", index=False)
    oof_df = pd.concat({m: pd.DataFrame(o, index=ids).add_prefix("repeat_") for m, o in oofs.items()}, names=["model", "sample"])
    oof_df.to_csv(out / "oof_scores.csv")

    # --------------------------------------------------------------- best model
    real = summary[summary["model"] != "dummy"]
    best = real.iloc[0]["model"]
    pipe_b, grid_b = zoo[best]
    modal = {}
    if grid_b:
        params = [json.loads(p) for p in folds.loc[folds["model"] == best, "best_params"]]
        for k in grid_b:
            modal[k] = Counter(json.dumps(p.get(k)) for p in params).most_common(1)[0][0]
            modal[k] = json.loads(modal[k])
    ctx.log.info("BEST MODEL: %s (outer AUROC %.3f); modal hyper-parameters %s", best, real.iloc[0]["AUROC_mean"], modal)
    best_pipe = sklearn.clone(pipe_b).set_params(**modal)

    # permutation null (hyper-parameters fixed at the modal values)
    obs, perm_scores, pval = permutation_test_score(best_pipe, X, y, scoring="roc_auc",
                                                    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=seed),
                                                    n_permutations=cfg["n_permutation_null"], random_state=seed,
                                                    n_jobs=n_jobs)
    pd.DataFrame({"perm_AUROC": perm_scores}).to_csv(out / "permutation_null.csv", index=False)
    ctx.log.info("permutation test: observed AUROC %.3f vs null mean %.3f (max %.3f), p = %.4f", obs,
                 perm_scores.mean(), perm_scores.max(), pval)

    lo_b = lobo(pipe_b, grid_b, X, y, batch, cfg["inner_splits"], seed)
    lo_b.to_csv(out / "leave_one_batch_out.csv", index=False)
    ctx.log.info("leave-one-batch-out AUROC: %s", lo_b.round(3).to_dict("records"))

    # ----------------------------------------------------------- external tests (every other gland cohort)
    ext_rows = []
    for e_acc, e_cfg in ctx.datasets():
        if e_acc == acc or e_cfg["type"] == "scrna_10x":
            continue
        em = pd.read_csv(s2 / f"{e_acc}_samples_final.csv", index_col=0)
        em = em[em["group"].isin([e_cfg["case"], e_cfg["control"]])]
        ex = read_table(s2 / f"{e_acc}_gene_expr.parquet").loc[em.index]
        ye = (em["group"] == e_cfg["case"]).astype(int).to_numpy()
        shared = [g for g in X_df.columns if g in ex.columns]
        Xtr_z, Xex_z = zscore(X_df[shared]), zscore(ex[shared])     # per-dataset gene standardisation
        for name, (pipe, grid) in zoo.items():
            if name == "dummy":
                continue
            est, _ = fit_tuned(pipe, grid, Xtr_z.to_numpy(), y, cfg["inner_splits"], seed, n_jobs)
            s = score_vec(est, Xex_z.to_numpy())
            rng = np.random.default_rng(seed)
            bs = [roc_auc_score(ye[i], s[i]) for i in (rng.integers(0, len(ye), len(ye)) for _ in range(500))
                  if len(set(ye[i])) == 2]
            ext_rows.append({"cohort": e_acc, "tissue": e_cfg["tissue"], "control": e_cfg["control"], "model": name,
                             "genes_shared": len(shared), "n_case": int(ye.sum()), "n_control": int((1 - ye).sum()),
                             "AUROC": roc_auc_score(ye, s), "CI_low": np.percentile(bs, 2.5), "CI_high": np.percentile(bs, 97.5)})
        best_ext = [r for r in ext_rows if r["cohort"] == e_acc and r["model"] == best][0]
        ctx.log.info("  external %-10s (%s, %d vs %d): %s AUROC %.3f [%.2f, %.2f]", e_acc, e_cfg["tissue"],
                     best_ext["n_case"], best_ext["n_control"], best, best_ext["AUROC"], best_ext["CI_low"], best_ext["CI_high"])
    pd.DataFrame(ext_rows).to_csv(out / "external_validation.csv", index=False)

    # ----------------------------------------------------------------- figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name in oofs:
        s = oofs[name][0]
        fpr, tpr, _ = roc_curve(y, s)
        ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, s):.2f})", lw=1.4)
    ax[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    ax[0].set_xlabel("FPR")
    ax[0].set_ylabel("TPR")
    ax[0].set_title(f"Out-of-fold ROC, repeat 0 ({chosen})", fontsize=9)
    ax[0].legend(fontsize=7)
    sm = summary.sort_values("AUROC_mean")
    ax[1].barh(sm["model"], sm["AUROC_mean"], xerr=[sm["AUROC_mean"] - sm["CI95_low"].clip(upper=sm["AUROC_mean"]),
                                                      (sm["CI95_high"] - sm["AUROC_mean"]).clip(lower=0)], color="#4c72b0")
    ax[1].set_xlim(0.4, 1.02)
    ax[1].set_xlabel("nested-CV AUROC (mean; bars = bootstrap 95% CI of pooled OOF)")
    fig.tight_layout()
    fig.savefig(out / "model_comparison.png", dpi=150)
    plt.close(fig)

    dump_json({"matrix": chosen, "best_model": best, "modal_params": modal, "grid": {k: v for k, v in grid_b.items()},
               "permutation_p": float(pval), "observed_perm_AUROC": float(obs), "n_samples": int(len(y)),
               "outer": f"{cfg['outer_splits']}-fold x {cfg['outer_repeats']} repeats",
               "inner": f"{cfg['inner_splits']}-fold"}, out / "best_model.json")
    ctx.log.info("stage 5 done -> %s", out)

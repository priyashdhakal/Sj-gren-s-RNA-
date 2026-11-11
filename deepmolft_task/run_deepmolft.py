#!/usr/bin/env python
"""DeepMOLFT on Sjogren's salivary-gland RNA: leave-one-cohort-out training, SHAP target identification.

    python run_deepmolft.py            # full run (outputs/)
    python run_deepmolft.py --quick    # smoke test: 2 Optuna trials, 1 seed, 40 epochs (outputs_quick/)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import deepmolft_sjs  # noqa: F401  (imports torch before shap)
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, matthews_corrcoef, roc_auc_score)

from deepmolft_sjs.data import build_fold, load_pooled
from deepmolft_sjs.explain import deep_shap
from deepmolft_sjs.figures import make_figures
from deepmolft_sjs.train import fit_baseline, fit_deepmolft, inner_split, predict, tune

ROOT = Path(__file__).resolve().parent


def metrics(y, p) -> dict:
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"n": len(y), "n_case": int(y.sum()), "n_control": int((1 - y).sum()), "AUROC": roc_auc_score(y, p),
            "AUPRC": average_precision_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, pred),
            "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "F1": f1_score(y, pred, zero_division=0), "MCC": matthews_corrcoef(y, pred), "Brier": brier_score_loss(y, p)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(ROOT / "config.yaml", encoding="utf-8"))
    if a.quick:
        cfg["optuna"]["n_trials"], cfg["validation"]["final_seeds"], cfg["model"]["epochs"] = 2, 1, 40
        cfg["outputs"] = "outputs_quick"
    out = ROOT / cfg["outputs"]
    (out / "figures").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(out / "run.log", mode="w")])
    log = logging.getLogger("deepmolft")
    logging.getLogger("fontTools").setLevel(logging.WARNING)      # PDF font subsetting chatter
    torch.set_num_threads(int(cfg["torch_threads"]))
    seed = int(cfg["seed"])

    p = load_pooled(cfg, ROOT)
    log.info("pooled data: %d samples x %d genes (in all %d cohorts); %d Sjogren's / %d controls; branches %s",
             len(p.y), p.X.shape[1], len(cfg["data"]["cohorts"]), p.y.sum(), (1 - p.y).sum(),
             p.branch.value_counts().to_dict())

    pred_rows, metric_rows, shap_rows, branch_rows, del_rows, hp_rows = [], [], [], [], [], []
    pooled_shap = []
    for k, test_cohort in enumerate(cfg["data"]["cohorts"]):
        t0 = time.time()
        te = np.where(p.cohort == test_cohort)[0]
        tr_all = np.where(p.cohort != test_cohort)[0]
        fd = build_fold(p, cfg, tr_all, te)
        branches = [b for b in fd.branches if len(fd.genes[b]) >= 2]
        ytr, yte, gtr = p.y[tr_all], p.y[te], p.group[tr_all]
        log.info("[fold %d] test %s (n=%d) | train n=%d | genes per branch %s", k + 1, test_cohort, len(te), len(tr_all),
                 {b: len(fd.genes[b]) for b in branches})

        # ---- Optuna on an inner split of the training cohorts -------------------------------
        study = tune(fd.train, ytr, gtr, branches, cfg, seed + k)
        hp = study.best_params
        hp_rows.append({"test_cohort": test_cohort, "inner_val_AUROC": study.best_value, **hp})
        log.info("  optuna best inner-val AUROC %.3f | %s", study.best_value, {kk: round(v, 4) for kk, v in hp.items()})

        # ---- final ensemble (different inner splits / seeds) -------------------------------------
        probs, shaps = [], []
        rng = np.random.default_rng(seed + k)
        bg_idx = rng.choice(len(tr_all), min(int(cfg["shap"]["background"]), len(tr_all)), replace=False)
        bg = {b: fd.train[b][bg_idx] for b in branches}
        models = []
        for s in range(int(cfg["validation"]["final_seeds"])):
            itr, iva = inner_split(ytr, gtr, cfg["validation"]["inner_val_fraction"], seed + 100 * k + s)
            m, _ = fit_deepmolft(fd.train, ytr, itr, iva, branches, hp, cfg["model"], seed + 100 * k + s)
            models.append(m)
            probs.append(predict(m, fd.test, branches))
            shaps.append(deep_shap(m, bg, fd.test, branches))
        prob = np.mean(probs, axis=0)
        sv = {b: np.mean([s_[b] for s_ in shaps], axis=0) for b in branches}

        # ---- baselines on the same fold features --------------------------------------------
        Xtr_c = np.hstack([fd.train[b] for b in branches])
        Xte_c = np.hstack([fd.test[b] for b in branches])
        preds = {"DeepMOLFT": prob}
        for bname in cfg["validation"]["baselines"]:
            est = fit_baseline(bname, Xtr_c, ytr, gtr, seed)
            preds[bname] = est.predict_proba(Xte_c)[:, 1]
        for mname, pr in preds.items():
            metric_rows.append({"test_cohort": test_cohort, "model": mname, **metrics(yte, pr)})
            pred_rows += [{"sample": s_, "test_cohort": test_cohort, "model": mname, "y": int(yy), "prob": float(pp)}
                          for s_, yy, pp in zip(p.X.index[te], yte, pr)]
        log.info("  held-out AUROC: %s", {m_: round(roc_auc_score(yte, pr), 3) for m_, pr in preds.items()})

        # ---- SHAP summaries -----------------------------------------------------------------------
        total = sum(np.abs(sv[b]).mean(axis=0).sum() for b in branches)
        for b in branches:
            mabs = np.abs(sv[b]).mean(axis=0)
            for g, v, s_, x in zip(fd.genes[b], mabs, sv[b].T, fd.test[b].T):
                c = np.corrcoef(x, s_)[0, 1] if np.std(x) > 0 and np.std(s_) > 0 else 0.0
                shap_rows.append({"test_cohort": test_cohort, "gene": g, "branch": b, "mean_abs_shap": float(v),
                                  "norm_importance": float(v / total), "direction_corr": float(c)})
            branch_rows.append({"test_cohort": test_cohort, "branch": b, "n_genes": len(fd.genes[b]),
                                "share_of_total_abs_shap": float(np.abs(sv[b]).mean(axis=0).sum() / total)})
        pooled_shap.append((test_cohort, branches, fd.genes, sv, fd.test))
        np.savez_compressed(out / f"shap_heldout_{test_cohort}.npz",
                            **{f"shap_{b}": sv[b] for b in branches}, **{f"x_{b}": fd.test[b] for b in branches},
                            **{f"genes_{b}": np.array(fd.genes[b]) for b in branches},
                            samples=np.array(p.X.index[te]), y=yte)

        # ---- deletion (faithfulness): remove top-k SHAP genes vs random genes ----------------------
        flat = [(b, j, float(np.abs(sv[b][:, j]).mean())) for b in branches for j in range(len(fd.genes[b]))]
        order = sorted(range(len(flat)), key=lambda i: -flat[i][2])

        def masked_auc(idx_list):
            Xm = {b: fd.test[b].copy() for b in branches}
            for i in idx_list:
                b, j, _ = flat[i]
                Xm[b][:, j] = 0.0                      # = training mean after scaling
            return roc_auc_score(yte, np.mean([predict(m, Xm, branches) for m in models], axis=0))

        for kk in cfg["shap"]["deletion_k"]:
            kk = min(kk, len(flat))
            del_rows.append({"test_cohort": test_cohort, "k": kk, "ordering": "SHAP", "AUROC": masked_auc(order[:kk])})
            rnd = [masked_auc(list(rng.choice(len(flat), kk, replace=False))) for _ in range(int(cfg["shap"]["deletion_random"]))]
            del_rows.append({"test_cohort": test_cohort, "k": kk, "ordering": "random", "AUROC": float(np.mean(rnd))})
        log.info("  fold done in %.0fs", time.time() - t0)

    # ---- save per-fold tables ---------------------------------------------------------------------
    met = pd.DataFrame(metric_rows)
    met.to_csv(out / "loco_metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(out / "loco_predictions.csv", index=False)
    pd.DataFrame(hp_rows).to_csv(out / "optuna_best_params.csv", index=False)
    sh = pd.DataFrame(shap_rows)
    sh.to_csv(out / "shap_by_fold.csv", index=False)
    br = pd.DataFrame(branch_rows)
    br.to_csv(out / "branch_contribution.csv", index=False)
    dl = pd.DataFrame(del_rows)
    dl.to_csv(out / "deletion_test.csv", index=False)

    pr = pd.DataFrame(pred_rows)
    summ = []
    for mname, d in pr.groupby("model"):
        m_ = met[met["model"] == mname]
        w = m_["n"] / m_["n"].sum()
        summ.append({"model": mname, "mean_AUROC": m_["AUROC"].mean(), "weighted_AUROC": float((m_["AUROC"] * w).sum()),
                     "median_AUROC": m_["AUROC"].median(), "pooled_AUROC": roc_auc_score(d["y"], d["prob"]),
                     **{f"pooled_{k_}": v for k_, v in metrics(d["y"].to_numpy(), d["prob"].to_numpy()).items()
                        if k_ not in ("n", "n_case", "n_control", "AUROC")}})
    summ = pd.DataFrame(summ).sort_values("weighted_AUROC", ascending=False)
    summ.to_csv(out / "loco_summary.csv", index=False)
    log.info("leave-one-cohort-out summary:\n%s", summ.round(3).to_string(index=False))

    # ---- SHAP consensus across folds ----------------------------------------------------------------
    nf = len(cfg["data"]["cohorts"])
    top_n = int(cfg["shap"]["top_n"])
    # Late fusion compresses every branch to 32 units, so per-gene SHAP scales inversely with branch size:
    # genes are ranked WITHIN their branch, and each branch's overall share is reported separately.
    sh["within_branch_share"] = sh["mean_abs_shap"] / sh.groupby(["test_cohort", "branch"])["mean_abs_shap"].transform("sum")
    sh["fold_rank"] = sh.groupby("test_cohort")["mean_abs_shap"].rank(ascending=False)
    sh["branch_rank"] = sh.groupby(["test_cohort", "branch"])["mean_abs_shap"].rank(ascending=False)
    cons = sh.groupby("gene").agg(branch=("branch", "first"), consensus_importance=("norm_importance", "sum"),
                                  within_branch_importance=("within_branch_share", "sum"),
                                  folds_selected=("test_cohort", "nunique"),
                                  folds_in_top50=("fold_rank", lambda r: int((r <= top_n).sum())),
                                  folds_in_branch_top20=("branch_rank", lambda r: int((r <= 20).sum())),
                                  direction_corr=("direction_corr", "mean"))
    cons["consensus_importance"] /= nf
    cons["within_branch_importance"] /= nf
    cons["branch_rank"] = cons.groupby("branch")["within_branch_importance"].rank(ascending=False, method="first").astype(int)
    cons["direction"] = np.where(cons["direction_corr"] >= 0, "higher_in_SjS", "lower_in_SjS")
    cons = cons.sort_values("consensus_importance", ascending=False)
    cons["shap_rank"] = np.arange(1, len(cons) + 1)

    # ---- compare with the main pipeline -------------------------------------------------------------
    de = pd.read_csv(ROOT / cfg["inputs"]["stage3"] / "de_all_genes.csv", index_col=0)
    rk = pd.read_csv(ROOT / cfg["inputs"]["stage7"] / "target_ranking_all.csv", index_col=0)
    cand = set(pd.read_csv(ROOT / cfg["inputs"]["stage3"] / "candidate_genes.csv", index_col=0).index)
    cons = cons.join(de[["meta_logFC", "meta_FDR", "adj_meta_FDR"]], how="left")
    cons = cons.join(rk[["rank", "priority_score", "top_compartment", "infiltration_robust", "clinical_sjogren_target"]]
                     .rename(columns={"rank": "pipeline_rank"}), how="left")
    cons["meta_candidate"] = cons.index.isin(cand)
    cons["direction_agrees_with_meta"] = (np.sign(cons["direction_corr"]) == np.sign(cons["meta_logFC"]))
    cons.index.name = "gene"
    cons.to_csv(out / "deepmolft_shap_targets.csv")
    top = cons.head(top_n)
    supported = cons[(cons["branch_rank"] <= 50) & cons["meta_candidate"] & cons["direction_agrees_with_meta"]]
    supported.sort_values(["branch", "branch_rank"]).to_csv(out / "deepmolft_supported_targets.csv")
    for b in cfg["branches"]:
        cons[cons["branch"] == b].sort_values("branch_rank").head(50).to_csv(out / f"deepmolft_top50_{b}_branch.csv")
    pipeline_top = set(rk.head(top_n).index)
    comp = {"top50_shap_in_meta_candidates": int(top["meta_candidate"].sum()),
            "top50_shap_direction_agrees_with_meta": int(top["direction_agrees_with_meta"].sum()),
            "overlap_top50_shap_vs_top50_pipeline": int(len(set(top.index) & pipeline_top)),
            "overlap_genes": sorted(set(top.index) & pipeline_top),
            "supported_targets_(branch_top50_and_meta_candidate_same_direction)": int(len(supported)),
            "supported_by_branch": supported["branch"].value_counts().to_dict(),
            "branch_share_mean": br.groupby("branch")["share_of_total_abs_shap"].mean().round(3).to_dict()}
    json.dump(comp, open(out / "comparison_with_pipeline.json", "w"), indent=2)
    log.info("top-%d SHAP genes: %s", 30, list(cons.index[:30]))
    log.info("comparison with main pipeline: %s", json.dumps(comp))

    make_figures(out / "figures", met, summ, cons, br, dl, pooled_shap, cfg)
    log.info("done -> %s", out)


if __name__ == "__main__":
    main()

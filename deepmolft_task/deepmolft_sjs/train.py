"""Training (DeepMOLFT main.py equivalent): Adam, class-balanced BCE + L1, batch 32, up to 500 epochs, early
stopping on validation loss (patience 50, best weights restored), Optuna over the DeepMOLFT search space.

Difference from the original, on purpose: validation data for early stopping and the Optuna objective is an
inner split of the TRAINING cohorts (stratified, grouped by patient) -- never the held-out test cohort.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Tuple

import numpy as np
import optuna
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from .model import DeepMOLFT

optuna.logging.set_verbosity(optuna.logging.WARNING)


def inner_split(y: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    k = max(2, int(round(1.0 / frac)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(np.zeros(len(y)), y, groups))
    return tr, va


def _tensors(X: Dict[str, np.ndarray], branches: List[str], idx=None) -> List[torch.Tensor]:
    return [torch.from_numpy(X[b] if idx is None else X[b][idx]) for b in branches]


def fit_deepmolft(X: Dict[str, np.ndarray], y: np.ndarray, tr: np.ndarray, va: np.ndarray, branches: List[str],
                  hp: Dict[str, float], mcfg: dict, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = DeepMOLFT([X[b].shape[1] for b in branches], hp, mcfg["branch_sizes"], mcfg["branch_dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    l1 = float(mcfg["l1"])
    ytr = torch.tensor(y[tr], dtype=torch.float32)
    yva = torch.tensor(y[va], dtype=torch.float32)
    if mcfg["class_balanced"]:
        n, pos = len(ytr), float(ytr.sum())
        w_pos, w_neg = n / (2 * max(pos, 1)), n / (2 * max(n - pos, 1))   # sklearn 'balanced'
    else:
        w_pos = w_neg = 1.0
    Xtr, Xva = _tensors(X, branches, tr), _tensors(X, branches, va)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    best, best_state, wait, best_epoch = np.inf, None, 0, 0
    rng = np.random.default_rng(seed)
    for epoch in range(int(mcfg["epochs"])):
        model.train()
        order = rng.permutation(len(tr))
        for s in range(0, len(order), int(mcfg["batch_size"])):
            b = order[s:s + int(mcfg["batch_size"])]
            logit = model(*[x[b] for x in Xtr]).squeeze(1)
            w = torch.where(ytr[b] > 0.5, torch.tensor(w_pos), torch.tensor(w_neg))
            loss = (bce(logit, ytr[b]) * w).mean() + l1 * model.l1_penalty()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(bce(model(*Xva).squeeze(1), yva).mean() + l1 * model.l1_penalty())
        if vl < best - 1e-6:
            best, best_state, wait, best_epoch = vl, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
            if wait >= int(mcfg["patience"]):
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def predict(model, X: Dict[str, np.ndarray], branches: List[str]) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(model(*_tensors(X, branches)).squeeze(1)).numpy()


def tune(X: Dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, branches: List[str], cfg: dict, seed: int):
    ocfg, mcfg = cfg["optuna"], cfg["model"]
    tr, va = inner_split(y, groups, cfg["validation"]["inner_val_fraction"], seed)
    Xva = {b: X[b][va] for b in branches}

    def objective(trial):
        hp = {"learning_rate": trial.suggest_float("learning_rate", *ocfg["learning_rate"], log=True),
              "merged_1_dropout": trial.suggest_float("merged_1_dropout", *ocfg["dropout"]),
              "merged_1_size": trial.suggest_int("merged_1_size", *ocfg["merged_1_size"]),
              "merged_2_dropout": trial.suggest_float("merged_2_dropout", *ocfg["dropout"]),
              "merged_2_size": trial.suggest_int("merged_2_size", *ocfg["merged_2_size"]),
              "merged_3_dropout": trial.suggest_float("merged_3_dropout", *ocfg["dropout"]),
              "merged_3_size": trial.suggest_int("merged_3_size", *ocfg["merged_3_size"])}
        model, ep = fit_deepmolft(X, y, tr, va, branches, hp, mcfg, seed)
        trial.set_user_attr("best_epoch", ep)
        return roc_auc_score(y[va], predict(model, Xva, branches))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=int(ocfg["n_trials"]))
    return study


def fit_baseline(name: str, Xtr: np.ndarray, ytr: np.ndarray, groups: np.ndarray, seed: int):
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    if name == "elasticnet":
        est = LogisticRegression(solver="saga", l1_ratio=0.5, max_iter=5000, class_weight="balanced", random_state=seed)
        gs = GridSearchCV(est, {"C": [0.01, 0.05, 0.1, 0.5, 1.0]}, cv=cv, scoring="roc_auc")
    else:
        est = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=seed, n_jobs=4)
        gs = GridSearchCV(est, {"max_features": ["sqrt", 0.1], "min_samples_leaf": [1, 3]}, cv=cv, scoring="roc_auc")
    gs.fit(Xtr, ytr, groups=groups)
    return gs.best_estimator_

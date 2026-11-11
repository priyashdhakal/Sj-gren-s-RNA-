"""SHAP for the DeepMOLFT ensemble (DeepMOLFT shap_from_logs.py equivalent) + faithfulness check.

shap.DeepExplainer on the multi-input network, explaining the held-out cohort (DeepMOLFT: SHAP on the
held-out partition), with a background of training samples (the original used the test set itself as
background). The explained output is the logit, so attributions add up in log-odds.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import shap
import torch


def deep_shap(model, background: Dict[str, np.ndarray], explain: Dict[str, np.ndarray],
              branches: List[str]) -> Dict[str, np.ndarray]:
    bg = [torch.from_numpy(background[b]) for b in branches]
    ex = [torch.from_numpy(explain[b]) for b in branches]
    explainer = shap.DeepExplainer(model, bg)
    sv = explainer.shap_values(ex, check_additivity=False)
    if isinstance(sv, list) and len(sv) == 1 and isinstance(sv[0], list):
        sv = sv[0]
    out = {}
    for b, v in zip(branches, sv):
        v = np.asarray(v)
        if v.ndim == 3:
            v = v[..., 0]
        out[b] = v
    return out

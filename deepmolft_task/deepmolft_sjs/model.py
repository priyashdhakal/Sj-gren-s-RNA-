"""DeepMOLFT late-fusion network (PyTorch port of DeepMOLFT src/model.py).

Each input branch: Dense 128 -> 64 -> 32, ReLU, L1(0.001) on the kernel, dropout 0.5 / 0.3 / 0.2.
Branch outputs are concatenated and passed through three tuned fused Dense layers (ReLU, L1, dropout) and a
single-unit output. The forward pass returns the logit; sigmoid is applied for probabilities (equivalent to
Keras' sigmoid output + binary cross-entropy). Weights use Glorot-uniform / zero bias, as in Keras.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn


def _dense(i: int, o: int) -> nn.Linear:
    layer = nn.Linear(i, o)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class DeepMOLFT(nn.Module):
    def __init__(self, input_dims: List[int], hp: Dict[str, float], branch_sizes=(128, 64, 32),
                 branch_dropout=(0.5, 0.3, 0.2)):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in input_dims:
            layers, prev = [], d
            for size, drop in zip(branch_sizes, branch_dropout):
                layers += [_dense(prev, size), nn.ReLU(), nn.Dropout(drop)]
                prev = size
            self.branches.append(nn.Sequential(*layers))
        fused_in = branch_sizes[-1] * len(input_dims)
        sizes = [int(hp["merged_1_size"]), int(hp["merged_2_size"]), int(hp["merged_3_size"])]
        drops = [hp["merged_1_dropout"], hp["merged_2_dropout"], hp["merged_3_dropout"]]
        layers, prev = [], fused_in
        for size, drop in zip(sizes, drops):
            layers += [_dense(prev, size), nn.ReLU(), nn.Dropout(drop)]
            prev = size
        self.fused = nn.Sequential(*layers)
        self.out = _dense(prev, 1)

    def forward(self, *xs: torch.Tensor) -> torch.Tensor:
        h = torch.cat([b(x) for b, x in zip(self.branches, xs)], dim=1)
        return self.out(self.fused(h))

    def l1_penalty(self) -> torch.Tensor:
        """Keras kernel_regularizer=l1(0.001) on every branch and fused Dense kernel (not the output)."""
        kernels = [m.weight for m in list(self.branches.modules()) + list(self.fused.modules())
                   if isinstance(m, nn.Linear)]
        return sum(w.abs().sum() for w in kernels)

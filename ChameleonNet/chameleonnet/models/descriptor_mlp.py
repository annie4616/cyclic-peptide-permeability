"""Tiny MLP for the curated 4D descriptor branch.

We standardize inputs at the dataset level (or via a wrapped scaler in the
trainer); this module just projects them into the shared hidden dim.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DescriptorMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # NaN guard: descriptor cells can be missing for some peptides.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return self.net(x)

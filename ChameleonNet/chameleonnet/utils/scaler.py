"""Standardize the descriptor branch using train-set statistics.

Why bake this into a class: descriptors include radically different scales
(SASA in tens, NPSA in single digits, Desolv in kJ/mol). The model can in
principle learn the rescaling but it's a free win to give it pre-normalized
inputs, especially with only 5160 peptides.
"""

from __future__ import annotations

import numpy as np
import torch


class DescriptorScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "DescriptorScaler":
        X = np.asarray(X, dtype=np.float64)
        # Treat NON-FINITE cells (NaN *and* ±inf — the full RDKit descriptor set
        # has overflowing columns like BCUT2D/Ipc) as missing so they don't
        # corrupt the per-column mean/std.
        valid = np.isfinite(X).astype(np.float64)
        masked = np.where(valid > 0, X, 0.0)
        n = valid.sum(axis=0).clip(min=1.0)
        self.mean = (masked * valid).sum(axis=0) / n
        var = (((masked - self.mean) ** 2) * valid).sum(axis=0) / n
        self.std = np.sqrt(var).clip(min=1e-6)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            return x
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        # Zero non-finite inputs, standardize, then clip to a sane range so a
        # single wild descriptor can't blow up the downstream MLP.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        z = (x - mean) / std
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)

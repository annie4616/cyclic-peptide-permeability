"""Composite loss = main MSE + chameleonic auxiliary + Tanimoto triplet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chameleonic import chameleonic_magnitude_loss
from .triplet import tanimoto_triplet_loss


@dataclass
class CompositeLoss:
    """Configuration container; call `composite_loss` for the actual computation."""
    lambda_chameleonic: float = 0.1
    lambda_triplet: float = 0.1
    chameleonic_norm_weight: float = 1.0
    chameleonic_head_weight: float = 1.0
    pampa_baseline: float = -8.0
    triplet_margin: float = 0.5
    triplet_sim_high: float = 0.7
    triplet_sim_low: float = 0.4
    triplet_pampa_gap: float = 1.0
    triplet_max: int = 64


def composite_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    smiles: Optional[List[str]] = None,
    fused_embedding: Optional[torch.Tensor] = None,
    cfg: CompositeLoss = CompositeLoss(),
) -> Dict[str, torch.Tensor]:
    """Returns a dict with 'total', 'mse', 'chameleonic', 'triplet'.

    The trainer logs the breakdown; only `total` is used for the backward
    pass. Auxiliary losses contribute zero (no-op) when their preconditions
    aren't met (e.g. RDKit missing → triplet returns 0).
    """
    pampa_pred = outputs["pampa_pred"]
    mse = F.mse_loss(pampa_pred, targets)

    cham = chameleonic_magnitude_loss(
        h_diff=outputs["h_diff"],
        chameleonic_pred=outputs["chameleonic_pred"],
        pampa=targets,
        pampa_baseline=cfg.pampa_baseline,
        norm_weight=cfg.chameleonic_norm_weight,
        head_weight=cfg.chameleonic_head_weight,
    )

    if fused_embedding is not None and smiles is not None and cfg.lambda_triplet > 0:
        trip = tanimoto_triplet_loss(
            embeddings=fused_embedding,
            smiles=smiles,
            pampa=targets,
            margin=cfg.triplet_margin,
            sim_high=cfg.triplet_sim_high,
            sim_low=cfg.triplet_sim_low,
            pampa_gap=cfg.triplet_pampa_gap,
            max_triplets=cfg.triplet_max,
        )
    else:
        trip = pampa_pred.new_zeros(())

    total = mse + cfg.lambda_chameleonic * cham + cfg.lambda_triplet * trip
    return {
        "total": total,
        "mse": mse.detach(),
        "chameleonic": cham.detach(),
        "triplet": trip.detach(),
    }

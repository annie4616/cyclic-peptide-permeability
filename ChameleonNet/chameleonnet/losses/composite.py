"""Composite loss = main MSE + chameleonic auxiliary + Tanimoto triplet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .chameleonic import chameleonic_magnitude_loss
from .residue_chameleon import residue_chameleon_psa_loss
from .triplet import tanimoto_triplet_loss


@dataclass
class CompositeLoss:
    """Configuration container; call `composite_loss` for the actual computation."""
    lambda_chameleonic: float = 0.1
    lambda_triplet: float = 0.1
    lambda_residue_psa: float = 0.0  # V2-only; >0 enables residue-Δ ↔ |ΔPSA| aux
    lambda_adv: float = 0.0  # weight on the scaffold-adversary CE in the total
    lambda_resid_l2: float = 0.0  # L2 on the physics-residual head output
    lambda_ib: float = 0.0  # KL weight for the information bottleneck
    lambda_distill: float = 0.0  # pull learned-only head toward gbr_pred
    chameleonic_norm_weight: float = 1.0
    chameleonic_head_weight: float = 1.0
    pampa_baseline: float = -8.0
    triplet_margin: float = 0.5
    triplet_sim_high: float = 0.7
    triplet_sim_low: float = 0.4
    triplet_pampa_gap: float = 1.0
    triplet_max: int = 64
    triplet_morgan_radius: int = 2
    triplet_morgan_nbits: int = 2048


def composite_loss(
    outputs: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    smiles: Optional[List[str]] = None,
    fused_embedding: Optional[torch.Tensor] = None,
    delta_psa_target: Optional[torch.Tensor] = None,
    scaffold_groups: Optional[torch.Tensor] = None,
    gbr_pred: Optional[torch.Tensor] = None,
    cfg: CompositeLoss = CompositeLoss(),
) -> Dict[str, torch.Tensor]:
    """Returns a dict with 'total', 'mse', 'chameleonic', 'triplet', 'residue_psa', 'adv'.

    The trainer logs the breakdown; only `total` is used for the backward
    pass. Auxiliary losses contribute zero (no-op) when their preconditions
    aren't met (e.g. RDKit missing → triplet returns 0; V1 model has no
    `delta_residue` → residue_psa returns 0).
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
            morgan_radius=cfg.triplet_morgan_radius,
            morgan_nbits=cfg.triplet_morgan_nbits,
        )
    else:
        trip = pampa_pred.new_zeros(())

    if (
        cfg.lambda_residue_psa > 0
        and "delta_residue" in outputs
        and "res_mask" in outputs
        and delta_psa_target is not None
    ):
        res_psa = residue_chameleon_psa_loss(
            delta_residue=outputs["delta_residue"],
            res_mask=outputs["res_mask"],
            delta_psa_target=delta_psa_target,
        )
    else:
        res_psa = pampa_pred.new_zeros(())

    # Scaffold adversary (gradient-reversal already applied inside the model).
    # ignore_index=-1 skips samples whose scaffold group is unknown (e.g. val
    # ids that weren't part of the train-only clustering). The encoder feels
    # this term only through the reversed gradient, scaled by the GRL lambda
    # the trainer schedules — so `lambda_adv` here just weights the CE.
    if (
        cfg.lambda_adv > 0
        and "scaffold_logits" in outputs
        and scaffold_groups is not None
    ):
        adv = F.cross_entropy(
            outputs["scaffold_logits"], scaffold_groups, ignore_index=-1
        )
        if not torch.isfinite(adv):  # whole batch masked → no signal
            adv = pampa_pred.new_zeros(())
    else:
        adv = pampa_pred.new_zeros(())

    resid_l2 = (
        outputs["resid"].pow(2).mean()
        if (cfg.lambda_resid_l2 > 0 and "resid" in outputs)
        else pampa_pred.new_zeros(())
    )
    ib = (
        outputs["ib_kl"]
        if (cfg.lambda_ib > 0 and "ib_kl" in outputs)
        else pampa_pred.new_zeros(())
    )
    distill = (
        F.mse_loss(outputs["distill_pred"], gbr_pred.detach())
        if (cfg.lambda_distill > 0 and "distill_pred" in outputs and gbr_pred is not None)
        else pampa_pred.new_zeros(())
    )

    total = (
        mse
        + cfg.lambda_chameleonic * cham
        + cfg.lambda_triplet * trip
        + cfg.lambda_residue_psa * res_psa
        + cfg.lambda_adv * adv
        + cfg.lambda_resid_l2 * resid_l2
        + cfg.lambda_ib * ib
        + cfg.lambda_distill * distill
    )
    return {
        "total": total,
        "mse": mse.detach(),
        "chameleonic": cham.detach(),
        "triplet": trip.detach(),
        "residue_psa": res_psa.detach(),
        "adv": adv.detach(),
        "resid_l2": resid_l2.detach(),
        "ib": ib.detach(),
        "distill": distill.detach(),
    }

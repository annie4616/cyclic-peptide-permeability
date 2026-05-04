"""Chameleonic-magnitude auxiliary loss.

Hypothesis: more permeable peptides exhibit larger conformational change
between water and hexane (the "chameleonic" effect). We push the model to
make ||h_water - h_hexane|| correlate with PAMPA.

Implementation choices:
  - Use L1 between the L2 norm of h_diff and a target scalar derived from
    PAMPA. We rescale PAMPA to a non-negative target via (PAMPA - PAMPA_min)
    so higher permeability → larger expected magnitude.
  - Optionally also regress an explicit auxiliary head's prediction against
    PAMPA — that's `chameleonic_pred` from the model. This adds capacity for
    the model to learn a non-linear mapping from h_diff to PAMPA-magnitude.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def chameleonic_magnitude_loss(
    h_diff: torch.Tensor,
    chameleonic_pred: torch.Tensor,
    pampa: torch.Tensor,
    pampa_baseline: float = -8.0,
    norm_weight: float = 1.0,
    head_weight: float = 1.0,
) -> torch.Tensor:
    """Two-part auxiliary loss.

    1. norm term: encourage ||h_diff||_2 to track (PAMPA - pampa_baseline).
       This is the geometric "chameleonic strength is proportional to
       permeability" signal.
    2. head term: a learned head predicts PAMPA from h_diff alone. MSE
       against the true PAMPA. This forces h_diff to be informative about
       permeability on its own, not just useful as an extra feature.

    pampa_baseline shifts the regression target so the magnitude is non-
    negative. -8 is roughly the lower clip in CycPeptMPDB PAMPA values.
    """
    target = (pampa - pampa_baseline).clamp_min(0.0)
    norm = h_diff.norm(dim=-1)
    loss_norm = F.l1_loss(norm, target)
    loss_head = F.mse_loss(chameleonic_pred, pampa)
    return norm_weight * loss_norm + head_weight * loss_head

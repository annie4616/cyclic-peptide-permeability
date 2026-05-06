"""V2 residue-resolved chameleonic auxiliary losses.

We don't yet have per-residue ΔPSA in the dataset (would require running
freeSASA per frame per residue — a separate preprocessing job). What we do
have is the *global* ΔPSA scalar in the descriptor block. We can already use
it to constrain the residue-level head:

    ||delta_residue||_2 (summed over residues) should track |global ΔPSA|

This is a weak but well-defined signal: when water-vs-hexane PSA hardly
changes, the residue-level Δ vectors should also be small in aggregate. When
PSA changes a lot, at least one residue should carry a large Δ.

When per-residue ΔPSA becomes available, swap the target on the right-hand
side for the per-residue ground truth and remove the aggregation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def residue_chameleon_psa_loss(
    delta_residue: torch.Tensor,   # (B, R, F)
    res_mask: torch.Tensor,        # (B, R) True = pad
    delta_psa_target: torch.Tensor,  # (B,) scalar |Water_PSA - Hexane_PSA|
) -> torch.Tensor:
    """Encourage the aggregated residue Δ magnitude to track |global ΔPSA|.

    Returns a scalar loss. Robust to NaN targets (peptides with missing PSA
    columns contribute zero).
    """
    valid = (~res_mask).float().unsqueeze(-1)  # (B, R, 1)
    norm_per_res = (delta_residue * valid).norm(dim=-1)  # (B, R)
    aggregated = norm_per_res.sum(dim=-1)  # (B,)

    target = delta_psa_target.abs()
    finite = torch.isfinite(target)
    if not finite.any():
        return delta_residue.new_zeros(())
    return F.l1_loss(aggregated[finite], target[finite])

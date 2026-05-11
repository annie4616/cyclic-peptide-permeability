"""Atom -> residue pooling.

V2 introduces an explicit residue-level intermediate. Atoms within the same
residue are pooled (mean + attention) into a single token, then aligned across
the water and hexane environments to support residue-resolved chameleonic
contrast.

Inputs use the same flat-conformer layout as the existing encoder
(`(M, Nmax, F)` per-atom features with a `(M, Nmax)` pad mask). We additionally
need a per-atom residue index — the dataset already carries `res` (residue
*name id*); for pooling we need the *positional* residue index (1..R within
the peptide), which the dataset also exposes.

Why this is in its own module: the current `ConformerEncoder` is fine; we just
wrap it with a pooler so V1 stays untouched.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtomToResiduePool(nn.Module):
    """Pool per-atom features to per-residue tokens.

    For each (conformer, residue) bucket we compute:
      pooled = mean_over_atoms + attn_over_atoms

    The attention path lets the model up-weight functionally important atoms
    (e.g. amide N for H-bonding) within a residue.
    """

    def __init__(self, hidden_dim: int, attn_hidden: int = 64):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(
        self,
        h_atom: torch.Tensor,        # (M, Nmax, F)
        res_pos: torch.Tensor,       # (M, Nmax) int — residue position id (1..R), 0 = pad
        pad_mask: torch.Tensor,      # (M, Nmax) bool — True = pad atom
        max_residues: int,           # R_max in the batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (h_res, res_mask).

        h_res:    (M, R_max, F) residue tokens (zero where padded)
        res_mask: (M, R_max) bool — True = pad residue (no atoms in this slot)
        """
        M, Nmax, F_dim = h_atom.shape
        device = h_atom.device

        valid = (~pad_mask).float().unsqueeze(-1)  # (M, Nmax, 1)
        # Treat pad atoms as residue-0; we'll mask their contribution out below.
        rp = res_pos.clamp_min(0)

        # ---- mean pool ----
        sum_buf = h_atom.new_zeros(M, max_residues + 1, F_dim)
        cnt_buf = h_atom.new_zeros(M, max_residues + 1, 1)
        idx = rp.unsqueeze(-1).expand(-1, -1, F_dim)
        sum_buf.scatter_add_(1, idx, h_atom * valid)
        cnt_buf.scatter_add_(1, rp.unsqueeze(-1), valid)
        # zero out residue-0 (pad slot) so it doesn't contaminate downstream
        sum_buf[:, 0] = 0
        cnt_buf[:, 0] = 0
        mean_pool = sum_buf / cnt_buf.clamp_min(1.0)
        # drop the pad slot (index 0) → (M, R_max, F)
        mean_pool = mean_pool[:, 1:]
        cnt_pool = cnt_buf[:, 1:].squeeze(-1)  # (M, R_max)

        # ---- attention pool (per-residue softmax over atoms) ----
        scores = self.attn(h_atom).squeeze(-1)  # (M, Nmax)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        # group-wise softmax: for each residue id, softmax over atoms in that group.
        # Done by computing per-(sample, residue) max for stability.
        max_per_res = h_atom.new_full((M, max_residues + 1), float("-inf"))
        max_per_res.scatter_reduce_(1, rp, scores, reduce="amax", include_self=True)
        # If a residue slot has no atoms, max stays -inf → softmax = 0.
        gather_max = max_per_res.gather(1, rp)
        weights = torch.exp(scores - gather_max)
        # weights at pad atoms become exp(-inf) ≈ 0
        weights = weights.masked_fill(pad_mask, 0.0)
        denom = h_atom.new_zeros(M, max_residues + 1)
        denom.scatter_add_(1, rp, weights)
        gather_denom = denom.gather(1, rp).clamp_min(1e-9)
        weights = weights / gather_denom  # (M, Nmax)

        attn_buf = h_atom.new_zeros(M, max_residues + 1, F_dim)
        attn_buf.scatter_add_(1, idx, h_atom * weights.unsqueeze(-1))
        attn_buf[:, 0] = 0
        attn_pool = attn_buf[:, 1:]

        # ---- combine ----
        h_res = self.proj(torch.cat([mean_pool, attn_pool], dim=-1))  # (M, R_max, F)

        # residue mask: True where the residue has zero atoms in this conformer
        res_mask = cnt_pool == 0  # (M, R_max)
        # zero out tokens for padded residues so downstream layers don't leak
        h_res = h_res * (~res_mask).unsqueeze(-1).float()
        return h_res, res_mask


def pool_residues_over_conformers(
    h_res: torch.Tensor,            # (M, R_max, F)
    res_mask: torch.Tensor,         # (M, R_max) True = pad
    batch_index: torch.Tensor,      # (M,) sample id per conformer
    batch_size: int,
    conformer_weights: torch.Tensor | None = None,  # (M,) — optional population weight per conformer
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-conformer residue tokens into per-peptide residue tokens.

    For each (sample, residue) we average across the conformers that *have*
    that residue (i.e. residue not padded). When `conformer_weights` is given
    we do a population-weighted mean instead (cluster centroids: weight ∝
    cluster size). Conformer-level attention pooling could replace the mean
    here, but the existing `ConformerAttentionPool` is global (one weight per
    conformer). We keep this lightweight.
    """
    M, R, F_dim = h_res.shape
    device = h_res.device

    valid = (~res_mask).float().unsqueeze(-1)  # (M, R, 1)
    if conformer_weights is not None:
        w = conformer_weights.view(M, 1, 1)  # (M, 1, 1)
        valid = valid * w
    sum_buf = h_res.new_zeros(batch_size, R, F_dim)
    cnt_buf = h_res.new_zeros(batch_size, R, 1)

    bi = batch_index.view(-1, 1, 1).expand(-1, R, F_dim)
    sum_buf.scatter_add_(0, bi, h_res * valid)
    bi_cnt = batch_index.view(-1, 1, 1).expand(-1, R, 1)
    cnt_buf.scatter_add_(0, bi_cnt, valid)

    pooled = sum_buf / cnt_buf.clamp_min(1e-9)
    pep_mask = cnt_buf.squeeze(-1) <= 0  # (B, R) — residue absent from all conformers
    pooled = pooled * (~pep_mask).unsqueeze(-1).float()
    return pooled, pep_mask

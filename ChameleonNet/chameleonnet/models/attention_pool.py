"""Attention pooling over a per-sample bag of conformer embeddings.

The conformer encoder returns one vector per conformer (M total across the
batch). We aggregate those down to one vector per peptide (B total) using
learned attention weights, which is a softer Boltzmann surrogate: the model
learns which conformers matter most for permeability.

Why attention and not Boltzmann from raw energies: the assay CSV has
Desolvation_Free_Energy as a single scalar per peptide, not a per-conformer
energy. We therefore need a learned weighting. Interpretability-wise, the
attention weights can still be inspected post-hoc as "which conformer the
model trusted".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformerAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, attn_hidden: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(
        self,
        h: torch.Tensor,  # (M, hidden_dim)  — flat per-conformer embeddings
        batch_index: torch.Tensor,  # (M,) — which sample each conformer belongs to
        batch_size: int,
        log_prior: torch.Tensor | None = None,  # (M,) — optional log-prior bias (e.g. log cluster size)
    ) -> torch.Tensor:
        scores = self.score(h).squeeze(-1)  # (M,)
        # Adding log_prior turns the post-softmax weights into a posterior
        # proportional to prior * exp(score). For cluster centroids the prior
        # is the cluster size, so larger basins get more weight by default
        # while the model can still up/down-weight via the learned score.
        if log_prior is not None:
            scores = scores + log_prior

        # Per-sample softmax: subtract per-batch max for stability, then
        # exponentiate and normalize against the per-sample sum.
        out = torch.zeros(batch_size, h.shape[-1], device=h.device, dtype=h.dtype)

        # Stable softmax via per-batch max.
        max_per_batch = torch.full((batch_size,), float("-inf"), device=h.device)
        max_per_batch = max_per_batch.scatter_reduce(
            0, batch_index, scores, reduce="amax", include_self=True
        )
        # Edge case: a batch slot with no conformers (shouldn't happen) keeps -inf.
        max_per_batch = torch.where(
            torch.isfinite(max_per_batch),
            max_per_batch,
            torch.zeros_like(max_per_batch),
        )

        weights = torch.exp(scores - max_per_batch[batch_index])
        denom = torch.zeros(batch_size, device=h.device).scatter_add_(
            0, batch_index, weights
        ).clamp_min(1e-9)
        norm_weights = weights / denom[batch_index]

        # Weighted sum into out[batch_index].
        out = out.index_add(0, batch_index, h * norm_weights.unsqueeze(-1))
        return out  # (B, hidden_dim)

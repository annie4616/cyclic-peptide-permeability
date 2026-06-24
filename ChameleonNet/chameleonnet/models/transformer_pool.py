"""Transformer pooling over a per-sample *sequence* of conformer embeddings.

Unlike `ConformerAttentionPool` (a permutation-invariant softmax bag), this
treats each peptide's conformers as an ordered sequence in trajectory time:

    [CLS] h_0 h_1 ... h_{K-1}

A learnable positional embedding encodes the trajectory order, a learnable
[CLS] token is prepended, and a small TransformerEncoder mixes them. The CLS
output is the pooled per-peptide environment vector — analogous to BERT-style
sequence pooling, but the "tokens" are EGNN-encoded conformers.

The conformer encoder hands us a *flat* (M, hidden) tensor with a `batch_index`
mapping each conformer to its sample (sample sizes K vary). We scatter that flat
layout back into a padded (B, Lmax, hidden) batch — preserving the per-sample
trajectory order, which `_collate_env` already lays out as frame 0..K-1 — then
run the transformer with a key-padding mask so pad slots are ignored.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConformerTransformerPool(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        nhead: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.1,
        max_conformers: int = 256,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.normal_(self.cls, std=0.02)
        # +1 position slot reserved for CLS at index 0; conformers occupy 1..K.
        self.pos_embed = nn.Parameter(torch.zeros(1, max_conformers + 1, hidden_dim))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.max_conformers = max_conformers
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def _to_padded(
        self,
        h: torch.Tensor,          # (M, hidden) flat, conformers in trajectory order per sample
        batch_index: torch.Tensor,  # (M,)
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Scatter flat conformers into (B, Lmax, hidden) + a (B, Lmax) pad mask.

        Relies on batch_index being grouped/sorted per sample (it is — the
        collate appends sample 0's K conformers, then sample 1's, ...), so the
        within-sample arrival order is the trajectory order.
        """
        device = h.device
        M = h.shape[0]
        # K per sample.
        K = torch.zeros(batch_size, dtype=torch.long, device=device).scatter_add_(
            0, batch_index, torch.ones(M, dtype=torch.long, device=device)
        )
        Lmax = int(K.max().item()) if K.numel() > 0 else 0
        # Within-sample position (0..K-1) of each conformer, in arrival order.
        # batch_index is contiguous per sample (collate appends sample 0's K
        # conformers, then sample 1's, ...), and arrival order == trajectory
        # order, so the running per-sample count is the trajectory position.
        # offset[b] = number of conformers of samples before b (exclusive prefix
        # sum of K); within = global arrival index - offset[sample].
        offset = torch.zeros(batch_size, dtype=torch.long, device=device)
        if batch_size > 0:
            offset[1:] = torch.cumsum(K, dim=0)[:-1]
        order = torch.arange(M, device=device)
        within = order - offset[batch_index]  # (M,) position 0..K-1

        padded = torch.zeros(batch_size, Lmax, self.hidden_dim, device=device, dtype=h.dtype)
        pad_mask = torch.ones(batch_size, Lmax, dtype=torch.bool, device=device)  # True = pad
        padded[batch_index, within] = h
        pad_mask[batch_index, within] = False
        return padded, pad_mask, Lmax

    def forward(
        self,
        h: torch.Tensor,            # (M, hidden)
        batch_index: torch.Tensor,  # (M,)
        batch_size: int,
        log_prior: torch.Tensor | None = None,  # accepted for API parity; unused (order matters here)
    ) -> torch.Tensor:
        if h.shape[0] == 0:
            return torch.zeros(batch_size, self.hidden_dim, device=h.device, dtype=h.dtype)
        padded, pad_mask, Lmax = self._to_padded(h, batch_index, batch_size)

        cls = self.cls.expand(batch_size, -1, -1)  # (B, 1, hidden)
        seq = torch.cat([cls, padded], dim=1)       # (B, 1+Lmax, hidden)
        # Positional embeddings: CLS at slot 0, conformers at 1..Lmax. Clip the
        # table to the needed length (Lmax may exceed max_conformers only if a
        # trajectory is longer than configured — then we tile the last slot).
        L = 1 + Lmax
        if L <= self.pos_embed.shape[1]:
            pos = self.pos_embed[:, :L]
        else:
            extra = L - self.pos_embed.shape[1]
            tail = self.pos_embed[:, -1:].expand(-1, extra, -1)
            pos = torch.cat([self.pos_embed, tail], dim=1)
        seq = seq + pos

        # CLS is never masked; conformer pad slots are.
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=h.device)
        key_padding_mask = torch.cat([cls_mask, pad_mask], dim=1)  # (B, 1+Lmax) True = ignore

        out = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        return self.norm(out[:, 0])  # (B, hidden) — CLS

"""3D conformer encoder.

A small EGNN-style network: each layer updates per-atom features using
relative-distance messages, which is invariant to rotation and translation.

Why EGNN and not SchNet: SchNet only uses pairwise distances; EGNN additionally
updates positions equivariantly, which we don't strictly need for the final
scalar embedding, but the distance-based message aggregation matches what we
want (chameleonic shape contrast comes from geometry, not chirality).

We keep this from-scratch instead of pulling torch_geometric to avoid a heavy
dependency for what is a 3-4 layer GNN. If torch_geometric is later required
for other parts of the pipeline, this can be swapped out for `egnn_pytorch`.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class _EGNNLayer(nn.Module):
    """One EGNN layer with scalar feature updates.

    Per-pair message m_ij = phi_e([h_i, h_j, ||x_i - x_j||^2])
    Per-node update    h_i' = phi_h([h_i, sum_j m_ij])

    We skip the position update because we only consume h at the end. This
    makes the layer essentially "SchNet with concatenated endpoint features".
    """

    def __init__(self, hidden_dim: int, edge_hidden: int = 64):
        # 노드 수에 따라 agg의 크기가 엄청 커지거나 부호가 상쇄되어 학습이 불안정해질 수 있어서 마지막 SiLU를 넣음
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, edge_hidden), # h_i, h_j, d2
            nn.SiLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.SiLU(),
        )
        # 출력의 활성화를 두지 않는게 관행이라고 함.
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_hidden, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        h: torch.Tensor,  # (B, N, F)
        x: torch.Tensor,  # (B, N, 3)
        pad_mask: torch.Tensor,  # (B, N) True = pad
    ) -> torch.Tensor:
        B, N, F = h.shape
        # Pairwise squared distances. (B, N, N, 1)
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        d2 = (diff * diff).sum(-1, keepdim=True)

        h_i = h.unsqueeze(2).expand(B, N, N, F)
        h_j = h.unsqueeze(1).expand(B, N, N, F)
        e_in = torch.cat([h_i, h_j, d2], dim=-1)
        m = self.edge_mlp(e_in)  # (B, N, N, edge_hidden)

        # Mask out edges where j is a pad atom.
        edge_mask = (~pad_mask).float().unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        # Also kill self-edges to avoid trivial reinforcement.
        eye = torch.eye(N, device=h.device, dtype=torch.bool)
        self_mask = (~eye).float().unsqueeze(0).unsqueeze(-1)  # (1, N, N, 1)
        m = m * edge_mask * self_mask

        agg = m.sum(dim=2)  # (B, N, edge_hidden)
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        # Zero out pad nodes so they don't leak through residual paths.
        h_new = h_new * (~pad_mask).float().unsqueeze(-1)
        return h_new


class ConformerEncoder(nn.Module):
    """Encodes a flat batch of conformers (sum_K, Nmax, 3) into per-conformer vectors.

    Inputs (from `_pad_atom_dim`):
      coords:   (M, Nmax, 3)    M개 conformer, 최대 원자 수, 3D 좌표
      z:        (M, Nmax)        atomic numbers (0 = pad)
      res:      (M, Nmax)        residue ids (0 = pad)
      pad_mask: (M, Nmax) bool   True = padding atom

    Output: (M, hidden_dim) per-conformer pooled embedding.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        max_z: int = 36,
        residue_vocab_size: int = 64,
        edge_hidden: int = 64,
    ):
        super().__init__()
        self.atom_embed = nn.Embedding(max_z, hidden_dim, padding_idx=0)
        self.res_embed = nn.Embedding(residue_vocab_size, hidden_dim, padding_idx=0)
        self.input_mix = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [_EGNNLayer(hidden_dim, edge_hidden) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        coords: torch.Tensor,
        z: torch.Tensor,
        res: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Initial node features = atomic-number embedding + residue embedding.
        h = self.atom_embed(z) + self.res_embed(res)
        h = self.input_mix(h)
        h = h * (~pad_mask).float().unsqueeze(-1)
        for layer in self.layers:
            h = layer(h, coords, pad_mask)
        h = self.out_norm(h)
        # Mean-pool over real atoms per conformer.
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        return pooled  # (M, hidden_dim)

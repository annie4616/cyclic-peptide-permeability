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

from pathlib import Path
from typing import List, Optional, Union

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


def _load_pretrained_residue_embed(
    path: Union[str, Path],
    expected_vocab_size: int,
    expected_tokens: Optional[List[str]] = None,
) -> tuple[nn.Embedding, int]:
    """Load a (V, lm_hidden) tensor from `build_residue_lm_embeddings.py` and
    wrap it as a frozen nn.Embedding with padding_idx=0.

    Raises if the loaded vocab size or token order doesn't match the current
    ResidueVocab — vocab drift here would silently swap residue priors and
    poison training, so we'd rather fail loudly at init.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    weights = payload["embeddings"] if isinstance(payload, dict) else payload
    if not isinstance(weights, torch.Tensor):
        raise TypeError(
            f"Expected a torch.Tensor of embeddings in {path}, got {type(weights)}"
        )
    if weights.dim() != 2:
        raise ValueError(
            f"Pretrained residue embeddings must be 2D (V, F); got shape {tuple(weights.shape)}"
        )
    V, lm_hidden = weights.shape
    if V != expected_vocab_size:
        raise ValueError(
            f"Pretrained residue vocab size ({V}) != current vocab size "
            f"({expected_vocab_size}). Re-run build_residue_lm_embeddings.py "
            "against the current ResidueVocab."
        )
    if expected_tokens is not None and isinstance(payload, dict):
        stored = payload.get("tokens")
        if stored is not None and list(stored) != list(expected_tokens):
            raise ValueError(
                "Pretrained residue embedding token order does not match the "
                "current ResidueVocab. Re-run build_residue_lm_embeddings.py."
            )
    emb = nn.Embedding.from_pretrained(
        weights.float(), freeze=True, padding_idx=0
    )
    return emb, lm_hidden


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
        max_z: int = 36, # H ~ Br 커버. 원자번호
        residue_vocab_size: int = 64,
        edge_hidden: int = 64,
        residue_emb_path: Optional[Union[str, Path]] = None,
        residue_vocab_tokens: Optional[List[str]] = None,
    ):
        super().__init__()
        self.atom_embed = nn.Embedding(max_z, hidden_dim, padding_idx=0)
        if residue_emb_path is not None:
            # PeptideCLM-2 사전추출 임베딩을 frozen lookup으로 사용하고, projection만 학습.
            # residue_vocab_tokens가 주어지면 저장된 임베딩의 토큰 순서가 현재 vocab과
            # 동일한지 확인해 잘못된 정렬로 인한 silent corruption을 막는다.
            self.res_embed, lm_hidden = _load_pretrained_residue_embed(
                residue_emb_path,
                expected_vocab_size=residue_vocab_size,
                expected_tokens=residue_vocab_tokens,
            )
            self.res_proj: nn.Module = nn.Linear(lm_hidden, hidden_dim)
        else:
            self.res_embed = nn.Embedding(residue_vocab_size, hidden_dim, padding_idx=0)
            self.res_proj = nn.Identity()
        self.input_mix = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [_EGNNLayer(hidden_dim, edge_hidden) for _ in range(num_layers)]
        )
        self.out_norm = nn.LayerNorm(hidden_dim)

    def atom_features(self, z: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        """Initial per-atom features: atom embedding + (projected) residue embedding.

        Exposed so callers that bypass forward (V2's residue-pool path) get the
        same atom-level prior — including the PeptideCLM-2 projection when a
        pretrained residue embedding is in use.
        """
        return self.atom_embed(z) + self.res_proj(self.res_embed(res))

    def forward(
        self,
        coords: torch.Tensor,
        z: torch.Tensor,
        res: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Initial node features = atomic-number embedding + (projected) residue embedding.
        # atom_features 헬퍼를 거치게 해서 V2의 atom-level forward 경로(residue pool)와
        # 같은 처리가 보장되도록 한다.
        h = self.atom_features(z, res)  # (M, Nmax, F)
        h = self.input_mix(h) # linear layer로 섞어주기. 원자 번호와 잔기 정보가 섞여서 초기 노드 피처가 됨
        h = h * (~pad_mask).float().unsqueeze(-1) # 패딩 노드 0
        for layer in self.layers:
            h = layer(h, coords, pad_mask) # EGNN layer 통과하면서 노드 feature 업데이트
        h = self.out_norm(h) # 정규화
        # Mean-pool over real atoms per conformer.
        valid = (~pad_mask).float().unsqueeze(-1) # (M, Nmax, 1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0) # 패딩 된 원자 빼고 실제 원자만 풀링
        return pooled  # (M, hidden_dim)

"""Main ChameleonNet module.

Architecture:

  Water conformers   ─► ConformerEncoder (shared) ─► AttentionPool ─► h_water
  Hexane conformers  ─► ConformerEncoder (shared) ─► AttentionPool ─► h_hexane
                                                                   │
                                       chameleonic vector =        ▼
                                       [h_w, h_h, h_w - h_h]
  Sequence/SMILES    ─► SequenceEncoder ───────────────────────► h_seq
  4D descriptors     ─► DescriptorMLP ──────────────────────────► h_desc
                                                                   │
                                                                   ▼
                                          concat ─► head ─► PAMPA prediction
                                                          \\─► chameleonic-magnitude head
                                                          \\─► (h_water, h_hexane) returned
                                                              for env-contrast loss

The forward returns a dict so the loss module can pull whichever signals it
needs without forcing the trainer to thread several tensors manually.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..data.residue_vocab import ResidueVocab
from .attention_pool import ConformerAttentionPool
from .conformer_encoder import ConformerEncoder
from .descriptor_mlp import DescriptorMLP
from .sequence_encoder import SequenceEncoder


class ChameleonNet(nn.Module):
    def __init__(
        self,
        vocab: ResidueVocab,
        descriptor_dim: int,
        hidden_dim: int = 128,
        conformer_layers: int = 3,
        sequence_backend: str = "learned",
        peptideclm_name_or_path: Optional[str] = None,
        head_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Shared 3D encoder for both environments — sharing weights means the
        # only difference between h_water and h_hexane comes from the input
        # geometry, which is the chameleonic signal we want to capture.
        self.conformer_encoder = ConformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=conformer_layers,
            residue_vocab_size=len(vocab),
        )
        # Separate attention poolers per environment: solvents may weight
        # different conformers differently (a low-energy water conformer might
        # not be the dominant hexane conformer).
        self.water_pool = ConformerAttentionPool(hidden_dim)
        self.hexane_pool = ConformerAttentionPool(hidden_dim)

        self.sequence_encoder = SequenceEncoder(
            vocab=vocab,
            hidden_dim=hidden_dim,
            backend=sequence_backend,
            peptideclm_name_or_path=peptideclm_name_or_path,
        )
        self.descriptor_mlp = DescriptorMLP(in_dim=descriptor_dim, hidden_dim=hidden_dim)

        # Concatenated input to the head:
        #   h_water | h_hexane | (h_water - h_hexane) | h_seq | h_desc
        fused_dim = hidden_dim * 5
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # Auxiliary head: predicts the chameleonic magnitude scalar from the
        # difference vector. Trained with regression against PAMPA so the
        # ||h_w - h_h|| signal is pushed to align with permeability strength.
        self.chameleonic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode_environment(
        self,
        env: Dict[str, torch.Tensor],
        pool: ConformerAttentionPool,
    ) -> torch.Tensor:
        """Run the shared 3D encoder + the env-specific attention pooler."""
        h = self.conformer_encoder(
            coords=env["coords"],
            z=env["z"],
            res=env["res"],
            pad_mask=env["pad_mask"],
        )
        batch_size = int(env["sample_K"].shape[0])
        return pool(h, env["batch_index"], batch_size)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        h_water = self.encode_environment(batch["water"], self.water_pool)
        h_hexane = self.encode_environment(batch["hexane"], self.hexane_pool)
        h_diff = h_water - h_hexane

        h_seq = self.sequence_encoder(
            sequences=batch["sequences"], smiles=batch.get("smiles")
        )
        h_desc = self.descriptor_mlp(batch["descriptors"])

        fused = torch.cat([h_water, h_hexane, h_diff, h_seq, h_desc], dim=-1)
        pampa_pred = self.head(fused).squeeze(-1)
        cham_pred = self.chameleonic_head(h_diff).squeeze(-1)

        return {
            "pampa_pred": pampa_pred,
            "chameleonic_pred": cham_pred,
            "h_water": h_water,
            "h_hexane": h_hexane,
            "h_diff": h_diff,
        }

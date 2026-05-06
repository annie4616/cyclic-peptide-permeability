"""ChameleonNet V2 — residue-resolved chameleonic model.

V2 keeps the V1 dual-environment EGNN backbone and adds two things that V1
glosses over:

  1. Residue-level intermediate. Atoms get pooled to per-residue tokens after
     the 3D encoder. Cyclic-peptide permeability literature (and the EDA on
     this dataset) shows residue identity (Pro/Leu/MLe/Phe) carries most of
     the signal, so a residue-level representation is closer to the inductive
     bias of the problem than per-atom features alone.

  2. Residue-resolved chameleonic Δ. Instead of `||h_water - h_hexane||` on
     globally pooled vectors, we compute Δ per residue, score each residue's
     "chameleonic strength", and feed the attention-weighted Δ into the head
     in addition to the global pooled vectors. This makes the model identify
     *which residue* shifts most between solvents — which is exactly the
     mechanistic picture (e.g. an N-Me amide forming/breaking an
     intramolecular H-bond when the dielectric changes).

Δ-augmented descriptors (Water_3D_PSA - Hexane_3D_PSA, etc.) are a separate
branch handled at dataset level via `descriptor_cols` configuration. V2 just
expects the descriptor vector to already include those engineered diff
features. See `data/delta_descriptors.py`.

The forward signature is a superset of V1's, so the existing trainer,
composite loss, and triplet branch all work unchanged. V2-only outputs (e.g.
residue-resolved Δ tensor) ride along in the output dict for new aux losses
that we'll layer on later.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from ..data.residue_vocab import ResidueVocab
from .attention_pool import ConformerAttentionPool
from .conformer_encoder import ConformerEncoder
from .descriptor_mlp import DescriptorMLP
from .residue_pool import AtomToResiduePool, pool_residues_over_conformers
from .sequence_encoder import SequenceEncoder


class ResidueChameleonHead(nn.Module):
    """Attention over per-residue Δ to produce a global chameleonic vector.

    score_r = MLP([h_w[r], h_h[r], Δ[r]])
    weight  = softmax_r(score_r) over present residues
    h_chameleon = Σ_r weight[r] * Δ[r]

    Returns the global chameleon vector and the per-residue scores so they
    can be regressed against per-residue ΔPSA targets in the loss.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_water_res: torch.Tensor,   # (B, R, F)
        h_hexane_res: torch.Tensor,  # (B, R, F)
        res_mask: torch.Tensor,      # (B, R) True = pad/missing
    ) -> Dict[str, torch.Tensor]:
        delta = h_water_res - h_hexane_res
        x = torch.cat([h_water_res, h_hexane_res, delta], dim=-1)
        score = self.score(x).squeeze(-1)  # (B, R)
        score = score.masked_fill(res_mask, float("-inf"))
        weight = torch.softmax(score, dim=-1)
        # If a sample has zero residues (shouldn't happen) softmax produces
        # NaNs; nan_to_num as a guard.
        weight = torch.nan_to_num(weight, nan=0.0)
        h_chameleon = (weight.unsqueeze(-1) * delta).sum(dim=1)  # (B, F)
        return {
            "h_chameleon": h_chameleon,
            "delta_residue": delta,           # (B, R, F)
            "chameleon_weight": weight,       # (B, R) — interpretable
        }


class ChameleonNetV2(nn.Module):
    """Residue-resolved chameleonic model. Drop-in for V1 in the trainer."""

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

        # 3D encoder shared across environments (same reasoning as V1).
        self.conformer_encoder = ConformerEncoder(
            hidden_dim=hidden_dim,
            num_layers=conformer_layers,
            residue_vocab_size=len(vocab),
        )
        # We need atom-level h to do residue pooling. The current
        # ConformerEncoder mean-pools internally; we monkeypatch by calling a
        # custom forward that returns the per-atom features pre-pool.
        # Simpler: re-implement the pooled forward here with the encoder's
        # internal layers.
        # NOTE: ConformerEncoder doesn't expose an "atom features" output, so
        # we wrap it to capture the pre-pool tensor.

        self.atom_to_residue = AtomToResiduePool(hidden_dim)

        # Per-environment global pooler (kept for the global-vector branch).
        self.water_pool = ConformerAttentionPool(hidden_dim)
        self.hexane_pool = ConformerAttentionPool(hidden_dim)

        self.chameleon_head = ResidueChameleonHead(hidden_dim)

        self.sequence_encoder = SequenceEncoder(
            vocab=vocab,
            hidden_dim=hidden_dim,
            backend=sequence_backend,
            peptideclm_name_or_path=peptideclm_name_or_path,
        )
        self.descriptor_mlp = DescriptorMLP(in_dim=descriptor_dim, hidden_dim=hidden_dim)

        # Final-head input: [h_w | h_h | h_diff_global | h_chameleon | h_seq | h_desc]
        # h_diff_global is kept so V2 strictly contains V1's fusion as a subset
        # — this makes the V2 head able to fall back to V1 behavior if the
        # residue branch ends up degenerate.
        fused_dim = hidden_dim * 6
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

        # Auxiliary head against PAMPA from the residue-resolved chameleonic
        # vector (V1 has the same idea but on the global diff).
        self.chameleonic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ------------------------------------------------------------------ utils

    def _atom_features(self, env: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the conformer encoder up to the per-atom features (pre-pool).

        The vendored ConformerEncoder pools internally; we replicate its inner
        pipeline here so we can hand the atom-level tensor to the residue
        pooler. Keeps V1 untouched at the cost of a tiny code duplication.
        """
        enc = self.conformer_encoder
        h = enc.atom_embed(env["z"]) + enc.res_embed(env["res"])
        h = enc.input_mix(h)
        pad_mask = env["pad_mask"]
        h = h * (~pad_mask).float().unsqueeze(-1)
        for layer in enc.layers:
            h = layer(h, env["coords"], pad_mask)
        h = enc.out_norm(h)
        return h  # (M, Nmax, F)

    def _encode_environment(
        self,
        env: Dict[str, torch.Tensor],
        pool: ConformerAttentionPool,
        max_residues: int,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        h_atom = self._atom_features(env)  # (M, Nmax, F)

        # Global per-conformer mean pool (mirrors V1 exactly).
        valid = (~env["pad_mask"]).float().unsqueeze(-1)
        h_conf = (h_atom * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # (M, F)
        h_global = pool(h_conf, env["batch_index"], batch_size)  # (B, F)

        # Residue pooling — needs per-atom residue position id.
        res_pos = env.get("res_pos")
        if res_pos is None:
            # No positional ids → residue branch is disabled; return zeros.
            B = batch_size
            R = max(1, max_residues)
            h_res_pep = h_atom.new_zeros(B, R, h_atom.shape[-1])
            res_mask = torch.ones(B, R, dtype=torch.bool, device=h_atom.device)
            return {"h_global": h_global, "h_res": h_res_pep, "res_mask": res_mask}

        h_res_conf, conf_res_mask = self.atom_to_residue(
            h_atom=h_atom, res_pos=res_pos, pad_mask=env["pad_mask"],
            max_residues=max_residues,
        )
        h_res_pep, pep_res_mask = pool_residues_over_conformers(
            h_res=h_res_conf, res_mask=conf_res_mask,
            batch_index=env["batch_index"], batch_size=batch_size,
        )
        return {"h_global": h_global, "h_res": h_res_pep, "res_mask": pep_res_mask}

    # ------------------------------------------------------------------ forward

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        water = batch["water"]
        hexane = batch["hexane"]

        batch_size = int(water["sample_K"].shape[0])
        # R_max is per-environment in principle, but the same peptide should
        # have the same residue count in water and hexane. Take the max for
        # safety; pooled tensors will mask the extra slots.
        rmax_w = int(water.get("max_residues", 0))
        rmax_h = int(hexane.get("max_residues", 0))
        max_residues = max(rmax_w, rmax_h, 1)

        env_w = self._encode_environment(water, self.water_pool, max_residues, batch_size)
        env_h = self._encode_environment(hexane, self.hexane_pool, max_residues, batch_size)

        h_water = env_w["h_global"]
        h_hexane = env_h["h_global"]
        h_diff = h_water - h_hexane  # (B, F) — kept for V1 compatibility

        # Residue alignment: water and hexane parses don't always agree on the
        # padded R_max if one PDB has a stray HETATM, so AND the masks.
        res_mask = env_w["res_mask"] | env_h["res_mask"]
        cham = self.chameleon_head(env_w["h_res"], env_h["h_res"], res_mask)
        h_chameleon = cham["h_chameleon"]  # (B, F)

        h_seq = self.sequence_encoder(
            sequences=batch["sequences"], smiles=batch.get("smiles")
        )
        h_desc = self.descriptor_mlp(batch["descriptors"])

        fused = torch.cat(
            [h_water, h_hexane, h_diff, h_chameleon, h_seq, h_desc], dim=-1
        )
        pampa_pred = self.head(fused).squeeze(-1)
        cham_pred = self.chameleonic_head(h_chameleon).squeeze(-1)

        # Output dict — the first five keys are the V1-compatible signals so
        # the existing composite loss / trainer keeps working as-is.
        return {
            "pampa_pred": pampa_pred,
            "chameleonic_pred": cham_pred,
            "h_water": h_water,
            "h_hexane": h_hexane,
            "h_diff": h_diff,
            # V2-only signals (consumed by V2 aux losses; ignored by V1 loss)
            "h_chameleon": h_chameleon,
            "delta_residue": cham["delta_residue"],
            "chameleon_weight": cham["chameleon_weight"],
            "res_mask": res_mask,
        }

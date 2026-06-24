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
from .adversary import ScaffoldAdversary
from .attention_pool import ConformerAttentionPool
from .conformer_encoder import ConformerEncoder
from .nsegnn_encoder import NSEGNNConformerEncoder
from .descriptor_mlp import DescriptorMLP
from .sequence_encoder import SequenceEncoder
from .transformer_pool import ConformerTransformerPool


class ChameleonNet(nn.Module):
    def __init__(
        self,
        vocab: ResidueVocab,
        descriptor_dim: int,
        hidden_dim: int = 128,
        conformer_layers: int = 3,
        sequence_backend: str = "learned",
        peptideclm_name_or_path: Optional[str] = None,
        helmbert_name_or_path: Optional[str] = None,
        head_hidden: int = 256,
        dropout: float = 0.1,
        residue_emb_path: Optional[str] = None,
        n_scaffold_groups: int = 0,
        adv_hidden: int = 128,
        modality_dropout: float = 0.0,
        physics_residual: bool = False,
        info_bottleneck: bool = False,
        gbr_residual: bool = False,
        distill: bool = False,
        hexane_only: bool = False,
        water_only: bool = False,
        no_diff: bool = False,
        no_conformer: bool = False,
        conformer_encoder_arch: str = "egnn",
        pool_type: str = "attention",
        pool_transformer_layers: int = 2,
        pool_transformer_heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Hexane-only ablation: encode ONLY the hexane conformer ensemble with the
        # EGNN. The water branch is skipped entirely and h_water / h_diff are held
        # at zero, so the chameleonic (water-vs-hexane contrast) signal is absent —
        # the trainer forces lambda_chameleonic=0 in this mode. Fusion layout is
        # kept identical so the head/checkpoint shapes are unchanged.
        self.hexane_only = bool(hexane_only)
        # Water-only ablation: symmetric to hexane_only — encode ONLY the water
        # conformer ensemble; the hexane branch is skipped and h_hexane/h_diff
        # held at zero (chameleonic contrast absent, lambda forced to 0).
        self.water_only = bool(water_only)
        # No-diff ablation: keep BOTH environment encoders (h_water, h_hexane) but
        # remove the chameleonic difference branch — h_diff is held at zero in the
        # fusion vector and the chameleonic head/loss are inert (lambda -> 0).
        # Tests whether the explicit (water - hexane) contrast adds anything over
        # having both environment embeddings available.
        self.no_diff = bool(no_diff)
        # No-conformer ablation: skip BOTH 3D environment encoders entirely
        # (h_water/h_hexane/h_diff held at zero) — prediction uses only the
        # sequence + descriptor branches. Tests whether the 3D conformer signal
        # contributes anything over sequence+descriptors. ("0 conformers")
        self.no_conformer = bool(no_conformer)
        self.modality_dropout = float(modality_dropout)
        self.physics_residual = bool(physics_residual)
        self.info_bottleneck = bool(info_bottleneck)
        self.gbr_residual = bool(gbr_residual)
        self.distill = bool(distill)

        # Shared 3D encoder for both environments — sharing weights means the
        # only difference between h_water and h_hexane comes from the input
        # geometry, which is the chameleonic signal we want to capture.
        #   "egnn"     — default SchNet-ish EGNN, per-conformer encode + pool.
        #   "ns_egnn"  — Non-Stationary EGNN: equivariant E_GCL + multi-scale STFT
        #     over the conformer trajectory; returns a per-molecule env embedding
        #     directly (the AttentionPool path is bypassed for this arch).
        self.conformer_encoder_arch = str(conformer_encoder_arch)
        if self.conformer_encoder_arch == "ns_egnn":
            self.conformer_encoder = NSEGNNConformerEncoder(
                hidden_dim=hidden_dim, num_layers=conformer_layers,
                residue_vocab_size=len(vocab),
                residue_vocab_tokens=list(vocab._tokens),
            )
        else:
            self.conformer_encoder = ConformerEncoder(
                hidden_dim=hidden_dim,
                num_layers=conformer_layers,
                residue_vocab_size=len(vocab),
                residue_emb_path=residue_emb_path,
                residue_vocab_tokens=list(vocab._tokens),
            )
        # Separate poolers per environment: solvents may weight different
        # conformers differently (a low-energy water conformer might not be the
        # dominant hexane conformer).
        #   "attention"  — permutation-invariant softmax bag (default; original).
        #   "transformer"— treats each peptide's conformers as a trajectory-
        #     ordered sequence ([CLS] h_0 h_1 ...) with positional embeddings and
        #     a small TransformerEncoder; the CLS output is the env vector.
        self.pool_type = str(pool_type)
        if self.pool_type == "transformer":
            def _mk_pool() -> nn.Module:
                return ConformerTransformerPool(
                    hidden_dim,
                    num_layers=pool_transformer_layers,
                    nhead=pool_transformer_heads,
                    dropout=dropout,
                )
        elif self.pool_type == "attention":
            def _mk_pool() -> nn.Module:
                return ConformerAttentionPool(hidden_dim)
        else:
            raise ValueError(
                f"pool_type must be 'attention' or 'transformer', got {pool_type!r}"
            )
        self.water_pool = _mk_pool()
        self.hexane_pool = _mk_pool()

        self.sequence_encoder = SequenceEncoder(
            vocab=vocab,
            hidden_dim=hidden_dim,
            backend=sequence_backend,
            peptideclm_name_or_path=peptideclm_name_or_path,
            helmbert_name_or_path=helmbert_name_or_path,
        )
        self.descriptor_mlp = DescriptorMLP(in_dim=descriptor_dim, hidden_dim=hidden_dim)

        # Learned representation = [h_water | h_hexane | h_diff | h_seq] (4*hidden);
        # physics descriptors h_desc (hidden) are kept on a separate, privileged
        # path so the OOD-robust signal is never bottlenecked or dropped.
        learned_dim = hidden_dim * 4

        # Optional variational information bottleneck on the learned vector.
        if self.info_bottleneck:
            self.ib_proj = nn.Linear(learned_dim, 2 * learned_dim)

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

        # Physics-residual head: main prediction is driven by the descriptor
        # branch; the learned branches contribute only a gated residual (gate
        # starts near zero), so OOD gracefully falls back onto transferable
        # physics when the learned representation doesn't generalize.
        if self.physics_residual:
            self.desc_main_head = nn.Sequential(
                nn.Linear(hidden_dim, head_hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            self.resid_head = nn.Sequential(
                nn.Linear(learned_dim, head_hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            self.resid_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2)≈0.12

        # GBR-residual head: prediction = gbr_pred + gated learned residual, so
        # the deep net only models what the descriptor-GBR misses. Reuses a
        # learned-rep residual head + gate (built here if not already present).
        if self.gbr_residual:
            self.gbr_resid_head = nn.Sequential(
                nn.Linear(learned_dim, head_hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(head_hidden, 1),
            )
            self.gbr_gate = nn.Parameter(torch.tensor(-2.0))

        # Distillation head: a learned-only PAMPA prediction pulled toward the
        # GBR target (anchors the encoder to the transferable function).
        if self.distill:
            self.distill_head = nn.Sequential(
                nn.Linear(learned_dim, head_hidden), nn.SiLU(),
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

        # Optional scaffold adversary (gradient-reversal). Reads only the
        # *learned* representation [h_water | h_hexane | h_diff | h_seq] — the
        # physics descriptors h_desc are kept out of the adversarial path so
        # their transferable signal is preserved. `adv_lambda` is the GRL
        # strength; the trainer warms it 0 -> max over training. Disabled
        # (and architecturally absent) when n_scaffold_groups <= 0, so existing
        # baselines are byte-for-byte unchanged.
        self.n_scaffold_groups = int(n_scaffold_groups)
        self.adv_lambda: float = 0.0
        if self.n_scaffold_groups > 0:
            self.scaffold_adversary = ScaffoldAdversary(
                in_dim=hidden_dim * 4,
                num_groups=self.n_scaffold_groups,
                hidden=adv_hidden,
                dropout=dropout,
            )
        else:
            self.scaffold_adversary = None

    def encode_environment(
        self,
        env: Dict[str, torch.Tensor],
        pool: nn.Module,
    ) -> torch.Tensor:
        """Run the shared 3D encoder + the env-specific attention pooler.

        When `env` carries per-conformer cluster weights (centroid mode), we
        pass them in as a log-prior so larger basins get more attention by
        default — same path V2 uses.
        """
        # NS-EGNN encodes the whole conformer trajectory per molecule and returns
        # a per-molecule env vector directly (no separate pooling step).
        if self.conformer_encoder_arch == "ns_egnn":
            return self.conformer_encoder.forward_env(
                coords=env["coords"], z=env["z"], res=env["res"],
                pad_mask=env["pad_mask"], batch_index=env["batch_index"],
                batch_size=int(env["sample_K"].shape[0]),
            )
        h = self.conformer_encoder(
            coords=env["coords"],
            z=env["z"],
            res=env["res"],
            pad_mask=env["pad_mask"],
        )
        batch_size = int(env["sample_K"].shape[0])
        weights = env.get("weights")
        log_prior = torch.log(weights.clamp_min(1e-6)) if weights is not None else None
        return pool(h, env["batch_index"], batch_size, log_prior=log_prior)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        if self.no_conformer:
            # Skip both 3D encoders; build zero env embeddings of shape (B, hidden).
            B = len(batch["sequences"])
            dev = batch["descriptors"].device
            h_water = torch.zeros(B, self.hidden_dim, device=dev)
            h_hexane = torch.zeros(B, self.hidden_dim, device=dev)
        elif self.water_only:
            # Encode ONLY water; hold hexane/diff slots at zero (symmetric to
            # hexane_only). Fusion layout unchanged so head/checkpoint shapes match.
            h_water = self.encode_environment(batch["water"], self.water_pool)
            h_hexane = torch.zeros_like(h_water)
        else:
            h_hexane = self.encode_environment(batch["hexane"], self.hexane_pool)
            if self.hexane_only:
                # Skip the water EGNN pass entirely; hold the water/diff slots at zero
                # so downstream fusion shapes are unchanged but carry no water signal.
                h_water = torch.zeros_like(h_hexane)
            else:
                h_water = self.encode_environment(batch["water"], self.water_pool)

        h_seq = self.sequence_encoder(
            sequences=batch["sequences"],
            smiles=batch.get("smiles"),
            helms=batch.get("helms"),
        )
        h_desc = self.descriptor_mlp(batch["descriptors"])

        # Modality dropout (train only): independently zero the conformer block
        # and the sequence block so the model can't rely solely on the
        # scaffold-memorising learned branches. Descriptors are never dropped.
        if self.training and self.modality_dropout > 0:
            if torch.rand((), device=h_water.device) < self.modality_dropout:
                h_water = torch.zeros_like(h_water)
                h_hexane = torch.zeros_like(h_hexane)
            if torch.rand((), device=h_seq.device) < self.modality_dropout:
                h_seq = torch.zeros_like(h_seq)

        # In single-environment modes the chameleonic contrast is undefined; keep
        # h_diff at zero so the chameleonic head/loss stay inert (lambda is also
        # forced to 0 by the trainer) and no single-env signal leaks via the diff.
        h_diff = (torch.zeros_like(h_water)
                  if (self.hexane_only or self.water_only or self.no_diff or self.no_conformer)
                  else (h_water - h_hexane))
        learned = torch.cat([h_water, h_hexane, h_diff, h_seq], dim=-1)

        out: Dict[str, torch.Tensor] = {
            "h_water": h_water,
            "h_hexane": h_hexane,
            "h_diff": h_diff,
        }

        # Adversary reads the (pre-bottleneck) learned representation.
        if self.scaffold_adversary is not None:
            out["scaffold_logits"] = self.scaffold_adversary(learned, self.adv_lambda)

        # Variational information bottleneck on the learned representation.
        if self.info_bottleneck:
            mu, logvar = self.ib_proj(learned).chunk(2, dim=-1)
            logvar = logvar.clamp(-8.0, 8.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                learned = mu + std * torch.randn_like(std)
            else:
                learned = mu
            out["ib_kl"] = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()

        if self.gbr_residual:
            gbr = batch["gbr_pred"]
            resid = self.gbr_resid_head(learned).squeeze(-1)
            pampa_pred = gbr + torch.sigmoid(self.gbr_gate) * resid
            out["resid"] = resid
        elif self.physics_residual:
            main = self.desc_main_head(h_desc).squeeze(-1)
            resid = self.resid_head(learned).squeeze(-1)
            pampa_pred = main + torch.sigmoid(self.resid_gate) * resid
            out["resid"] = resid
        else:
            fused = torch.cat([learned, h_desc], dim=-1)
            pampa_pred = self.head(fused).squeeze(-1)

        if self.distill:
            out["distill_pred"] = self.distill_head(learned).squeeze(-1)

        out["pampa_pred"] = pampa_pred
        out["chameleonic_pred"] = self.chameleonic_head(h_diff).squeeze(-1)
        return out

"""Sequence encoder branch.

Three modes:
  1. peptideclm — wrap a HuggingFace PeptideCLM checkpoint (or any masked-LM
     SMILES encoder) and pool its [CLS]/mean output. This is the path that
     reuses MCPerm's pretrained representations.
  2. helmbert   — wrap a HuggingFace HELM-BERT checkpoint that tokenizes raw
     HELM strings (Flansma/helm-bert). Same pooling path as peptideclm but
     the batch must carry "helms" instead of "smiles".
  3. learned    — fall back to a small Transformer over the residue id
     sequence (using our ResidueVocab). Useful for a self-contained run that
     doesn't require downloading PeptideCLM weights.

We default to mode 3 to keep this module importable even on a fresh machine,
and gate modes 1/2 behind transformers being installed and a checkpoint being
provided. The trainer config selects between them.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..data.residue_vocab import ResidueVocab


class _LearnedSequenceEncoder(nn.Module):
    """Tiny Transformer over residue-id sequences."""

    def __init__(
        self,
        vocab: ResidueVocab,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 4,
        max_len: int = 64,
    ):
        super().__init__()
        self.vocab = vocab
        self.max_len = max_len
        self.embed = nn.Embedding(len(vocab), hidden_dim, padding_idx=vocab.pad_id)
        self.pos = nn.Embedding(max_len, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim * 2,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, sequences: List[List[str]]) -> torch.Tensor:
        device = self.cls.device
        B = len(sequences)
        # Reserve slot 0 for CLS — encode residues into slots 1..L.
        L = min(self.max_len - 1, max((len(s) for s in sequences), default=0))
        ids = torch.full((B, L + 1), self.vocab.pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(sequences):
            for j, name in enumerate(seq[:L]):
                ids[i, j + 1] = self.vocab.encode(name)
        emb = self.embed(ids)
        emb[:, 0:1, :] = self.cls.expand(B, 1, -1)
        positions = torch.arange(L + 1, device=device).unsqueeze(0).expand(B, -1)
        emb = emb + self.pos(positions)
        key_padding_mask = ids == self.vocab.pad_id
        # The CLS slot must not be masked even if all residues are pad.
        key_padding_mask[:, 0] = False
        out = self.encoder(emb, src_key_padding_mask=key_padding_mask)
        return self.norm(out[:, 0])  # (B, hidden_dim)


class SequenceEncoder(nn.Module):
    """Dispatches to a learned-from-scratch or PeptideCLM-backed encoder."""

    def __init__(
        self,
        vocab: ResidueVocab,
        hidden_dim: int = 128,
        backend: str = "learned",
        peptideclm_name_or_path: Optional[str] = None,
        helmbert_name_or_path: Optional[str] = None,
    ):
        super().__init__()
        self.backend = backend
        self.hidden_dim = hidden_dim

        if backend == "learned":
            self.impl = _LearnedSequenceEncoder(vocab=vocab, hidden_dim=hidden_dim)
        elif backend in {"peptideclm", "helmbert"}:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as e:
                raise ImportError(
                    f"backend={backend!r} requires `transformers`. Install it "
                    "or set backend='learned'."
                ) from e
            if backend == "peptideclm":
                name = peptideclm_name_or_path
                if name is None:
                    raise ValueError(
                        "peptideclm_name_or_path must be provided when backend='peptideclm'."
                    )
            else:
                name = helmbert_name_or_path
                if name is None:
                    raise ValueError(
                        "helmbert_name_or_path must be provided when backend='helmbert'."
                    )
            self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            self.lm = AutoModel.from_pretrained(name, trust_remote_code=True)
            # Freeze the pretrained backbone so only the projection head learns.
            # This mirrors the residue-embedding branch (which uses a frozen
            # PeptideCLM-2 lookup) and keeps the two branches in a consistent
            # representation space. Eval mode disables dropout in the LM so the
            # frozen forward is deterministic.
            for p in self.lm.parameters():
                p.requires_grad_(False)
            self.lm.eval()
            # Standard HF models expose `hidden_size`; PeptideCLM-2 uses `embed_dim`.
            lm_hidden = (
                getattr(self.lm.config, "hidden_size", None)
                or getattr(self.lm.config, "embed_dim", None)
                or getattr(self.lm.config, "d_model", None)
            )
            if lm_hidden is None:
                raise ValueError(
                    f"Could not determine hidden dim of {name}; "
                    f"expected one of hidden_size/embed_dim/d_model on its config."
                )
            self.proj = nn.Linear(lm_hidden, hidden_dim)
        else:
            raise ValueError(f"Unknown sequence backend: {backend}")

    def forward(
        self,
        sequences: List[List[str]],
        smiles: Optional[List[str]] = None,
        helms: Optional[List[str]] = None,
    ) -> torch.Tensor:
        if self.backend == "learned":
            return self.impl(sequences)

        if self.backend == "peptideclm":
            if smiles is None:
                raise ValueError("backend='peptideclm' needs smiles in the batch.")
            inputs = smiles
        else:  # helmbert
            if helms is None:
                raise ValueError("backend='helmbert' needs helms in the batch.")
            inputs = helms

        device = next(self.lm.parameters()).device
        toks = self.tokenizer(
            inputs, padding=True, truncation=True, return_tensors="pt"
        ).to(device)
        # The LM is frozen — force eval (model.train() would otherwise re-enable
        # dropout on it) and skip the autograd graph for the backbone so we
        # don't allocate activations we'll never differentiate through.
        self.lm.eval()
        with torch.no_grad():
            out = self.lm(**toks)
            mask = toks["attention_mask"].unsqueeze(-1).float()
            h = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return self.proj(h)

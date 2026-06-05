"""Pre-extract per-residue PeptideCLM embeddings.

Pipeline:
  1. Load the union vocab from EDA CSVs (same source as ResidueVocab.from_csvs).
  2. Discover the PDB-residue-code -> HELM-monomer-code mapping by aligning each
     PDB to its assay-CSV Sequence list (majority vote over the dataset).
  3. Look up each HELM monomer in the official CycPeptMPDB monomer table to get
     a SMILES (`replaced_SMILES` column). Fall back to a small curated dictionary
     for terminal caps that aren't in the table.
  4. Run PeptideCLM on each residue SMILES one at a time, mean-pool over real
     tokens, and stack the results into a (vocab_size, lm_hidden) tensor.
  5. Save the tensor plus the ordered token list to a .pt file. The embedding
     row at the vocab pad_id (=0) is forced to zero so `nn.Embedding.from_pretrained(
     ..., padding_idx=0)` behaves correctly.

The output is intended to be loaded by ConformerEncoder via
`nn.Embedding.from_pretrained(weights, freeze=..., padding_idx=0)` followed by
an optional `Linear(lm_hidden, hidden_dim)` projection.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow importing the project's ResidueVocab so we use the exact same id order.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from chameleonnet.data.residue_vocab import ResidueVocab  # noqa: E402


# ----------------------------------------------------------------------------
# Step 1+2 — vocab and PDB <-> HELM alignment
# ----------------------------------------------------------------------------

_PDB_FNAME_RE = re.compile(r"^(?P<src>.+)_(?P<pid>\d+)_H2O_Str\.pdb$")


def _pdb_residue_sequence(path: Path) -> List[str]:
    """Walk a PDB once and return its per-residue 3/4-letter code stream."""
    seq: List[str] = []
    last: Optional[Tuple[str, str]] = None
    with open(path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            r = line[17:21].strip()
            s = line[22:26].strip()
            key = (r, s)
            if key != last:
                seq.append(r)
                last = key
    return seq


def _load_assay_sequences(csv_path: Path) -> Dict[Tuple[str, str], List[str]]:
    """(Source, CycPeptMPDB_ID) -> HELM-style residue token list."""
    out: Dict[Tuple[str, str], List[str]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            src = (row.get("Source") or "").strip()
            pid = (row.get("CycPeptMPDB_ID") or "").strip()
            try:
                seq = ast.literal_eval(row.get("Sequence") or "")
            except (ValueError, SyntaxError):
                continue
            if isinstance(seq, list):
                out[(src, pid)] = [str(x) for x in seq]
    return out


def build_pdb_to_helm_mapping(
    pdb_dir: Path, assay_csv: Path
) -> Dict[str, Counter]:
    """Majority-vote mapping from a PDB residue code to a HELM monomer.

    Two alignment passes:
      (a) strict equal-length match after stripping `*-` / `-*` cap tokens from
          the assay sequence — for backbone residues.
      (b) length-off-by-1 match — for terminal caps (PDB has an extra residue
          like ACE which assay represents as `ac-`).
    """
    seqs = _load_assay_sequences(assay_csv)
    mapping: Dict[str, Counter] = defaultdict(Counter)
    for path in pdb_dir.glob("*_H2O_Str.pdb"):
        m = _PDB_FNAME_RE.match(path.name)
        if m is None:
            continue
        assay_seq = seqs.get((m.group("src"), m.group("pid")))
        if assay_seq is None:
            continue
        pdb_seq = _pdb_residue_sequence(path)

        # Pass (a): backbone alignment with caps stripped.
        a = list(assay_seq)
        while a and a[0].endswith("-"):
            a.pop(0)
        while a and a[-1].startswith("-"):
            a.pop()
        if len(a) == len(pdb_seq):
            for pdb_code, helm_code in zip(pdb_seq, a):
                mapping[pdb_code][helm_code] += 1
            continue

        # Pass (b): cap residue is the extra one in the PDB stream.
        if len(assay_seq) == len(pdb_seq) - 1:
            if pdb_seq[0] not in pdb_seq[1:]:  # head cap
                mapping[pdb_seq[0]][assay_seq[0]] += 1
            if pdb_seq[-1] not in pdb_seq[:-1]:  # tail cap
                mapping[pdb_seq[-1]][assay_seq[-1]] += 1
    return mapping


# ----------------------------------------------------------------------------
# Step 3 — HELM monomer -> SMILES
# ----------------------------------------------------------------------------

# Last-resort fallbacks for tokens that aren't in the monomer table. These are
# canonical residues / common terminal caps; we use neutral methyl-amide-like
# capped SMILES so PeptideCLM sees a plausible chemical context.
_CANONICAL_SMILES: Dict[str, str] = {
    "A": "C[C@H](N)C(=O)O",
    "R": "N[C@@H](CCCNC(=N)N)C(=O)O",
    "N": "N[C@@H](CC(=O)N)C(=O)O",
    "D": "N[C@@H](CC(=O)O)C(=O)O",
    "C": "N[C@@H](CS)C(=O)O",
    "Q": "N[C@@H](CCC(=O)N)C(=O)O",
    "E": "N[C@@H](CCC(=O)O)C(=O)O",
    "G": "NCC(=O)O",
    "H": "N[C@@H](Cc1cnc[nH]1)C(=O)O",
    "I": "CC[C@H](C)[C@H](N)C(=O)O",
    "L": "CC(C)C[C@H](N)C(=O)O",
    "K": "NCCCC[C@H](N)C(=O)O",
    "M": "CSCC[C@H](N)C(=O)O",
    "F": "N[C@@H](Cc1ccccc1)C(=O)O",
    "P": "OC(=O)[C@@H]1CCCN1",
    "S": "N[C@@H](CO)C(=O)O",
    "T": "C[C@@H](O)[C@H](N)C(=O)O",
    "W": "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O",
    "Y": "N[C@@H](Cc1ccc(O)cc1)C(=O)O",
    "V": "CC(C)[C@H](N)C(=O)O",
    # caps:
    "ac-": "CC(=O)N",
    "-pip": "N1CCCCC1",
    "-NH2": "N",
    "-OH": "O",
}


def load_monomer_smiles(monomer_csv: Path) -> Dict[str, str]:
    """Read the CycPeptMPDB monomer table and return Symbol -> SMILES.

    Prefer `replaced_SMILES` (no attachment-point markers, ready for RDKit);
    fall back to stripping CXSMILES if the replaced column is missing.
    """
    table: Dict[str, str] = {}
    with open(monomer_csv, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sym = (row.get("Symbol") or "").strip()
            if not sym:
                continue
            smi = (row.get("replaced_SMILES") or "").strip()
            if not smi:
                cx = (row.get("CXSMILES") or "").strip()
                # CXSMILES carries attachment-point sgroups in `|...|`; cut them
                # so RDKit can parse it as a plain SMILES.
                smi = cx.split("|", 1)[0].strip().replace("[*]", "[H]")
            if smi:
                table[sym] = smi
    return table


def resolve_smiles(
    helm_code: str, monomer_smiles: Dict[str, str]
) -> Optional[str]:
    """Pick a SMILES for a HELM monomer token, with cascading fallbacks."""
    if helm_code in monomer_smiles:
        return monomer_smiles[helm_code]
    if helm_code in _CANONICAL_SMILES:
        return _CANONICAL_SMILES[helm_code]
    # `dX` for an unknown canonical X — treat as the L-form (chirality info is
    # discarded but at least we keep the residue type prior).
    if len(helm_code) == 2 and helm_code.startswith("d") and helm_code[1] in _CANONICAL_SMILES:
        return _CANONICAL_SMILES[helm_code[1]]
    return None


def vocab_to_smiles(
    vocab: ResidueVocab,
    pdb_to_helm: Dict[str, Counter],
    monomer_smiles: Dict[str, str],
    fallback_overrides: Dict[str, str],
) -> Tuple[List[Optional[str]], List[str]]:
    """For each vocab token (in id order), pick the most-common HELM monomer
    and resolve it to a SMILES. Returns (smiles_per_id, helm_code_per_id)."""
    smiles_per_id: List[Optional[str]] = [None] * len(vocab)
    helm_per_id: List[str] = [""] * len(vocab)
    for token, idx in sorted(vocab._index.items(), key=lambda kv: kv[1]):
        if idx == vocab.pad_id:
            continue
        # Manual override wins over automatic alignment.
        if token in fallback_overrides:
            smi = fallback_overrides[token]
            if smi:
                smiles_per_id[idx] = smi
                helm_per_id[idx] = f"<override:{token}>"
            continue
        # Pick the dominant HELM monomer code that aligned with this PDB code.
        counter = pdb_to_helm.get(token)
        if counter:
            helm_code = counter.most_common(1)[0][0]
            helm_per_id[idx] = helm_code
            smi = resolve_smiles(helm_code, monomer_smiles)
            if smi is None and token in _CANONICAL_SMILES:
                smi = _CANONICAL_SMILES[token]
            smiles_per_id[idx] = smi
        else:
            # No PDB alignment — last-resort: token itself in canonical dict?
            smiles_per_id[idx] = _CANONICAL_SMILES.get(token)
            helm_per_id[idx] = f"<noalign:{token}>"
    return smiles_per_id, helm_per_id


def vocab_to_helm_codes(
    vocab: ResidueVocab,
    pdb_to_helm: Dict[str, Counter],
    overrides: Dict[str, str],
) -> Tuple[List[Optional[str]], List[str]]:
    """For each vocab token, pick the dominant HELM monomer code string.

    HELM-BERT was pretrained on HELM monomer tokens directly, so the input it
    expects per residue is a short string like 'A', 'dP', 'meA', 'Mono39',
    'Et_Gly' — not a SMILES. Returns (text_per_id, source_per_id) where
    source_per_id is purely diagnostic and labels how each token was resolved.
    """
    text_per_id: List[Optional[str]] = [None] * len(vocab)
    source_per_id: List[str] = [""] * len(vocab)
    for token, idx in sorted(vocab._index.items(), key=lambda kv: kv[1]):
        if idx == vocab.pad_id:
            continue
        if token in overrides:
            text_per_id[idx] = overrides[token]
            source_per_id[idx] = f"<override:{token}>"
            continue
        counter = pdb_to_helm.get(token)
        if counter:
            helm_code = counter.most_common(1)[0][0]
            text_per_id[idx] = helm_code
            source_per_id[idx] = helm_code
        else:
            # Last resort: use the PDB residue code itself as the input string.
            # HELM-BERT can still produce a non-trivial embedding for it via its
            # character-level tokenizer fallback.
            text_per_id[idx] = token
            source_per_id[idx] = f"<noalign:{token}>"
    return text_per_id, source_per_id


# ----------------------------------------------------------------------------
# Step 4 — PeptideCLM embedding extraction
# ----------------------------------------------------------------------------


def extract_embeddings(
    smiles_per_id: List[Optional[str]],
    model_name: str,
    device: str,
    batch_size: int,
) -> Tuple[torch.Tensor, List[bool]]:
    """Return (V, lm_hidden) embeddings + a boolean mask of which ids actually
    got a meaningful (non-fallback) embedding."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()

    cfg = model.config
    lm_hidden = (
        getattr(cfg, "hidden_size", None)
        or getattr(cfg, "embed_dim", None)
        or getattr(cfg, "d_model", None)
    )
    if lm_hidden is None:
        raise RuntimeError(
            f"Could not determine hidden dim of {model_name}; "
            f"expected hidden_size/embed_dim/d_model on its config."
        )

    V = len(smiles_per_id)
    out = torch.zeros((V, lm_hidden), dtype=torch.float32)
    has_emb = [False] * V

    # Process in mini-batches over the valid SMILES indices.
    valid_idx = [i for i, s in enumerate(smiles_per_id) if s]
    with torch.no_grad():
        for start in range(0, len(valid_idx), batch_size):
            chunk = valid_idx[start : start + batch_size]
            batch_smiles = [smiles_per_id[i] for i in chunk]
            toks = tokenizer(
                batch_smiles,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            outputs = model(**toks)
            # Mean-pool over non-pad tokens.
            mask = toks["attention_mask"].unsqueeze(-1).float()
            h = (outputs.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            h = h.detach().cpu().float()
            for j, idx in enumerate(chunk):
                out[idx] = h[j]
                has_emb[idx] = True
    return out, has_emb


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--water-vocab",
        default="/hdd0/sohyun/cyclic-peptide-permeability/eda/water_residue_vocab.csv",
    )
    p.add_argument(
        "--hexane-vocab",
        default="/hdd0/sohyun/cyclic-peptide-permeability/eda/hexane_residue_vocab.csv",
    )
    p.add_argument(
        "--pdb-dir",
        default="/hdd0/sohyun/cyclic-peptide-permeability/CycPeptMPDB-4D/Water/Structures",
    )
    p.add_argument(
        "--assay-csv",
        default=(
            "/hdd0/sohyun/cyclic-peptide-permeability/data/"
            "CycPeptMPDB-4D_with_assay_descriptors_preprocessed.csv"
        ),
    )
    p.add_argument(
        "--monomer-csv",
        default=str(_HERE.parent / "data" / "residue_embeddings" / "CycPeptMPDB_Monomer_All.csv"),
    )
    p.add_argument(
        "--backend",
        choices=["peptideclm", "helmbert"],
        default="peptideclm",
        help=(
            "Which sequence-branch backend to extract residue embeddings for. "
            "`peptideclm` feeds the LM a per-residue SMILES; `helmbert` feeds "
            "it the HELM monomer code text directly."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "HuggingFace model id. If omitted, defaults to "
            "aaronfeller/peptideclm-2-mlm-base for backend=peptideclm and "
            "Flansma/helm-bert for backend=helmbert."
        ),
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Output .pt path. If omitted, defaults to "
            "data/residue_embeddings/peptideclm_residue_emb.pt or "
            "helmbert_residue_emb.pt depending on backend."
        ),
    )
    args = p.parse_args()

    # Backend-specific defaults so callers can just `--backend helmbert` and
    # everything else lines up with the trainer's checkpoint id.
    if args.model is None:
        args.model = (
            "aaronfeller/peptideclm-2-mlm-base"
            if args.backend == "peptideclm"
            else "Flansma/helm-bert"
        )
    if args.out is None:
        fname = (
            "peptideclm_residue_emb.pt"
            if args.backend == "peptideclm"
            else "helmbert_residue_emb.pt"
        )
        args.out = str(_HERE.parent / "data" / "residue_embeddings" / fname)

    # 1. Build the vocab in the same order ResidueVocab.from_csvs would.
    vocab = ResidueVocab.from_csvs(args.water_vocab, args.hexane_vocab)
    print(f"[vocab] size = {len(vocab)} (pad_id=0)")

    # 2. PDB code -> HELM monomer alignment from the dataset itself.
    pdb_to_helm = build_pdb_to_helm_mapping(Path(args.pdb_dir), Path(args.assay_csv))
    print(f"[align] PDB residues with at least one alignment: {len(pdb_to_helm)}")

    # 3. Resolve each vocab token to an LM input string. The dispatch differs:
    #    - peptideclm wants a SMILES (built via HELM monomer → CycPeptMPDB table)
    #    - helmbert wants the HELM monomer code text itself
    if args.backend == "peptideclm":
        monomer_smiles = load_monomer_smiles(Path(args.monomer_csv))
        print(f"[monomer table] {len(monomer_smiles)} HELM symbols loaded")
        overrides: Dict[str, str] = {
            "TNH": _CANONICAL_SMILES["T"],
            "ACE": _CANONICAL_SMILES["ac-"],
            "ARG": _CANONICAL_SMILES["R"],
            "MTR": "CN[C@@H]([C@@H](C)O)C(=O)O",
        }
        inputs_per_id, source_per_id = vocab_to_smiles(
            vocab, pdb_to_helm, monomer_smiles, overrides
        )
        input_kind = "smiles"
    else:  # helmbert
        # For HELM-BERT the overrides supply the canonical HELM token, not a
        # SMILES — `ac-` is the literal monomer string, etc.
        overrides = {
            "TNH": "T",
            "ACE": "ac-",
            "ARG": "R",
            "MTR": "meT",
        }
        inputs_per_id, source_per_id = vocab_to_helm_codes(
            vocab, pdb_to_helm, overrides
        )
        input_kind = "helm_code"

    missing = [
        vocab._tokens[i]
        for i, s in enumerate(inputs_per_id)
        if i != vocab.pad_id and not s
    ]
    if missing:
        print(f"[warn] No input resolved for {len(missing)} tokens: {missing}")
    else:
        print(f"[inputs] all {len(vocab) - 1} non-pad tokens resolved to an LM input")

    # 4. Run the LM.
    print(f"[lm] loading {args.model} on {args.device}")
    emb, has_emb = extract_embeddings(
        inputs_per_id, args.model, args.device, args.batch_size
    )

    # 5. Zero the pad row explicitly and save.
    emb[vocab.pad_id] = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": emb,                              # (V, lm_hidden)
        "tokens": list(vocab._tokens),                  # ordered residue names
        "has_embedding": torch.tensor(has_emb, dtype=torch.bool),
        # `helm_codes` is preserved as a stable key downstream consumers can
        # read regardless of which backend wrote the file. `inputs` holds the
        # raw string fed into the LM (SMILES for peptideclm, HELM text for
        # helmbert) for traceability.
        "helm_codes": source_per_id,
        "inputs": inputs_per_id,
        "input_kind": input_kind,
        "backend": args.backend,
        "model_name": args.model,
        "pad_id": vocab.pad_id,
    }
    torch.save(payload, out_path)
    print(
        f"[save] {out_path}  shape={tuple(emb.shape)}  "
        f"resolved={sum(has_emb)}/{len(vocab)}  "
        f"any_nan={bool(torch.isnan(emb).any())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

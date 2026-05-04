"""MultiCycPermea-style Tanimoto-similarity triplet loss.

The point: in batches, find triplets (anchor, positive, negative) where
positive is structurally close (high Tanimoto on Morgan fingerprints) but
the model should distinguish based on PAMPA, while negative is structurally
far. Encouraging the fused embedding to pull anchors closer to positives
that share a permeability profile and away from those that don't gives the
model a useful auxiliary signal — it directly attacks the "permeability
cliffs" failure mode the MultiCycPermea paper highlighted.

We compute Tanimoto on-the-fly over Morgan fingerprints derived from the
batch's SMILES strings. RDKit is the natural dependency here; if RDKit isn't
available, this loss returns 0 and the trainer simply skips the term.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def _morgan_fp_array(smiles_list: List[str], n_bits: int = 2048):
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return None

    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            fps.append(None)
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
        # Convert to a bool array via the bitvector iterator.
        arr = [int(b) for b in bv.ToBitString()]
        fps.append(arr)
    return fps


def _tanimoto_matrix(fps) -> Optional[torch.Tensor]:
    if fps is None:
        return None
    valid_idx = [i for i, f in enumerate(fps) if f is not None]
    if len(valid_idx) < 3:
        return None
    arr = torch.tensor(
        [fps[i] for i in valid_idx], dtype=torch.float32
    )  # (V, n_bits)
    inter = arr @ arr.T
    sums = arr.sum(dim=-1)
    union = sums.unsqueeze(0) + sums.unsqueeze(1) - inter
    sim = inter / union.clamp_min(1.0)
    # Re-embed into the full (B, B) shape with NaN for invalid rows/cols so
    # the caller can mask cleanly.
    B = len(fps)
    full = torch.full((B, B), float("nan"))
    for ii, vi in enumerate(valid_idx):
        for jj, vj in enumerate(valid_idx):
            full[vi, vj] = sim[ii, jj]
    return full


def build_triplets(
    smiles: List[str],
    pampa: torch.Tensor,
    sim_high: float = 0.7,
    sim_low: float = 0.4,
    pampa_gap: float = 1.0,
    max_triplets: int = 64,
) -> List[Tuple[int, int, int]]:
    """Pick (anchor, positive, negative) indices from a batch.

    positive: Tanimoto(anchor, positive) >= sim_high  AND
              |PAMPA(anchor) - PAMPA(positive)| <= pampa_gap  (structurally similar
              AND permeability-similar — a "non-cliff" pair)
    negative: Tanimoto(anchor, negative) <= sim_low   (structurally far)

    Returns up to max_triplets distinct triplets.
    """
    fps = _morgan_fp_array(smiles)
    sim = _tanimoto_matrix(fps)
    if sim is None:
        return []

    triplets: List[Tuple[int, int, int]] = []
    B = sim.shape[0]
    for a in range(B):
        # Find candidate positives and negatives.
        sim_row = sim[a]
        pampa_diff = (pampa - pampa[a]).abs()
        pos_mask = (sim_row >= sim_high) & (pampa_diff <= pampa_gap)
        pos_mask[a] = False
        neg_mask = sim_row <= sim_low
        neg_mask[a] = False

        pos_idx = pos_mask.nonzero(as_tuple=False).flatten().tolist()
        neg_idx = neg_mask.nonzero(as_tuple=False).flatten().tolist()
        if not pos_idx or not neg_idx:
            continue
        # Take the first valid (deterministic, cheap) — random sampling would
        # also work but introduces noise across runs.
        triplets.append((a, pos_idx[0], neg_idx[0]))
        if len(triplets) >= max_triplets:
            break
    return triplets


def tanimoto_triplet_loss(
    embeddings: torch.Tensor,
    smiles: List[str],
    pampa: torch.Tensor,
    margin: float = 0.5,
    **build_kwargs,
) -> torch.Tensor:
    """Standard triplet margin loss using Tanimoto-mined triplets."""
    triplets = build_triplets(smiles=smiles, pampa=pampa, **build_kwargs)
    if not triplets:
        return embeddings.new_zeros(())
    a_idx = torch.tensor([t[0] for t in triplets], device=embeddings.device)
    p_idx = torch.tensor([t[1] for t in triplets], device=embeddings.device)
    n_idx = torch.tensor([t[2] for t in triplets], device=embeddings.device)
    return F.triplet_margin_loss(
        embeddings[a_idx], embeddings[p_idx], embeddings[n_idx], margin=margin
    )

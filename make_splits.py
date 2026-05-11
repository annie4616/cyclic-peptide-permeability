"""
Create fixed data splits for CycPeptMPDB-4D (4925 peptides).

Settings (following MultiCycPermea, Wang et al. 2025):
  1. ID         : random 80/10/10
  2. OD         : chirality-aware ECFP8 (Morgan r=4, 4096 bits,
                  useChirality=True) + single-linkage clustering at Tanimoto
                  distance 0.275; clusters assigned wholesale to train/val/test.
                  Single linkage guarantees inter-set Tanimoto distance >=
                  0.275 at the molecule level (sim <= 0.725).
                  (ECFP4 with default useChirality=False fails: cyclic
                  peptides differ heavily in stereo, and at r=2 stereo-blind
                  FPs collide -- only 806 distinct FPs across 5140 distinct
                  isomeric SMILES. r=4/4096/chirality gives 5140/5160
                  distinct FPs and unlocks t=0.275 as the largest threshold
                  at which single-linkage still bin-packs into ~80/10/10;
                  above that chaining empties the val fold.)
  3. OD_Murcko  : coarser, scaffold-level OD split. Bemis-Murcko scaffolds
                  are extracted, fingerprinted with ECFP8 (r=4, 4096 bits,
                  chirality-blind: scaffolds have no side chains so stereo
                  contributes little), pairwise Tanimoto-clustered with
                  single linkage at distance 0.275 (same threshold as
                  molecule-level OD), and each scaffold cluster is assigned
                  wholesale to one split. Cyclic peptides often collapse
                  to a small number of macrocyclic skeletons, so this split
                  tests generalisation to *unseen ring systems*
                  (complementary to OD, which tests generalisation to unseen
                  side-chain patterns within similar scaffolds).
  4. Cliff_pair  : same cliff-pair definition as below, then assigned
                   so every pair stays inside one split (connected-component
                   bin-packing). Use when pair-level cliff evaluation matters.
  5. Cliff_ratio : same cliff pairs, but cliff and non-cliff molecules are
                   each split 80/10/10 independently. Pair membership is
                   NOT preserved across splits; use when you only need the
                   cliff fraction balanced across train/val/test.

     Cliff pair definition (shared by both): normalized SMILES Levenshtein
     similarity >= 0.9 AND |dPAMPA| > 2. Similarity is computed as
         1 - lev(s_i, s_j) / max(|s_i|, |s_j|)
     via cliffs.get_levenshtein_matrix (MoleculeACE-style). For cyclic
     peptides small string-level edits (single-residue swap, N-methyl
     toggle, D/L flip) are exactly the "almost-identical sequence, different
     permeability" cases we want to flag, and SMILES-string similarity
     captures that more directly than connectivity-only fingerprints.

All splits saved as pickled dict of index arrays referring to rows of
CycPeptMPDB-4D_with_SMILES.csv (preserving that file's row order).
"""
from __future__ import annotations
import os, pickle, random
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from cliffs import get_levenshtein_matrix

RDLogger.DisableLog("rdApp.*")

ROOT = "/hdd0/sohyun/cyclic-peptide-permeability"
CSV = os.path.join(ROOT, "data", "CycPeptMPDB-4D_with_assay_descriptors_preprocessed.csv")
OUT = os.path.join(ROOT, "splits")
os.makedirs(OUT, exist_ok=True)

SEED = 42
rng = np.random.RandomState(SEED)
random.seed(SEED)


def morgan_fp(smiles: str, radius: int = 4, nbits: int = 4096):
    """Chirality-aware Morgan fingerprint (ECFP8, 4096 bits).

    Cyclic peptides differ heavily in stereochemistry (D/L residues, N-methyl
    chirality), and these differences shift permeability. The default radius
    is bumped from 2 to 4 because at r=2 with useChirality=False only 806
    distinct fingerprints emerged from 5140 distinct isomeric SMILES --
    stereoisomers were colliding into the same bits. r=4 with useChirality=

    
    True yields 5140/5160 distinct FPs, and the larger 4096-bit width avoids
    the hash collisions that artificially compress similarity at r=3 nbits
    =2048. Empirically, this combination raises the maximum splittable
    single-linkage threshold from 0.175 (r=3, 2048) to 0.275.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits,
                                                 useChirality=True)


def split_random(n: int):
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * 0.10))
    n_val = int(round(n * 0.10))
    test = idx[:n_test]
    val = idx[n_test:n_test + n_val]
    train = idx[n_test + n_val:]
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def split_scaffold(smiles_list):
    """
    Chirality-aware ECFP8 (Morgan r=4, 4096 bits, useChirality=True)
    -> Tanimoto distance matrix -> single-linkage agglomerative clustering
    at threshold 0.275.

    Why these knobs:
    - We want a real OD guarantee at the molecule level: inter-set Tanimoto
      distance >= t for *every* pair across train/val/test. Only single-
      linkage delivers this -- complete-linkage's stopping rule constrains
      intra-cluster diameter, not the inter-cluster minimum.
    - r=2 (ECFP4) without chirality flags collapsed 5140 distinct isomeric
      SMILES into 806 distinct fingerprints because stereoisomers shared
      the same connectivity-only bits. r=3 with chirality (nbits=2048) is
      better (5138 distinct) but still suffers hash-collision compression
      that caps the splittable threshold at 0.175. Going to r=4 / nbits=
      4096 removes both effects: 5140 distinct FPs, no saturation, and
      the splittable threshold rises to 0.275.
    - At t=0.275: ~212 clusters, largest=1096, packing ~80/10/10 ≈ 4172
      train / 483 val / 505 test. Inter-set Tanimoto distance is provably
      >= 0.275 (similarity <= 0.725).
    """
    print("[OD] computing molecule fingerprints...")
    fps = [morgan_fp(s) for s in smiles_list]
    bad = [i for i, f in enumerate(fps) if f is None]
    if bad:
        raise RuntimeError(f"failed to parse {len(bad)} SMILES (e.g. idx {bad[:3]})")

    n = len(fps)
    print(f"[OD] building {n}x{n} distance matrix...")
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j, s in enumerate(sims, start=i + 1):
            d = 1.0 - s
            dist[i, j] = d
            dist[j, i] = d

    condensed = squareform(dist, checks=False)
    OD_THRESH = 0.275
    print(f"[OD] hierarchical clustering (single linkage, t={OD_THRESH})...")
    Z = linkage(condensed, method="single")
    clusters = fcluster(Z, t=OD_THRESH, criterion="distance")

    cl_ids = list(np.unique(clusters))
    cl_to_mols = {c: np.where(clusters == c)[0] for c in cl_ids}
    cl_to_size = {c: len(cl_to_mols[c]) for c in cl_ids}
    print(f"[OD] {len(cl_ids)} clusters; "
          f"largest={max(cl_to_size.values())}, "
          f"singletons={sum(1 for v in cl_to_size.values() if v == 1)}")

    total = sum(cl_to_size.values())
    target_test = int(round(0.10 * total))
    target_val = int(round(0.10 * total))

    # If the largest cluster alone exceeds 80% of the data, splitting it would
    # leak its dominant chemotype into val/test, defeating the OD goal. In
    # that case keep it entirely in train and fill val/test from the rest.
    # Otherwise just bin-pack all clusters greedily (smallest-first) into
    # test, then val, with overflow falling to train.
    biggest = max(cl_ids, key=lambda c: cl_to_size[c])
    keep_biggest_in_train = cl_to_size[biggest] > 0.80 * total

    if keep_biggest_in_train:
        print(f"[OD] largest cluster ({cl_to_size[biggest]}) > 80% of data, "
              "keeping it train-only")
        train_cl = [biggest]
        candidates = [c for c in cl_ids if c != biggest]
    else:
        train_cl = []
        candidates = list(cl_ids)

    candidates.sort(key=lambda c: cl_to_size[c])  # smallest first
    val_cl, test_cl = [], []
    cnt_t = cnt_v = 0
    for c in candidates:
        cc = cl_to_size[c]
        if cnt_t + cc <= target_test:
            test_cl.append(c); cnt_t += cc
        elif cnt_v + cc <= target_val:
            val_cl.append(c); cnt_v += cc
        else:
            train_cl.append(c)

    def mols_for(cluster_list):
        out = []
        for c in cluster_list:
            out.extend(cl_to_mols[c].tolist())
        return np.sort(np.array(out, dtype=int))

    return {
        "train": mols_for(train_cl),
        "val":   mols_for(val_cl),
        "test":  mols_for(test_cl),
    }


def murcko_scaffold_smiles(smiles: str) -> str | None:
    """Return canonical SMILES of the Bemis-Murcko scaffold of `smiles`.

    For cyclic peptides the scaffold is typically the full macrocycle (all
    side chains stripped), so most molecules collapse to a small number of
    near-identical macrocyclic skeletons. This is exactly why ECFP-based
    OD split (above) is the primary OD setting -- but the scaffold view is
    still informative as a coarser, ring-system-level OD baseline.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        # fall back to the molecule itself if scaffold is empty (no rings)
        return Chem.MolToSmiles(mol)
    return Chem.MolToSmiles(scaffold)


def scaffold_fp(scaffold_smi: str, radius: int = 4, nbits: int = 4096):
    """Morgan fingerprint over a scaffold SMILES (ECFP8, 4096 bits,
    chirality-blind).

    Same radius/width as the molecule-level OD fingerprint to keep the two
    splits directly comparable. Chirality is left off here because Murcko
    scaffolds have already had the stereo-rich side chains stripped, so
    `useChirality=True` would mostly add noise rather than signal.
    """
    mol = Chem.MolFromSmiles(scaffold_smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def split_scaffold_murcko(smiles_list, od_thresh: float = 0.275):
    """Murcko-scaffold-based OD split.

    Pipeline:
      1. Compute the Bemis-Murcko scaffold for every molecule.
      2. Group molecules by identical scaffold SMILES (a "scaffold class").
      3. Build a Tanimoto distance matrix over the *unique* scaffolds
         (ECFP8, r=4, 4096 bits) and single-linkage cluster them at
         threshold `od_thresh` -- so structurally close scaffolds (e.g.
         macrocycles that differ only in ring size or one heteroatom)
         end up in the same cluster instead of being treated as independent
         groups.
      4. Each scaffold cluster is assigned wholesale to one of train/val/
         test via greedy bin-packing (smallest cluster first into test,
         then val, rest to train).

    For cyclic peptides this is a *coarser* OD split than the full-molecule
    ECFP8 version above: many peptides share the same macrocyclic skeleton
    so scaffold clusters are larger and fewer. That is intentional -- this
    split tests generalisation to unseen ring systems / scaffold classes,
    while the full-molecule OD split tests generalisation to unseen
    side-chain patterns within similar scaffolds.

    Why t=0.275 (matches the molecule-level OD threshold): with r=4/4096
    the scaffold-distance median is ~0.69 and single-linkage stays well-
    behaved up to t~0.30. Empirical sweep on 356 unique scaffolds:
        t=0.275 -> 74 clusters, largest=1098 (22.3%)  [used here]
        t=0.300 -> 35 clusters, largest=1098 (22.3%)
        t=0.350 -> 13 clusters, largest=3330 (67.6%)  [val/test starve]
        t=0.450 ->  3 clusters, largest=3613 (73.4%)  [degenerate]
    Picking 0.275 keeps OD and OD_Murcko thresholds aligned so that the
    "scaffold-OD vs molecule-OD" comparison is interpretable: both enforce
    inter-set Tanimoto distance >= 0.275 in their respective spaces.
    (At r=2/2048 chaining started already at t~0.15; r=4/4096 unlocks the
    full 0.275 threshold for scaffolds too.)
    """
    print("[OD-Murcko] computing Murcko scaffolds...")
    scaffolds = [murcko_scaffold_smiles(s) for s in smiles_list]
    bad = [i for i, sc in enumerate(scaffolds) if sc is None]
    if bad:
        raise RuntimeError(f"failed to parse {len(bad)} SMILES (e.g. idx {bad[:3]})")

    # group molecules by scaffold SMILES
    scaf_to_mols: dict[str, list[int]] = {}
    for i, sc in enumerate(scaffolds):
        scaf_to_mols.setdefault(sc, []).append(i)
    unique_scaffolds = list(scaf_to_mols.keys())
    print(f"[OD-Murcko] {len(unique_scaffolds)} unique scaffolds across "
          f"{len(smiles_list)} molecules; "
          f"largest scaffold class={max(len(v) for v in scaf_to_mols.values())}")

    # Tanimoto distance matrix over unique scaffolds
    print("[OD-Murcko] fingerprinting scaffolds...")
    scaf_fps = [scaffold_fp(sc) for sc in unique_scaffolds]
    scaf_bad = [i for i, f in enumerate(scaf_fps) if f is None]
    if scaf_bad:
        raise RuntimeError(f"failed to fingerprint {len(scaf_bad)} scaffolds")

    m = len(scaf_fps)
    if m == 1:
        # only one scaffold -- cannot split by scaffold; fall back to
        # putting everything in train (caller should pick a different split)
        print("[OD-Murcko] WARNING: only 1 unique scaffold; train-only split")
        all_idx = np.arange(len(smiles_list))
        return {"train": all_idx,
                "val": np.array([], dtype=int),
                "test": np.array([], dtype=int)}

    print(f"[OD-Murcko] building {m}x{m} scaffold distance matrix...")
    dist = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        sims = DataStructs.BulkTanimotoSimilarity(scaf_fps[i], scaf_fps[i + 1:])
        for j, s in enumerate(sims, start=i + 1):
            d = 1.0 - s
            dist[i, j] = d
            dist[j, i] = d

    condensed = squareform(dist, checks=False)
    print(f"[OD-Murcko] hierarchical clustering (single linkage, t={od_thresh})...")
    Z = linkage(condensed, method="single")
    scaf_clusters = fcluster(Z, t=od_thresh, criterion="distance")

    # map: cluster id -> list of molecule indices
    cl_to_mols: dict[int, list[int]] = {}
    for scaf_idx, cl in enumerate(scaf_clusters):
        cl_to_mols.setdefault(int(cl), []).extend(scaf_to_mols[unique_scaffolds[scaf_idx]])

    cl_ids = list(cl_to_mols.keys())
    cl_to_size = {c: len(cl_to_mols[c]) for c in cl_ids}
    print(f"[OD-Murcko] {len(cl_ids)} scaffold clusters; "
          f"largest={max(cl_to_size.values())}, "
          f"singletons={sum(1 for v in cl_to_size.values() if v == 1)}")

    total = sum(cl_to_size.values())
    target_test = int(round(0.10 * total))
    target_val = int(round(0.10 * total))

    # If the largest scaffold cluster alone exceeds 80% of the data, keep it
    # train-only (same logic as the molecule-level OD split).
    biggest = max(cl_ids, key=lambda c: cl_to_size[c])
    keep_biggest_in_train = cl_to_size[biggest] > 0.80 * total

    if keep_biggest_in_train:
        print(f"[OD-Murcko] largest scaffold cluster ({cl_to_size[biggest]}) "
              "> 80% of data, keeping it train-only")
        train_cl = [biggest]
        candidates = [c for c in cl_ids if c != biggest]
    else:
        train_cl = []
        candidates = list(cl_ids)

    candidates.sort(key=lambda c: cl_to_size[c])  # smallest first
    val_cl, test_cl = [], []
    cnt_t = cnt_v = 0
    for c in candidates:
        cc = cl_to_size[c]
        if cnt_t + cc <= target_test:
            test_cl.append(c); cnt_t += cc
        elif cnt_v + cc <= target_val:
            val_cl.append(c); cnt_v += cc
        else:
            train_cl.append(c)

    def mols_for(cluster_list):
        out = []
        for c in cluster_list:
            out.extend(cl_to_mols[c])
        return np.sort(np.array(out, dtype=int))

    return {
        "train": mols_for(train_cl),
        "val":   mols_for(val_cl),
        "test":  mols_for(test_cl),
    }


def find_cliff_pairs(smiles_list, pampa, sim_thresh=0.9, dy_thresh=2.0):
    """Return list of (i,j) pairs with normalized SMILES Levenshtein
    similarity >= sim_thresh AND |dPAMPA| > dy_thresh.

    Similarity is delegated to cliffs.get_levenshtein_matrix:
        sim(s_i, s_j) = 1 - lev(s_i, s_j) / max(|s_i|, |s_j|)
    PAMPA is a log-permeability value (negative, ~[-9.5, -4]), so we keep
    the absolute-difference threshold rather than cliffs.ActivityCliffs's
    fold-change (which is meaningless for negative values).
    """
    print("[Cliff] computing pairwise SMILES Levenshtein similarity...")
    sim_mat = get_levenshtein_matrix(smiles_list)
    pampa = np.asarray(pampa, dtype=np.float32)
    n = len(smiles_list)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if sim_mat[i, j] >= sim_thresh and abs(pampa[i] - pampa[j]) > dy_thresh:
                pairs.append((i, j))
    print(f"[Cliff] total pairs: {len(pairs)}")
    return pairs


def split_cliff(n, cliff_pairs):
    """Build connected components over cliff_pairs, then assign each component
    entirely to one of train/val/test. This keeps every pair within one split,
    so pair-level evaluation is meaningful.

    Non-cliff molecules are distributed independently with the same 80/10/10
    ratio.
    """
    # union-find on molecules that appear in any pair
    parent = {}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cliff_mols = set()
    for i, j in cliff_pairs:
        cliff_mols.add(int(i)); cliff_mols.add(int(j))
        parent.setdefault(int(i), int(i))
        parent.setdefault(int(j), int(j))
    for i, j in cliff_pairs:
        union(int(i), int(j))

    comps = {}
    for m in cliff_mols:
        r = find(m)
        comps.setdefault(r, []).append(m)

    comp_list = list(comps.values())
    # shuffle components for random assignment, but place large components in
    # train first so val/test hit their targets
    rng_ = random.Random(SEED)
    rng_.shuffle(comp_list)
    comp_list.sort(key=len, reverse=True)

    total_cliff = sum(len(c) for c in comp_list)
    target_te = int(round(0.10 * total_cliff))
    target_va = int(round(0.10 * total_cliff))

    # greedy bin packing: fill test first, then val, rest to train
    tr_c, va_c, te_c = [], [], []
    cnt_t = cnt_v = 0
    for comp in reversed(comp_list):  # smallest first
        L = len(comp)
        if cnt_t + L <= target_te:
            te_c.extend(comp); cnt_t += L
        elif cnt_v + L <= target_va:
            va_c.extend(comp); cnt_v += L
        else:
            tr_c.extend(comp)

    cliff_set = set(cliff_mols)
    non_cliff = [i for i in range(n) if i not in cliff_set]
    rng_.shuffle(non_cliff)
    n_te = int(round(len(non_cliff) * 0.10))
    n_va = int(round(len(non_cliff) * 0.10))
    te_n = non_cliff[:n_te]; va_n = non_cliff[n_te:n_te + n_va]; tr_n = non_cliff[n_te + n_va:]

    return {
        "train": np.sort(np.array(tr_c + tr_n, dtype=int)),
        "val":   np.sort(np.array(va_c + va_n, dtype=int)),
        "test":  np.sort(np.array(te_c + te_n, dtype=int)),
        "cliff_mols":  np.array(sorted(cliff_mols), dtype=int),
        "cliff_pairs": np.array(cliff_pairs, dtype=int),
    }


def split_cliff_ratio(n, cliff_pairs):
    """Distribute cliff and non-cliff molecules independently with the same
    80/10/10 ratio. Pair membership is NOT preserved -- two molecules from
    the same cliff pair may end up in different splits. Use this when you
    want each split to contain the same fraction of cliff molecules but do
    not need within-split pair evaluation.
    """
    cliff_mols = set()
    for i, j in cliff_pairs:
        cliff_mols.add(int(i)); cliff_mols.add(int(j))

    rng_ = random.Random(SEED)

    def split_indices(idx_list):
        idx_list = list(idx_list)
        rng_.shuffle(idx_list)
        n_te = int(round(len(idx_list) * 0.10))
        n_va = int(round(len(idx_list) * 0.10))
        return (idx_list[n_te + n_va:],          # train
                idx_list[n_te:n_te + n_va],      # val
                idx_list[:n_te])                 # test

    cliff_list = sorted(cliff_mols)
    non_cliff = [i for i in range(n) if i not in cliff_mols]
    tr_c, va_c, te_c = split_indices(cliff_list)
    tr_n, va_n, te_n = split_indices(non_cliff)

    return {
        "train": np.sort(np.array(tr_c + tr_n, dtype=int)),
        "val":   np.sort(np.array(va_c + va_n, dtype=int)),
        "test":  np.sort(np.array(te_c + te_n, dtype=int)),
        "cliff_mols":  np.array(sorted(cliff_mols), dtype=int),
        "cliff_pairs": np.array(cliff_pairs, dtype=int),
    }


def main():
    df = pd.read_csv(CSV)
    n = len(df)
    smiles = df["SMILES"].tolist()
    pampa = df["PAMPA"].values
    print(f"dataset: {n} peptides")

    splits = {}

    print("\n=== ID split (random 80/10/10) ===")
    splits["ID"] = split_random(n)
    for k, v in splits["ID"].items():
        print(f"  {k}: {len(v)}")

    print("\n=== OD split (chirality-aware ECFP8 r=4/4096, single-linkage, Tanimoto dist >= 0.275 between sets) ===")
    splits["OD"] = split_scaffold(smiles)
    for k, v in splits["OD"].items():
        print(f"  {k}: {len(v)}")

    print("\n=== OD_Murcko split (Murcko scaffold ECFP8 r=4/4096, single-linkage on scaffolds, Tanimoto dist >= 0.275) ===")
    splits["OD_Murcko"] = split_scaffold_murcko(smiles)
    for k, v in splits["OD_Murcko"].items():
        print(f"  {k}: {len(v)}")

    print("\n=== Cliff pairs (SMILES Levenshtein sim>=0.9, |dPAMPA|>2) ===")
    cliff_pairs = find_cliff_pairs(smiles, pampa)

    print("\n=== Cliff_pair split (pairs preserved within a split) ===")
    splits["Cliff_pair"] = split_cliff(n, cliff_pairs)
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(splits['Cliff_pair'][k])}")
    print(f"  cliff molecules: {len(splits['Cliff_pair']['cliff_mols'])}")
    print(f"  cliff pairs:     {len(splits['Cliff_pair']['cliff_pairs'])}")

    print("\n=== Cliff_ratio split (cliff fraction balanced; pairs may cross splits) ===")
    splits["Cliff_ratio"] = split_cliff_ratio(n, cliff_pairs)
    cm = set(int(x) for x in splits["Cliff_ratio"]["cliff_mols"])
    for k in ("train", "val", "test"):
        idx = splits["Cliff_ratio"][k]
        n_cliff = sum(1 for i in idx if int(i) in cm)
        print(f"  {k}: {len(idx)}  (cliff mols: {n_cliff}, "
              f"frac: {n_cliff/len(idx):.3f})")

    out = os.path.join(OUT, "splits_v3.pkl")
    with open(out, "wb") as f:
        pickle.dump(splits, f)
    print(f"\nsaved -> {out}")

    # also save human-readable index CSVs
    for name, spl in splits.items():
        for k in ("train", "val", "test"):
            ids = df.iloc[spl[k]]["CycPeptMPDB_ID"].values
            pd.Series(ids, name="CycPeptMPDB_ID").to_csv(
                os.path.join(OUT, f"{name}_{k}_ids.csv"), index=False)


if __name__ == "__main__":
    main()

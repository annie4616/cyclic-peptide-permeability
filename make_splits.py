"""
Create fixed data splits for CycPeptMPDB-4D (4925 peptides).

Three settings (following MultiCycPermea, Wang et al. 2025):
  1. ID    : random 80/10/10
  2. OD    : chirality-aware ECFP8 (Morgan r=4, 4096 bits,
             useChirality=True) + single-linkage clustering at Tanimoto
             distance 0.275; clusters assigned wholesale to train/val/test.
             Single linkage guarantees inter-set Tanimoto distance >= 0.275
             at the molecule level (sim <= 0.725).
             (Murcko scaffolds are not used: cyclic peptides collapse to a
             single macrocyclic scaffold. ECFP4 with default useChirality
             =False also fails: cyclic peptides differ heavily in stereo,
             and at r=2 stereo-blind FPs collide -- only 806 distinct FPs
             across 5140 distinct isomeric SMILES. r=4/4096/chirality gives
             5140/5160 distinct FPs and unlocks t=0.275 as the largest
             threshold at which single-linkage still bin-packs into
             ~80/10/10; above that chaining empties the val fold.)
  3. Cliff : pairs with Tanimoto >= 0.9 AND |dPAMPA| > 2 distributed evenly.
             Uses standard ECFP4 (r=2, 2048 bits, useChirality=False) for
             the similarity test -- the conventional activity-cliff
             definition. Different from OD's chirality-aware ECFP8 because
             "cliff" classically means same connectivity / different
             activity, whereas OD wants stereo-resolved separation.

All splits saved as pickled dict of index arrays referring to rows of
CycPeptMPDB-4D_with_SMILES.csv (preserving that file's row order).
"""
from __future__ import annotations
import os, pickle, random
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

RDLogger.DisableLog("rdApp.*")

ROOT = "/ssd0/sohyun/cyclic_peptide/cyclic_peptide_permeability"
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


def find_cliff_pairs(smiles_list, pampa, sim_thresh=0.9, dy_thresh=2.0):
    """Return list of (i,j) pairs with Tanimoto >= sim_thresh & |dPAMPA| > dy_thresh.

    Uses standard ECFP4 (Morgan r=2, 2048 bits, useChirality=False) for the
    similarity test -- the conventional activity-cliff definition. This is
    intentionally different from the OD-split fingerprint (ECFP8/r=4/4096/
    chirality): OD wants stereo-resolved structural separation, whereas the
    cliff definition follows the cheminformatics convention of "same scaffold
    /connectivity, different activity," for which ECFP4 is standard.
    """
    print("[Cliff] computing Morgan fingerprints (ECFP4, useChirality=False)...")
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048,
                                                         useChirality=False)
                   if m is not None else None)
    pampa = np.asarray(pampa, dtype=np.float32)
    n = len(fps)
    pairs = []
    for i in range(n):
        if fps[i] is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        for j_off, s in enumerate(sims):
            j = i + 1 + j_off
            if s >= sim_thresh and abs(pampa[i] - pampa[j]) > dy_thresh:
                pairs.append((i, j))
        if i % 500 == 0:
            print(f"  scanned {i}/{n}, pairs so far: {len(pairs)}")
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

    print("\n=== Cliff split (Tanimoto>=0.9, |dPAMPA|>2) ===")
    cliff_pairs = find_cliff_pairs(smiles, pampa)
    splits["Cliff"] = split_cliff(n, cliff_pairs)
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(splits['Cliff'][k])}")
    print(f"  cliff molecules: {len(splits['Cliff']['cliff_mols'])}")
    print(f"  cliff pairs:     {len(splits['Cliff']['cliff_pairs'])}")

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

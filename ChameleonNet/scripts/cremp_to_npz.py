"""Convert CREMP-CycPeptMPDB pickles -> conformer .npz for encoder pretraining.

CREMP rd_mol carries 3D coordinates for thousands of conformers but NO per-atom
residue labels, and its monomer vocabulary does not correspond to our 54-token
ResidueVocab. So residue-id alignment is infeasible; we pretrain the
geometry+atomic-number pathway (atom_embed + EGNN) and set res=0 (pad). z
(atomic number, H included) and coordinates ARE compatible with our pipeline.

Output: <out_dir>/<idx>.npz with coords (K,N,3) float32, z (N,) int64,
res (N,) int64 zeros. K conformers uniformly sampled per molecule.
"""
from __future__ import annotations
import pickle, glob, os, sys
import numpy as np

SRC = "/hdd0/sohyun/cyclic-peptide-permeability/data/CREMP_CycPeptMPDB_pickle/pickle"
OUT = "/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet/.cache_cremp"
FRAC = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10   # 10% subset
K = int(sys.argv[2]) if len(sys.argv) > 2 else 30          # conformers per molecule
SEED = 0


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{SRC}/*.pickle"))
    rng = np.random.RandomState(SEED)
    n_take = max(1, int(len(files) * FRAC))
    idx = rng.choice(len(files), size=n_take, replace=False)
    files = [files[i] for i in sorted(idx)]
    print(f"[cremp] {len(files)} molecules (frac={FRAC}), K={K} conformers each", flush=True)

    written = 0
    for i, f in enumerate(files):
        try:
            d = pickle.load(open(f, "rb"))
            m = d["rd_mol"]
            N = m.GetNumAtoms()
            z = np.array([a.GetAtomicNum() for a in m.GetAtoms()], dtype=np.int64)
            nconf = m.GetNumConformers()
            sel = rng.choice(nconf, size=min(K, nconf), replace=False)
            coords = np.stack([m.GetConformer(int(c)).GetPositions() for c in sel], axis=0).astype(np.float32)
            res = np.zeros(N, dtype=np.int64)  # no residue labels available
            # Per-conformer relative energy (kcal/mol) — aligned to rd_mol conformer
            # order (conformers[i] <-> conformer i). Used as the SSL target (the
            # invariant encoder learns a force-field-like potential).
            confs = d.get("conformers", [])
            relE = np.array([float(confs[int(c)].get("relativeenergy", np.nan))
                             if int(c) < len(confs) else np.nan for c in sel], dtype=np.float32)
            np.savez_compressed(f"{OUT}/{written}.npz", coords=coords, z=z, res=res, relE=relE)
            written += 1
        except Exception as e:
            print(f"  skip {os.path.basename(f)}: {e}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)} processed, {written} written", flush=True)
    print(f"[cremp] done: {written} npz in {OUT}", flush=True)


if __name__ == "__main__":
    main()

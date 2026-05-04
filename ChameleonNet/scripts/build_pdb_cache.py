"""One-shot script: parse every PDB referenced by the assay CSV into the cache.

Run this before the first training launch. After this completes, training
will skip the PDB parsing entirely and read .npz tensors instead, which cuts
epoch time from minutes to seconds on the data side.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from chameleonnet.data.dataset import _build_pdb_paths  # type: ignore
from chameleonnet.data.dataset import _PeptideRecord, ChameleonDataset
from chameleonnet.data.residue_vocab import ResidueVocab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="/ssd0/sohyun/cyclic_peptide_permeability/data/CycPeptMPDB-4D_with_assay_descriptors.csv")
    parser.add_argument("--pdb_root", default="/ssd0/sohyun/cyclic_peptide_permeability/CycPeptMPDB-4D")
    parser.add_argument("--cache_dir", default="/ssd0/sohyun/cyclic_peptide_permeability/ChameleonNet/.cache_pdb")
    parser.add_argument("--use_trajectory", default="true")
    args = parser.parse_args()

    use_traj = args.use_trajectory.lower() in {"1", "true", "yes"}

    # Read the full id list from the CSV.
    ids: list[int] = []
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ids.append(int(row["CycPeptMPDB_ID"]))
            except (KeyError, ValueError):
                continue

    vocab = ResidueVocab.from_csvs(
        "/ssd0/sohyun/cyclic_peptide_permeability/water_residue_vocab.csv",
        "/ssd0/sohyun/cyclic_peptide_permeability/hexane_residue_vocab.csv",
    )

    ds = ChameleonDataset(
        ids=ids, csv_path=args.csv, pdb_root=args.pdb_root, vocab=vocab,
        use_trajectory=use_traj, max_conformers=10**6, cache_dir=args.cache_dir,
    )
    ok = 0
    fail = 0
    for i in range(len(ds)):
        try:
            ds[i]
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 10:
                print(f"FAIL pid={ds.ids[i]}: {e}")
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(ds)}] ok={ok} fail={fail}")
    print(f"done. ok={ok} fail={fail}")


if __name__ == "__main__":
    main()

"""Regenerate ONLY the OD_Murcko split at Tanimoto distance 0.300.

Leaves ID / OD / Cliff_* splits untouched. Writes:
  - splits/OD_Murcko_{train,val,test}_ids.csv   (new 0.300 split)
  - adds/updates the "OD_Murcko" key in splits/splits_v6.pkl

The previous 0.275 split's id CSVs are already archived as
splits/OD_Murcko_0.275_{train,val,test}_ids.csv. This script also dumps the
0.275 *index arrays* (recomputed at t=0.275) to splits/OD_Murcko_0.275.pkl so
the archived split is recoverable as integer row indices, not just IDs.
"""
import os, pickle
import pandas as pd

from make_splits import CSV, OUT, split_scaffold_murcko

PKL = os.path.join(OUT, "splits_v6.pkl")


def write_ids(df, spl, name):
    for k in ("train", "val", "test"):
        ids = df.iloc[spl[k]]["CycPeptMPDB_ID"].values
        pd.Series(ids, name="CycPeptMPDB_ID").to_csv(
            os.path.join(OUT, f"{name}_{k}_ids.csv"), index=False)


def main():
    df = pd.read_csv(CSV, low_memory=False)
    smiles = df["SMILES"].tolist()
    print(f"dataset: {len(df)} peptides\n")

    # --- archive: recompute 0.275 index arrays for the pickled record ---
    print("=== OD_Murcko @ t=0.275 (archive, index arrays only) ===")
    od_275 = split_scaffold_murcko(smiles, od_thresh=0.275)
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(od_275[k])}")
    with open(os.path.join(OUT, "OD_Murcko_0.275.pkl"), "wb") as f:
        pickle.dump(od_275, f)
    print("  -> saved splits/OD_Murcko_0.275.pkl")

    # --- new split at 0.300 ---
    print("\n=== OD_Murcko @ t=0.300 (new) ===")
    od_300 = split_scaffold_murcko(smiles, od_thresh=0.300)
    for k in ("train", "val", "test"):
        print(f"  {k}: {len(od_300[k])}")

    write_ids(df, od_300, "OD_Murcko")
    print("  -> wrote splits/OD_Murcko_{train,val,test}_ids.csv")

    # --- update pickle, leaving every other split untouched ---
    with open(PKL, "rb") as f:
        splits = pickle.load(f)
    splits["OD_Murcko"] = od_300
    splits["OD_Murcko_0.275"] = od_275
    with open(PKL, "wb") as f:
        pickle.dump(splits, f)
    print(f"\nupdated {PKL}: keys = {list(splits.keys())}")


if __name__ == "__main__":
    main()

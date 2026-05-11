"""Build MultiCycPermea-ready CSVs for the OD_Murcko split.

Reads CycPeptMPDB_ID lists from splits/OD_Murcko_{train,val,test}_ids.csv,
joins with the 4D merged CSV (SMILES + targets) and the author train/val/test
to fill the full descriptor schema. Missing columns are NaN.

Output: MultiCycPermea/DL/data/ours/OD_Murcko_{train,val,test}.csv
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path("/hdd0/sohyun/cyclic-peptide-permeability")
SPLITS_DIR = ROOT / "splits"
MERGED = ROOT / "data" / "CycPeptMPDB-4D_with_SMILES.csv"
AUTHOR_DIR = ROOT / "MultiCycPermea" / "data" / "remove_strange_values"
OUT_DIR = ROOT / "MultiCycPermea" / "DL" / "data" / "ours"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT = "OD_Murcko"


def main():
    df = pd.read_csv(MERGED)
    author_cols = list(pd.read_csv(AUTHOR_DIR / "train.csv", nrows=1).columns)

    author_all = []
    for k in ("train", "val", "test"):
        author_all.append(pd.read_csv(AUTHOR_DIR / f"{k}.csv"))
    author_full = pd.concat(author_all, ignore_index=True).drop_duplicates(
        subset="CycPeptMPDB_ID", keep="first")
    print(f"author lookup rows: {len(author_full)} with {len(author_full.columns)} cols")

    base_extra = author_full.drop(columns=["SMILES", "PAMPA", "Permeability"], errors="ignore")
    merged = df.merge(base_extra, on="CycPeptMPDB_ID", how="left", suffixes=("", "_author"))
    for c in author_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    out_cols = [c for c in author_cols if c in merged.columns]
    # Our 4D CSV has only PAMPA (which is the permeability target). Use it
    # for both Permeability and PAMPA so author-schema downstream code that
    # may read either column still works.
    merged["PAMPA"] = df["PAMPA"].values
    merged["Permeability"] = df["PAMPA"].values
    merged = merged[out_cols]

    by_id = merged.set_index("CycPeptMPDB_ID")

    for k in ("train", "val", "test"):
        ids_path = SPLITS_DIR / f"{SPLIT}_{k}_ids.csv"
        ids = pd.read_csv(ids_path)["CycPeptMPDB_ID"].tolist()
        keep = [i for i in ids if i in by_id.index]
        miss = [i for i in ids if i not in by_id.index]
        sub = by_id.loc[keep].reset_index()
        out = OUT_DIR / f"{SPLIT}_{k}.csv"
        sub.to_csv(out, index=False)
        print(f"wrote {out}  rows={len(sub)} (requested={len(ids)}, missing={len(miss)})")

    print("\ndone.")


if __name__ == "__main__":
    main()

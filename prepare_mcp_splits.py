"""
Build MultiCycPermea-ready CSVs for our 4D splits (ID / OD / Cliff).

MultiCycPermea's dataset loader (DL/dataset.py) reads CSV with at least:
  - CycPeptMPDB_ID
  - SMILES
  - target column (Permeability or PAMPA)

We also carry all columns from the original author train.csv schema when
available, because some fingerprint/descriptor columns can be used by the
tabular branch if enabled. Missing columns for our 5160 peptides are filled
with NaN; training uses only SMILES + image by default.

Output:
  MultiCycPermea/DL/data/ours/{ID,OD,Cliff}_{train,val,test}.csv
"""
from __future__ import annotations
import os, pickle
from pathlib import Path
import pandas as pd

ROOT = Path("/ssd0/sohyun/cyclic_peptide_permeability")
SPLITS_PKL = ROOT / "splits" / "splits_v1.pkl"
MERGED = ROOT / "CycPeptMPDB-4D_with_SMILES.csv"
AUTHOR_CSV = ROOT / "MultiCycPermea" / "data" / "remove_strange_values" / "train.csv"
OUT_DIR = ROOT / "MultiCycPermea" / "DL" / "data" / "ours"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    splits = pickle.load(open(SPLITS_PKL, "rb"))
    df = pd.read_csv(MERGED)
    # author schema columns (superset) to keep downstream tabular features usable
    author_cols = list(pd.read_csv(AUTHOR_CSV, nrows=1).columns)

    # Pull as many of those columns as possible by joining author train/val/test
    # into one big lookup (it has SMILES + full descriptor set for overlapping
    # IDs).
    author_all = []
    for split in ["train", "val", "test"]:
        p = ROOT / "MultiCycPermea" / "data" / "remove_strange_values" / f"{split}.csv"
        author_all.append(pd.read_csv(p))
    author_full = pd.concat(author_all, ignore_index=True).drop_duplicates(
        subset="CycPeptMPDB_ID", keep="first")
    print(f"author lookup rows: {len(author_full)} with {len(author_full.columns)} cols")

    # overlap check
    overlap = df.merge(author_full[["CycPeptMPDB_ID"]], on="CycPeptMPDB_ID", how="inner")
    print(f"our 5160 <-> author lookup overlap: {len(overlap)}")

    # Build per-row output using author schema; add/override target columns
    # PAMPA and Permeability from our merged CSV (so PAMPA reflects 4D value).
    base_extra = author_full.drop(columns=["SMILES", "PAMPA", "Permeability"], errors="ignore")
    merged = df.merge(base_extra, on="CycPeptMPDB_ID", how="left",
                      suffixes=("", "_author"))
    # If any author_cols missing, add as NaN
    for c in author_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    # Reorder to author schema
    out_cols = [c for c in author_cols if c in merged.columns]
    # make sure PAMPA & Permeability are from our df
    merged["PAMPA"] = df["PAMPA"].values
    merged["Permeability"] = df["Permeability"].values
    merged = merged[out_cols]

    # Emit split CSVs. splits indices refer to df (CycPeptMPDB-4D_with_SMILES.csv) rows.
    for name, spl in splits.items():
        if name not in ("ID", "OD", "Cliff"):
            continue
        for k in ("train", "val", "test"):
            idx = spl[k]
            sub = merged.iloc[idx].copy()
            out = OUT_DIR / f"{name}_{k}.csv"
            sub.to_csv(out, index=False)
            print(f"wrote {out}  rows={len(sub)}")

    print("\ndone.")


if __name__ == "__main__":
    main()

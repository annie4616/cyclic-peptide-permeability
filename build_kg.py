"""
Build MultiCycPermea's KG.pkl: a DataFrame of pairwise Tanimoto similarity
between all cyclic peptides we'll ever feed to the model.

Covers the union of IDs from author splits and our 4D subset.
Indexed by CycPeptMPDB_ID for both rows and columns.

The model reads weights as:
    weight = 1 - similarity_matrix_df.loc[ids, ids]
so the matrix must be a symmetric float DataFrame with matching index/columns.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/ssd0/sohyun/cyclic_peptide_permeability")
MCP = ROOT / "MultiCycPermea"

def load_all():
    frames = []
    for split in ["train", "val", "test"]:
        p = MCP / "data" / "remove_strange_values" / f"{split}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, usecols=["CycPeptMPDB_ID", "SMILES"]))
    frames.append(pd.read_csv(ROOT / "CycPeptMPDB-4D_with_SMILES.csv",
                              usecols=["CycPeptMPDB_ID", "SMILES"]))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset="CycPeptMPDB_ID", keep="first").reset_index(drop=True)
    return df


def main():
    df = load_all()
    n = len(df)
    print(f"building KG for {n} peptides")

    fps = []
    for smi in tqdm(df["SMILES"], desc="fp"):
        m = Chem.MolFromSmiles(smi)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))

    sim = np.zeros((n, n), dtype=np.float32)
    for i in tqdm(range(n), desc="tanimoto"):
        sim[i, i] = 1.0
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
        sim[i, i + 1:] = sims
        sim[i + 1:, i] = sims

    ids = df["CycPeptMPDB_ID"].tolist()
    kg = pd.DataFrame(sim, index=ids, columns=ids)

    # MCP reads KG.pkl from the CWD where `main.py` is run, i.e. DL/
    out = MCP / "DL" / "KG.pkl"
    kg.to_pickle(out)
    print(f"saved {out}  shape={kg.shape}  size={out.stat().st_size/1e6:.1f}MB")


if __name__ == "__main__":
    main()

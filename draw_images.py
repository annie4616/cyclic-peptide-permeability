"""
Generate 2D molecular images for cyclic peptides used by MultiCycPermea.

Covers the union of IDs from:
  - MultiCycPermea author splits (data/remove_strange_values/{train,val,test}.csv)
  - CycPeptMPDB-4D subset (CycPeptMPDB-4D_with_SMILES.csv)

Each peptide gets one PNG named {CycPeptMPDB_ID}.png in:
  MultiCycPermea/DL/data/cycle_peptide/cycle_peptide_image_png/
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdCoordGen, rdDepictor
import cairosvg
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/ssd0/sohyun/cyclic_peptide_permeability")
MCP = ROOT / "MultiCycPermea"
OUT = MCP / "DL" / "data" / "cycle_peptide" / "cycle_peptide_image_png"
OUT.mkdir(parents=True, exist_ok=True)

rdDepictor.SetPreferCoordGen(True)


def load_all_ids_smiles() -> pd.DataFrame:
    # author splits
    frames = []
    for split in ["train", "val", "test"]:
        p = MCP / "data" / "remove_strange_values" / f"{split}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, usecols=["CycPeptMPDB_ID", "SMILES"]))
    # 4D subset (merged)
    p4 = ROOT / "CycPeptMPDB-4D_with_SMILES.csv"
    if p4.exists():
        frames.append(pd.read_csv(p4, usecols=["CycPeptMPDB_ID", "SMILES"]))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="CycPeptMPDB_ID")
    return df


def draw_one(row) -> tuple[int, bool, str]:
    pid, smi = int(row["CycPeptMPDB_ID"]), row["SMILES"]
    out_path = OUT / f"{pid}.png"
    if out_path.exists():
        return pid, True, "skip"
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return pid, False, "invalid_smiles"
        rdCoordGen.AddCoords(mol)
        view = Draw.rdMolDraw2D.MolDraw2DSVG(600, 600)
        view.DrawMolecule(Draw.rdMolDraw2D.PrepareMolForDrawing(mol))
        view.FinishDrawing()
        svg = view.GetDrawingText()
        cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(out_path))
        return pid, True, "ok"
    except Exception as e:
        return pid, False, f"error:{type(e).__name__}"


def main():
    df = load_all_ids_smiles()
    print(f"Total unique peptides to render: {len(df)}")
    print(f"Output: {OUT}")
    rows = df.to_dict("records")

    ok = 0; fail = 0; fails = []
    # ThreadPool: RDKit + cairosvg release the GIL enough for threading
    with ThreadPoolExecutor(max_workers=8) as ex:
        for pid, good, msg in tqdm(ex.map(draw_one, rows), total=len(rows)):
            if good:
                ok += 1
            else:
                fail += 1
                fails.append((pid, msg))
                if fail <= 10:
                    print(f"  fail {pid}: {msg}", file=sys.stderr)
    print(f"\ndone. ok={ok} fail={fail}")
    if fails:
        with open(ROOT / "image_draw_failures.txt", "w") as f:
            for pid, msg in fails:
                f.write(f"{pid}\t{msg}\n")
        print(f"failures written to {ROOT/'image_draw_failures.txt'}")


if __name__ == "__main__":
    main()

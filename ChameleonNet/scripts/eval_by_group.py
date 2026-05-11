"""Evaluate a trained checkpoint and split test metrics by Molecule_Shape.

Loads best.pt from `cfg.output_dir`, runs each test split in `cfg.eval_schemes`,
groups peptides by (Molecule_Shape, Monomer_Length), and writes per-group
regression metrics to `<output_dir>/test_metrics_by_group.json`.

Usage:
    python -m scripts.eval_by_group --config configs/v2_id.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from chameleonnet.training import Trainer, load_config
from chameleonnet.training.trainer import _move_batch
from chameleonnet.data.dataset import ChameleonDataset, chameleon_collate
from chameleonnet.data.splits import load_split
from chameleonnet.utils.metrics import regression_metrics


def load_groups(csv_path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                pid = int(row["CycPeptMPDB_ID"])
            except (KeyError, ValueError):
                continue
            shape = (row.get("Molecule_Shape") or "").strip() or "?"
            length = (row.get("Monomer_Length") or "").strip() or "?"
            out[pid] = f"{shape}_{length}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.wandb_mode = "disabled"
    trainer = Trainer(cfg)

    ckpt_path = Path(cfg.output_dir) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(ckpt["model"])
    trainer.model.eval()

    groups = load_groups(cfg.csv_path)
    results_all: Dict[str, Dict] = {}

    for scheme in cfg.eval_schemes:
        try:
            ids = load_split(cfg.splits_dir, scheme, "test")
        except (FileNotFoundError, ValueError):
            continue

        ds = ChameleonDataset(
            ids=ids,
            csv_path=cfg.csv_path,
            pdb_root=cfg.pdb_root,
            vocab=trainer.vocab,
            descriptor_cols=trainer.descriptor_cols,
            use_trajectory=cfg.use_trajectory,
            max_conformers=cfg.max_conformers,
            cache_dir=cfg.cache_dir,
            augment_delta_descriptors=(
                cfg.augment_delta_descriptors or cfg.model_arch == "v2"
            ),
            conformer_source=getattr(cfg, "conformer_source", "trajectory"),
        )
        scaler = trainer.scaler
        orig = ds.__getitem__

        def __getitem__(idx: int, _orig=orig, _scaler=scaler):
            s = _orig(idx)
            s["descriptors"] = _scaler.transform(s["descriptors"])
            return s

        ds.__getitem__ = __getitem__  # type: ignore[method-assign]

        loader = DataLoader(
            ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, collate_fn=chameleon_collate,
        )

        all_pids: List[int] = []
        all_preds: List[float] = []
        all_targs: List[float] = []
        with torch.no_grad():
            for batch in loader:
                batch_d = _move_batch(batch, trainer.device)
                outputs = trainer.model(batch_d)
                all_pids.extend(batch_d["pids"])
                all_preds.extend(outputs["pampa_pred"].detach().cpu().numpy().tolist())
                all_targs.extend(batch_d["pampa"].detach().cpu().numpy().tolist())

        preds = np.array(all_preds, dtype=np.float64)
        targs = np.array(all_targs, dtype=np.float64)
        g_arr = np.array([groups.get(p, "Unknown") for p in all_pids])

        result: Dict[str, Dict] = {"all": {**regression_metrics(targs, preds), "n": int(len(preds))}}
        for g in sorted(set(g_arr.tolist())):
            mask = g_arr == g
            if mask.sum() == 0:
                continue
            result[g] = {**regression_metrics(targs[mask], preds[mask]), "n": int(mask.sum())}
        results_all[scheme] = result
        print(f"[{scheme}]")
        print(json.dumps(result, indent=2))

    out_path = Path(cfg.output_dir) / "test_metrics_by_group.json"
    out_path.write_text(json.dumps(results_all, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

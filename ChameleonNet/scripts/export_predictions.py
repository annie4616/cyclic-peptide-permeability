"""Dump per-molecule (CycPeptMPDB_ID, pred, target) for a trained run.

Loads best.pt from cfg.output_dir, runs inference on each scheme in
cfg.eval_schemes, and writes predictions.csv to the run directory.

Usage:
    python -m scripts.export_predictions --config configs/v1_od_murcko_traj_notri.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from chameleonnet.training import Trainer, load_config
from chameleonnet.training.trainer import _move_batch
from chameleonnet.data.dataset import ChameleonDataset, chameleon_collate
from chameleonnet.data.splits import load_split


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

        rows = []
        with torch.no_grad():
            for batch in loader:
                batch_d = _move_batch(batch, trainer.device)
                outputs = trainer.model(batch_d)
                preds = outputs["pampa_pred"].detach().cpu().numpy().tolist()
                targs = batch_d["pampa"].detach().cpu().numpy().tolist()
                for pid, p, t in zip(batch_d["pids"], preds, targs):
                    rows.append((int(pid), float(p), float(t)))

        out_path = Path(cfg.output_dir) / f"predictions_{scheme}.csv"
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["CycPeptMPDB_ID", "pred", "target"])
            w.writerows(rows)
        print(f"wrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()

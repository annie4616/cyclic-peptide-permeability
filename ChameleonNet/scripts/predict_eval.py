"""Load trained ChameleonNet run(s), dump per-pid predictions, and optionally
blend with GBR / ensemble across runs. Used for Round-2 stacking and finals.

Usage:
  # dump preds for a run on OD_Murcko val+test
  python scripts/predict_eval.py dump <run_dir>
  # blend one or more runs with GBR (alpha tuned on val) and report OD test
  python scripts/predict_eval.py blend <run_dir>[,<run_dir>...]
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from chameleonnet.training.config import TrainConfig
from chameleonnet.training.trainer import _load_gbr_preds, _move_batch
from chameleonnet.data.dataset import (ChameleonDataset, DEFAULT_DESCRIPTORS,
    EXTENDED_DESCRIPTORS, full_descriptor_cols, chameleon_collate)
from chameleonnet.data.residue_vocab import ResidueVocab
from chameleonnet.data.splits import load_split
from chameleonnet.models.chameleonnet import ChameleonNet
from chameleonnet.utils.scaler import DescriptorScaler
from chameleonnet.utils.metrics import regression_metrics

ROOT = "/hdd0/sohyun/cyclic-peptide-permeability"
GBR = json.load(open(f"{ROOT}/ChameleonNet/runs/_logs/gbr_preds_OD_Murcko.json"))


def _descriptor_cols(cfg):
    if cfg.descriptor_cols: return cfg.descriptor_cols
    if getattr(cfg, "use_full_descriptors", False): return full_descriptor_cols(cfg.csv_path)
    if getattr(cfg, "use_extended_descriptors", False): return EXTENDED_DESCRIPTORS
    return DEFAULT_DESCRIPTORS


def load_run(run_dir):
    ckpt = torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=False)
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items() if k in TrainConfig.__dataclass_fields__})
    return ckpt, cfg


@torch.no_grad()
def predict(run_dir, fold="test", device="cuda"):
    ckpt, cfg = load_run(run_dir)
    vocab = ResidueVocab.from_csvs(*cfg.vocab_csvs)
    scaler = DescriptorScaler(); scaler.mean = ckpt["scaler_mean"]; scaler.std = ckpt["scaler_std"]
    augment = cfg.augment_delta_descriptors or cfg.model_arch == "v2"
    ds = ChameleonDataset(
        ids=load_split(cfg.splits_dir, cfg.split_scheme, fold), csv_path=cfg.csv_path,
        pdb_root=cfg.pdb_root, vocab=vocab, descriptor_cols=_descriptor_cols(cfg),
        use_trajectory=cfg.use_trajectory, max_conformers=cfg.max_conformers,
        cache_dir=cfg.cache_dir, augment_delta_descriptors=augment,
        conformer_source=getattr(cfg, "conformer_source", "trajectory"),
        gbr_preds=_load_gbr_preds(cfg))
    ds.scaler = scaler
    descriptor_dim = ds.descriptor_dim
    adv_kw = dict(n_scaffold_groups=(cfg.adv_n_groups if (cfg.lambda_adv>0 and cfg.adv_n_groups>0) else 0),
                  adv_hidden=cfg.adv_hidden, modality_dropout=cfg.modality_dropout,
                  physics_residual=cfg.physics_residual, info_bottleneck=cfg.info_bottleneck,
                  gbr_residual=cfg.gbr_residual, distill=cfg.lambda_distill>0)
    model = ChameleonNet(vocab=vocab, descriptor_dim=descriptor_dim, hidden_dim=cfg.hidden_dim,
        conformer_layers=cfg.conformer_layers, sequence_backend=cfg.sequence_backend,
        peptideclm_name_or_path=cfg.peptideclm_name_or_path, helmbert_name_or_path=cfg.helmbert_name_or_path,
        head_hidden=cfg.head_hidden, dropout=cfg.dropout, residue_emb_path=cfg.residue_emb_path, **adv_kw)
    model.load_state_dict(ckpt["model"]); model.to(device).eval()
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=4, collate_fn=chameleon_collate)
    pids, preds, ys = [], [], []
    for batch in loader:
        b = _move_batch(batch, torch.device(device))
        p = model(b)["pampa_pred"].cpu().numpy()
        pids += [int(x) for x in batch["pids"]]; preds += list(p); ys += [float(x) for x in batch["pampa"]]
    return np.array(pids), np.array(preds), np.array(ys)


def main():
    mode = sys.argv[1]
    runs = sys.argv[2].split(",")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # ensemble deep preds (mean over runs) on val & test, aligned by pid
    def ens(fold):
        acc = {}; ys = {}
        for r in runs:
            pid, pr, y = predict(r, fold, dev)
            for i, p in enumerate(pid):
                acc.setdefault(p, []).append(pr[i]); ys[p] = y[i]
        pids = sorted(acc)
        return np.array(pids), np.array([np.mean(acc[p]) for p in pids]), np.array([ys[p] for p in pids])
    vp, vd, vy = ens("val"); tp, td, ty = ens("test")
    vg = np.array([GBR[str(p)] for p in vp]); tg = np.array([GBR[str(p)] for p in tp])
    deep = regression_metrics(ty, td)
    print("[deep ensemble OD test]", {k: round(v,4) for k,v in deep.items()})
    if mode == "blend":
        alphas = np.linspace(0, 1, 21)
        best_a = min(alphas, key=lambda a: regression_metrics(vy, a*vd+(1-a)*vg)["mae"])
        bl = regression_metrics(ty, best_a*td+(1-best_a)*tg)
        print(f"[blend alpha={best_a:.2f} (deep weight)] OD test", {k: round(v,4) for k,v in bl.items()})
    print("[gbr OD test]", {k: round(v,4) for k,v in regression_metrics(ty, tg).items()})


if __name__ == "__main__":
    main()

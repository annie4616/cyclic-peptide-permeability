"""Training loop for ChameleonNet.

Handles:
  - dataset/dataloader construction with the cached descriptor scaler
  - cosine LR with warmup
  - composite loss with logged sub-terms
  - best-checkpoint selection on validation MAE
  - multi-scheme evaluation (ID/OD/Cliff) at the end of training
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import (
    ChameleonDataset,
    DEFAULT_DESCRIPTORS,
    chameleon_collate,
)
from ..data.residue_vocab import ResidueVocab
from ..data.splits import load_split
from ..losses.composite import CompositeLoss, composite_loss
from ..models.chameleonnet import ChameleonNet
from ..utils.metrics import regression_metrics
from ..utils.scaler import DescriptorScaler
from ..utils.seed import set_seed
from .config import TrainConfig


def _device_of(cfg: TrainConfig) -> torch.device:
    if cfg.device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(cfg.device)


def _move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {kk: vv.to(device, non_blocking=True) if isinstance(vv, torch.Tensor) else vv
                      for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _build_loaders(cfg: TrainConfig, vocab: ResidueVocab):
    descriptor_cols = cfg.descriptor_cols or DEFAULT_DESCRIPTORS

    def make_dataset(scheme: str, fold: str) -> ChameleonDataset:
        ids = load_split(cfg.splits_dir, scheme, fold)
        return ChameleonDataset(
            ids=ids,
            csv_path=cfg.csv_path,
            pdb_root=cfg.pdb_root,
            vocab=vocab,
            descriptor_cols=descriptor_cols,
            use_trajectory=cfg.use_trajectory,
            max_conformers=cfg.max_conformers,
            cache_dir=cfg.cache_dir,
        )

    train_ds = make_dataset(cfg.split_scheme, "train")
    val_ds = make_dataset(cfg.split_scheme, "val")

    # Fit the descriptor scaler on the train descriptors only.
    train_desc = np.stack(
        [train_ds.records[pid].descriptors for pid in train_ds.ids], axis=0
    )
    scaler = DescriptorScaler().fit(train_desc)

    # Apply scaler at __getitem__ time via a thin wrapper to avoid mutating
    # the dataset's underlying records.
    def _wrap(ds: ChameleonDataset):
        orig = ds.__getitem__
        def __getitem__(idx: int) -> dict:
            sample = orig(idx)
            sample["descriptors"] = scaler.transform(sample["descriptors"])
            return sample
        ds.__getitem__ = __getitem__  # type: ignore[method-assign]
        return ds

    train_ds = _wrap(train_ds)
    val_ds = _wrap(val_ds)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=chameleon_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=chameleon_collate, pin_memory=True,
    )
    return train_loader, val_loader, scaler, descriptor_cols


def _cosine_warmup_lr(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / max(1, cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, cfg: TrainConfig):
        set_seed(cfg.seed)
        self.cfg = cfg
        self.device = _device_of(cfg)

        self.vocab = ResidueVocab.from_csvs(*cfg.vocab_csvs)
        self.train_loader, self.val_loader, self.scaler, self.descriptor_cols = _build_loaders(cfg, self.vocab)

        self.model = ChameleonNet(
            vocab=self.vocab,
            descriptor_dim=len(self.descriptor_cols),
            hidden_dim=cfg.hidden_dim,
            conformer_layers=cfg.conformer_layers,
            sequence_backend=cfg.sequence_backend,
            peptideclm_name_or_path=cfg.peptideclm_name_or_path,
            head_hidden=cfg.head_hidden,
            dropout=cfg.dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        self.loss_cfg = CompositeLoss(
            lambda_chameleonic=cfg.lambda_chameleonic,
            lambda_triplet=cfg.lambda_triplet,
            pampa_baseline=cfg.pampa_baseline,
        )

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

        self.best_val_mae: float = float("inf")
        self.best_epoch: int = -1
        self.history: List[Dict[str, float]] = []

    def _step(self, batch: dict, train: bool) -> Dict[str, float]:
        batch = _move_batch(batch, self.device)
        targets = batch["pampa"]

        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.set_grad_enabled(train):
            outputs = self.model(batch)
            # Take the head's penultimate layer output as the "fused
            # embedding" passed into the triplet loss. We reconstruct it from
            # the head's first two linear layers to avoid a second forward.
            with torch.set_grad_enabled(train):
                fused = torch.cat(
                    [outputs["h_water"], outputs["h_hexane"], outputs["h_diff"]],
                    dim=-1,
                )
            losses = composite_loss(
                outputs=outputs,
                targets=targets,
                smiles=batch["smiles"],
                fused_embedding=fused,
                cfg=self.loss_cfg,
            )
            if train:
                self.optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.grad_clip
                    )
                self.optimizer.step()

        return {
            "total": float(losses["total"].detach().cpu()),
            "mse": float(losses["mse"].cpu()),
            "chameleonic": float(losses["chameleonic"].cpu()),
            "triplet": float(losses["triplet"].cpu()),
            "preds": outputs["pampa_pred"].detach().cpu().numpy(),
            "targets": targets.detach().cpu().numpy(),
        }

    def _epoch_pass(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        agg: Dict[str, List[float]] = {"total": [], "mse": [], "chameleonic": [], "triplet": []}
        all_preds, all_targets = [], []
        for batch in loader:
            res = self._step(batch, train=train)
            for k in agg:
                agg[k].append(res[k])
            all_preds.append(res["preds"])
            all_targets.append(res["targets"])
        preds = np.concatenate(all_preds) if all_preds else np.zeros(0)
        targs = np.concatenate(all_targets) if all_targets else np.zeros(0)
        metrics = regression_metrics(targs, preds)
        out = {f"loss_{k}": float(np.mean(v)) if v else float("nan") for k, v in agg.items()}
        out.update({f"metric_{k}": v for k, v in metrics.items()})
        return out

    def _set_lr(self, factor: float) -> None:
        for g in self.optimizer.param_groups:
            g["lr"] = self.cfg.lr * factor

    def fit(self) -> None:
        for epoch in range(self.cfg.epochs):
            lr_factor = _cosine_warmup_lr(epoch, self.cfg)
            self._set_lr(lr_factor)
            t0 = time.time()
            train_stats = self._epoch_pass(self.train_loader, train=True)
            val_stats = self._epoch_pass(self.val_loader, train=False)
            dt = time.time() - t0

            log = {"epoch": epoch, "lr_factor": lr_factor, "elapsed_s": dt}
            log.update({f"train/{k}": v for k, v in train_stats.items()})
            log.update({f"val/{k}": v for k, v in val_stats.items()})
            self.history.append(log)
            print(json.dumps(log))

            val_mae = val_stats.get("metric_mae", float("inf"))
            if val_mae < self.best_val_mae:
                self.best_val_mae = val_mae
                self.best_epoch = epoch
                torch.save(
                    {"model": self.model.state_dict(),
                     "scaler_mean": self.scaler.mean,
                     "scaler_std": self.scaler.std,
                     "epoch": epoch,
                     "val_mae": val_mae,
                     "config": self.cfg.to_dict()},
                    self.output_dir / "best.pt",
                )

        (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Evaluate the best-checkpoint model on each requested split scheme's test fold."""
        ckpt_path = self.output_dir / "best.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])

        results: Dict[str, Dict[str, float]] = {}
        for scheme in self.cfg.eval_schemes:
            try:
                ids = load_split(self.cfg.splits_dir, scheme, "test")
            except (FileNotFoundError, ValueError):
                continue
            ds = ChameleonDataset(
                ids=ids,
                csv_path=self.cfg.csv_path,
                pdb_root=self.cfg.pdb_root,
                vocab=self.vocab,
                descriptor_cols=self.descriptor_cols,
                use_trajectory=self.cfg.use_trajectory,
                max_conformers=self.cfg.max_conformers,
                cache_dir=self.cfg.cache_dir,
            )
            scaler = self.scaler
            orig = ds.__getitem__

            def __getitem__(idx: int) -> dict:
                sample = orig(idx)
                sample["descriptors"] = scaler.transform(sample["descriptors"])
                return sample

            ds.__getitem__ = __getitem__  # type: ignore[method-assign]

            loader = DataLoader(
                ds, batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers, collate_fn=chameleon_collate,
            )
            stats = self._epoch_pass(loader, train=False)
            results[scheme] = {k: v for k, v in stats.items() if k.startswith("metric_")}
            print(f"[test/{scheme}] {results[scheme]}")

        (self.output_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))
        return results

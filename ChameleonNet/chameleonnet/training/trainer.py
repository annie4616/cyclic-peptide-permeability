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
    EXTENDED_DESCRIPTORS,
    chameleon_collate,
    full_descriptor_cols,
)
from ..data.residue_vocab import ResidueVocab
from ..data.scaffold_groups import build_scaffold_groups
from ..data.splits import load_split
from ..losses.composite import CompositeLoss, composite_loss
from ..data.delta_descriptors import apply_delta_features
from ..models.chameleonnet import ChameleonNet
from ..models.chameleonnet_v2 import ChameleonNetV2
from ..utils.metrics import regression_metrics
from ..utils.scaler import DescriptorScaler
from ..utils.seed import set_seed
from .config import TrainConfig

try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore


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


def _load_gbr_preds(cfg: TrainConfig) -> Dict[int, float]:
    """Load the pid -> GBR-prediction map when gbr_residual / distillation is on."""
    if not (getattr(cfg, "gbr_residual", False) or getattr(cfg, "lambda_distill", 0.0) > 0):
        return {}
    path = getattr(cfg, "gbr_preds_path", None)
    if not path:
        raise ValueError("gbr_residual/distill requires cfg.gbr_preds_path (run compute_gbr_preds.py)")
    raw = json.loads(Path(path).read_text())
    return {int(k): float(v) for k, v in raw.items()}


def _build_loaders(cfg: TrainConfig, vocab: ResidueVocab):
    if cfg.descriptor_cols:
        descriptor_cols = cfg.descriptor_cols
    elif getattr(cfg, "use_full_descriptors", False):
        descriptor_cols = full_descriptor_cols(cfg.csv_path)
    elif getattr(cfg, "use_extended_descriptors", False):
        descriptor_cols = EXTENDED_DESCRIPTORS
    else:
        descriptor_cols = DEFAULT_DESCRIPTORS
    augment = cfg.augment_delta_descriptors or cfg.model_arch == "v2"
    gbr_preds = _load_gbr_preds(cfg)

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
            augment_delta_descriptors=augment,
            conformer_source=getattr(cfg, "conformer_source", "trajectory"),
            gbr_preds=gbr_preds,
            conformer_stride=getattr(cfg, "conformer_stride", 0),
        )

    train_ds = make_dataset(cfg.split_scheme, "train")
    val_ds = make_dataset(cfg.split_scheme, "val")

    # Fit the descriptor scaler on the train descriptors only.
    # Important: when augmentation is on, fit the scaler in the *augmented*
    # space (raw 12 + 10 derived Δ features) so train/val/test stats line up
    # with what __getitem__ actually returns.
    raw_train = np.stack(
        [train_ds.records[pid].descriptors for pid in train_ds.ids], axis=0
    )
    if augment:
        train_desc = np.stack([apply_delta_features(v) for v in raw_train], axis=0)
    else:
        train_desc = raw_train
    scaler = DescriptorScaler().fit(train_desc)

    # Apply the scaler via the dataset's real (class-level) __getitem__ — see the
    # note in ChameleonDataset.__init__ on why monkeypatching the instance fails.
    train_ds.scaler = scaler
    val_ds.scaler = scaler

    # Conformer augmentation on the TRAIN dataset only (val stays deterministic).
    if getattr(cfg, "conformer_augment", False):
        train_ds.conformer_augment = True
        train_ds.coord_noise = cfg.coord_noise

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=chameleon_collate, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=chameleon_collate, pin_memory=True,
    )
    # Real descriptor dim seen by the model (raw 12, or 22 with Δ-augment).
    descriptor_dim = train_ds.descriptor_dim
    return train_loader, val_loader, scaler, descriptor_cols, descriptor_dim


def _cosine_warmup_lr(epoch: int, cfg: TrainConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return (epoch + 1) / max(1, cfg.warmup_epochs)
    progress = (epoch - cfg.warmup_epochs) / max(1, cfg.epochs - cfg.warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class Trainer:
    def __init__(self, cfg: TrainConfig):
        set_seed(cfg.seed) # 시드 고정
        self.cfg = cfg
        # cuda or cpu as specified in config
        self.device = _device_of(cfg)

        self.vocab = ResidueVocab.from_csvs(*cfg.vocab_csvs)
        (
            self.train_loader,
            self.val_loader,
            self.scaler,
            self.descriptor_cols,
            self.descriptor_dim,
        ) = _build_loaders(cfg, self.vocab)

        if cfg.model_arch == "v2":
            ModelCls = ChameleonNetV2
        elif cfg.model_arch == "v1":
            ModelCls = ChameleonNet
        else:
            raise ValueError(f"Unknown model_arch: {cfg.model_arch!r}; expected 'v1' or 'v2'")

        # Scaffold adversary: build train-only scaffold-cluster labels and a
        # pid -> group lookup. Only active when both knobs are set; the model
        # then grows a gradient-reversal head over the learned representation.
        self.scaffold_adv_on = cfg.lambda_adv > 0 and cfg.adv_n_groups > 0
        self.scaffold_groups: Dict[int, int] = {}
        n_scaffold_groups = 0
        if self.scaffold_adv_on:
            train_ids = load_split(cfg.splits_dir, cfg.split_scheme, "train")
            self.scaffold_groups = build_scaffold_groups(
                train_ids=train_ids,
                csv_path=cfg.csv_path,
                n_groups=cfg.adv_n_groups,
                seed=cfg.seed,
                cache_dir=cfg.cache_dir,
                scheme=cfg.split_scheme,
            )
            n_scaffold_groups = cfg.adv_n_groups

        adv_kwargs = (
            {
                "n_scaffold_groups": n_scaffold_groups,
                "adv_hidden": cfg.adv_hidden,
                "modality_dropout": cfg.modality_dropout,
                "physics_residual": cfg.physics_residual,
                "info_bottleneck": cfg.info_bottleneck,
                "gbr_residual": cfg.gbr_residual,
                "distill": cfg.lambda_distill > 0,
                "hexane_only": getattr(cfg, "hexane_only", False),
                "water_only": getattr(cfg, "water_only", False),
                "no_diff": getattr(cfg, "no_diff", False),
                "no_conformer": getattr(cfg, "no_conformer", False),
                "conformer_encoder_arch": getattr(cfg, "conformer_encoder_arch", "egnn"),
                "pool_type": getattr(cfg, "pool_type", "attention"),
                "pool_transformer_layers": getattr(cfg, "pool_transformer_layers", 2),
                "pool_transformer_heads": getattr(cfg, "pool_transformer_heads", 4),
            }
            if cfg.model_arch == "v1"
            else {}
        )
        self.model = ModelCls(
            vocab=self.vocab,
            descriptor_dim=self.descriptor_dim,
            hidden_dim=cfg.hidden_dim,
            conformer_layers=cfg.conformer_layers,
            sequence_backend=cfg.sequence_backend,
            peptideclm_name_or_path=cfg.peptideclm_name_or_path,
            helmbert_name_or_path=cfg.helmbert_name_or_path,
            head_hidden=cfg.head_hidden,
            dropout=cfg.dropout,
            residue_emb_path=cfg.residue_emb_path,
            **adv_kwargs,
        ).to(self.device)
        # Optionally warm-start the 3D encoder from CREMP denoising pretraining.
        pe = getattr(cfg, "pretrained_encoder_path", None)
        if pe:
            ckpt = torch.load(pe, map_location=self.device, weights_only=False)
            sd = ckpt["encoder"] if "encoder" in ckpt else ckpt
            missing, unexpected = self.model.conformer_encoder.load_state_dict(sd, strict=False)
            print(json.dumps({"pretrained_encoder": pe,
                              "loaded": len(sd), "missing": len(missing), "unexpected": len(unexpected)}))

        if self.scaffold_adv_on and getattr(self.model, "scaffold_adversary", None) is None:
            # Model arch doesn't support the adversary yet (e.g. v2) — fail loud
            # rather than silently training a plain model.
            raise ValueError(
                f"lambda_adv>0 but model_arch={cfg.model_arch!r} has no scaffold "
                "adversary; the GRL head is currently wired for model_arch='v1'."
            )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )

        # The chameleonic loss compares water vs hexane; with either single-env
        # ablation on it has no meaning, so force its weight to 0.
        _single_env = (getattr(cfg, "hexane_only", False) or getattr(cfg, "water_only", False)
                       or getattr(cfg, "no_diff", False) or getattr(cfg, "no_conformer", False))
        _lambda_cham = 0.0 if _single_env else cfg.lambda_chameleonic
        self.loss_cfg = CompositeLoss(
            lambda_chameleonic=_lambda_cham,
            lambda_triplet=cfg.lambda_triplet,
            lambda_residue_psa=cfg.lambda_residue_psa if cfg.model_arch == "v2" else 0.0,
            lambda_adv=cfg.lambda_adv if self.scaffold_adv_on else 0.0,
            lambda_resid_l2=cfg.lambda_resid_l2,
            lambda_ib=cfg.lambda_ib,
            lambda_distill=cfg.lambda_distill,
            pampa_baseline=cfg.pampa_baseline,
            triplet_sim_high=getattr(cfg, "triplet_sim_high", 0.7),
            triplet_sim_low=getattr(cfg, "triplet_sim_low", 0.4),
            triplet_morgan_radius=getattr(cfg, "triplet_morgan_radius", 4),
            triplet_morgan_nbits=getattr(cfg, "triplet_morgan_nbits", 4096),
        )

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

        self.best_val_mae: float = float("inf")
        self.best_epoch: int = -1
        self.history: List[Dict[str, float]] = []

        self._wandb_run = self._maybe_init_wandb()

    def _maybe_init_wandb(self):
        if wandb is None or self.cfg.wandb_mode == "disabled" or not self.cfg.wandb_project:
            return None
        return wandb.init(
            project=self.cfg.wandb_project,
            entity=self.cfg.wandb_entity,
            name=self.cfg.wandb_run_name or Path(self.cfg.output_dir).name,
            mode=self.cfg.wandb_mode,
            config=self.cfg.to_dict(),
            dir=str(self.output_dir),
            reinit=True,
        )

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
            scaffold_groups = None
            if self.scaffold_adv_on and "pids" in batch:
                scaffold_groups = torch.tensor(
                    [self.scaffold_groups.get(int(p), -1) for p in batch["pids"]],
                    dtype=torch.long,
                    device=self.device,
                )
            losses = composite_loss(
                outputs=outputs,
                targets=targets,
                smiles=batch["smiles"],
                fused_embedding=fused,
                delta_psa_target=batch.get("delta_psa_raw"),
                scaffold_groups=scaffold_groups,
                gbr_pred=batch.get("gbr_pred"),
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
            "residue_psa": float(losses.get("residue_psa", losses["mse"].new_zeros(())).cpu()),
            "adv": float(losses.get("adv", losses["mse"].new_zeros(())).cpu()),
            "preds": outputs["pampa_pred"].detach().cpu().numpy(),
            "targets": targets.detach().cpu().numpy(),
        }

    def _epoch_pass(self, loader: DataLoader, train: bool) -> Dict[str, float]:
        agg: Dict[str, List[float]] = {
            "total": [], "mse": [], "chameleonic": [], "triplet": [], "residue_psa": [], "adv": [],
        }
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
            # Ramp the gradient-reversal strength 0 -> adv_lambda_max so the
            # adversary head learns to read scaffold identity before the encoder
            # starts being pushed to erase it (stabilises DANN training).
            if self.scaffold_adv_on:
                w = self.cfg.adv_warmup_epochs
                frac = 1.0 if w <= 0 else min(1.0, epoch / w)
                self.model.adv_lambda = float(self.cfg.adv_lambda_max) * frac
            t0 = time.time()
            train_stats = self._epoch_pass(self.train_loader, train=True)
            val_stats = self._epoch_pass(self.val_loader, train=False)
            dt = time.time() - t0

            log = {"epoch": epoch, "lr_factor": lr_factor, "elapsed_s": dt}
            log.update({f"train/{k}": v for k, v in train_stats.items()})
            log.update({f"val/{k}": v for k, v in val_stats.items()})
            self.history.append(log)
            print(json.dumps(log))
            if self._wandb_run is not None:
                self._wandb_run.log(log, step=epoch)

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

            # Early stopping on val MAE plateau (disabled when patience<=0).
            # Don't start counting until warmup is complete — early-epoch val
            # MAE during a tiny lr is unreliable and can latch the "best" too
            # early, killing the run before real training begins.
            patience = getattr(self.cfg, "early_stop_patience", 0)
            if (
                patience > 0
                and epoch >= self.cfg.warmup_epochs + patience
                and (epoch - self.best_epoch) >= patience
            ):
                print(json.dumps({
                    "early_stop": True,
                    "epoch": epoch,
                    "best_epoch": self.best_epoch,
                    "best_val_mae": self.best_val_mae,
                    "patience": patience,
                }))
                if self._wandb_run is not None:
                    self._wandb_run.summary["early_stopped_epoch"] = epoch
                break

        (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2))
        if self._wandb_run is not None:
            self._wandb_run.summary["best_epoch"] = self.best_epoch
            self._wandb_run.summary["best_val_mae"] = self.best_val_mae

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
                augment_delta_descriptors=(
                    self.cfg.augment_delta_descriptors or self.cfg.model_arch == "v2"
                ),
                conformer_source=getattr(self.cfg, "conformer_source", "trajectory"),
                gbr_preds=_load_gbr_preds(self.cfg),
                conformer_stride=getattr(self.cfg, "conformer_stride", 0),
            )
            ds.scaler = self.scaler

            loader = DataLoader(
                ds, batch_size=self.cfg.batch_size, shuffle=False,
                num_workers=self.cfg.num_workers, collate_fn=chameleon_collate,
            )
            stats = self._epoch_pass(loader, train=False)
            results[scheme] = {k: v for k, v in stats.items() if k.startswith("metric_")}
            print(f"[test/{scheme}] {results[scheme]}")

        (self.output_dir / "test_metrics.json").write_text(json.dumps(results, indent=2))
        if self._wandb_run is not None:
            flat = {f"test/{scheme}/{k}": v for scheme, m in results.items() for k, v in m.items()}
            self._wandb_run.log(flat)
            for k, v in flat.items():
                self._wandb_run.summary[k] = v
            self._wandb_run.finish()
        return results

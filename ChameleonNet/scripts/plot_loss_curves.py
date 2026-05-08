"""Plot loss / metric curves from a training run's history.json.

Usage:
    python scripts/plot_loss_curves.py --run runs/v2_local
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LOSS_KEYS = ["total", "mse", "chameleonic", "triplet", "residue_psa"]
METRIC_KEYS = ["mae", "mse", "r2", "pearson"]


def load_history(run_dir: Path) -> list[dict]:
    p = run_dir / "history.json"
    if not p.exists():
        raise FileNotFoundError(f"No history.json under {run_dir}")
    return json.loads(p.read_text())


def _series(history, prefix: str, key: str) -> np.ndarray:
    full = f"{prefix}/{key}"
    arr = np.array([h.get(full, np.nan) for h in history], dtype=float)
    return arr


def plot(run_dir: Path, out_path: Path) -> None:
    history = load_history(run_dir)
    epochs = np.array([h["epoch"] for h in history])

    test_metrics_path = run_dir / "test_metrics.json"
    test_metrics = json.loads(test_metrics_path.read_text()) if test_metrics_path.exists() else {}

    val_mae = _series(history, "val", "metric_mae")
    best_epoch = int(epochs[np.nanargmin(val_mae)]) if np.isfinite(val_mae).any() else None

    fig, axes = plt.subplots(3, 2, figsize=(13, 11))
    fig.suptitle(
        f"ChameleonNet V2 training curves — {run_dir.name}\n"
        f"({len(history)} epochs"
        + (f", best val MAE @ epoch {best_epoch}" if best_epoch is not None else "")
        + ")",
        fontsize=12, weight="bold",
    )

    # 1. Total loss train vs val
    ax = axes[0, 0]
    for prefix, color, label in [("train", "#1f77b4", "train"), ("val", "#d62728", "val")]:
        y = _series(history, prefix, "loss_total")
        ax.plot(epochs, y, color=color, label=label, linewidth=1.5)
    ax.set_title("Total composite loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5, label="best epoch")

    # 2. Loss components (val)
    ax = axes[0, 1]
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for k, c in zip(LOSS_KEYS, palette):
        y = _series(history, "val", f"loss_{k}")
        if np.isfinite(y).any() and not np.allclose(np.nan_to_num(y), 0):
            ax.plot(epochs, y, label=k, color=c, linewidth=1.2)
    ax.set_title("Validation loss components")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # 3. MAE
    ax = axes[1, 0]
    for prefix, color, label in [("train", "#1f77b4", "train"), ("val", "#d62728", "val")]:
        y = _series(history, prefix, "metric_mae")
        ax.plot(epochs, y, color=color, label=label, linewidth=1.5)
    ax.set_title("MAE")
    ax.set_xlabel("epoch"); ax.set_ylabel("MAE")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    # Test horizontal lines per scheme
    if test_metrics:
        for scheme, m in test_metrics.items():
            v = m.get("metric_mae")
            if v is not None:
                ax.axhline(v, linestyle=":", alpha=0.6,
                           label=f"test/{scheme}={v:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4. MSE
    ax = axes[1, 1]
    for prefix, color, label in [("train", "#1f77b4", "train"), ("val", "#d62728", "val")]:
        # if metric_mse not present (older histories), derive from RMSE if available
        y = _series(history, prefix, "metric_mse")
        if not np.isfinite(y).any():
            rmse = _series(history, prefix, "metric_rmse")
            y = rmse * rmse
        ax.plot(epochs, y, color=color, label=label, linewidth=1.5)
    ax.set_title("MSE")
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    if test_metrics:
        for scheme, m in test_metrics.items():
            mse = m.get("metric_mse")
            if mse is None and "metric_rmse" in m:
                mse = m["metric_rmse"] ** 2
            if mse is not None:
                ax.axhline(mse, linestyle=":", alpha=0.6,
                           label=f"test/{scheme}={mse:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 5. R²
    ax = axes[2, 0]
    for prefix, color, label in [("train", "#1f77b4", "train"), ("val", "#d62728", "val")]:
        y = _series(history, prefix, "metric_r2")
        ax.plot(epochs, y, color=color, label=label, linewidth=1.5)
    ax.set_title("R²")
    ax.set_xlabel("epoch"); ax.set_ylabel("R²")
    ax.axhline(0, color="gray", linestyle="-", alpha=0.4, linewidth=0.8)
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    if test_metrics:
        for scheme, m in test_metrics.items():
            v = m.get("metric_r2")
            if v is not None:
                ax.axhline(v, linestyle=":", alpha=0.6,
                           label=f"test/{scheme}={v:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    # 6. Pearson r
    ax = axes[2, 1]
    for prefix, color, label in [("train", "#1f77b4", "train"), ("val", "#d62728", "val")]:
        y = _series(history, prefix, "metric_pearson")
        ax.plot(epochs, y, color=color, label=label, linewidth=1.5)
    ax.set_title("Pearson r")
    ax.set_xlabel("epoch"); ax.set_ylabel("Pearson r")
    if best_epoch is not None:
        ax.axvline(best_epoch, color="green", linestyle="--", alpha=0.5)
    if test_metrics:
        for scheme, m in test_metrics.items():
            v = m.get("metric_pearson")
            if v is not None:
                ax.axhline(v, linestyle=":", alpha=0.6,
                           label=f"test/{scheme}={v:.3f}")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=str, default="runs/v2_local",
                   help="Path to a training run directory containing history.json")
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG path (default: <run>/loss_curves.png)")
    args = p.parse_args()
    run = Path(args.run)
    out = Path(args.out) if args.out else run / "loss_curves.png"
    plot(run, out)


if __name__ == "__main__":
    main()

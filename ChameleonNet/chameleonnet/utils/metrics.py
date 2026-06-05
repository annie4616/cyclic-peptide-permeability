"""Regression metrics. Stays NumPy-only so it works on CPU eval too."""

from __future__ import annotations

from typing import Dict

import numpy as np


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return {"mae": float("nan"), "mse": float("nan"), "r2": float("nan"),
                "pearson": float("nan"), "spearman": float("nan")}
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err * err))
    var = float(np.var(y_true))
    r2 = float(1.0 - mse / var) if var > 0 else float("nan")
    if y_true.std() > 0 and y_pred.std() > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
        # Spearman (SCC) = Pearson on average ranks. Reported to match the
        # baseline table's SCC column without adding a scipy dependency.
        rt, rp = _rankdata(y_true), _rankdata(y_pred)
        spearman = float(np.corrcoef(rt, rp)[0, 1])
    else:
        pearson = float("nan")
        spearman = float("nan")
    return {"mae": mae, "mse": mse, "r2": r2, "pearson": pearson, "spearman": spearman}


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1..n) with ties averaged — matches scipy.stats.rankdata."""
    a = np.asarray(a, dtype=np.float64)
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts), dtype=np.float64)
    np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]

"""Summarize the two hexane-only runs (OD_Murcko + ID) and print a comparison.

Reads each run's test_metrics.json (final test eval) and history.json (best val)
and writes a combined hexane_only_summary.json + markdown table to runs/.
Run after both trainings finish.
"""

from __future__ import annotations

import json
from pathlib import Path

RUNS = Path("/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet/runs")
TARGETS = {
    "hexane_only / OD_Murcko": RUNS / "final_extdesc_hexane_only",
    "hexane_only / ID": RUNS / "final_extdesc_hexane_only_id",
    "xfmr_pool / OD_Murcko": RUNS / "final_extdesc_xfmrpool",
    "xfmr_pool / ID": RUNS / "final_extdesc_xfmrpool_id",
}

METRIC_KEYS = ["metric_mae", "metric_mse", "metric_r2", "metric_pearson", "metric_spearman"]


def _best_val(hist_path: Path):
    """Best (lowest val MAE) epoch from history.json, if present."""
    if not hist_path.exists():
        return None
    hist = json.loads(hist_path.read_text())
    rows = hist if isinstance(hist, list) else hist.get("epochs", [])
    best = None
    for r in rows:
        mae = r.get("val/metric_mae")
        if mae is None:
            continue
        if best is None or mae < best["val/metric_mae"]:
            best = r
    return best


def main() -> None:
    summary = {}
    lines = []
    header = ["run", "scheme"] + [k.replace("metric_", "test_") for k in METRIC_KEYS] + ["best_val_mae", "best_val_r2"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for name, d in TARGETS.items():
        tm_path = d / "test_metrics.json"
        if not tm_path.exists():
            lines.append(f"| {name} | (no test_metrics.json — run unfinished?) |" + " |" * (len(header) - 2))
            summary[name] = {"status": "missing"}
            continue
        tm = json.loads(tm_path.read_text())
        # test_metrics.json: {scheme: {metric_*: val}}
        for scheme, m in tm.items():
            best = _best_val(d / "history.json")
            row = [name, scheme]
            for k in METRIC_KEYS:
                v = m.get(k)
                row.append(f"{v:.4f}" if isinstance(v, (int, float)) else "-")
            bvm = best.get("val/metric_mae") if best else None
            bvr = best.get("val/metric_r2") if best else None
            row.append(f"{bvm:.4f}" if isinstance(bvm, (int, float)) else "-")
            row.append(f"{bvr:.4f}" if isinstance(bvr, (int, float)) else "-")
            lines.append("| " + " | ".join(str(x) for x in row) + " |")
            summary.setdefault(name, {})[scheme] = {
                "test": m,
                "best_val_mae": bvm,
                "best_val_r2": bvr,
            }

    table = "\n".join(lines)
    print(table)
    out_json = RUNS / "hexane_only_summary.json"
    out_md = RUNS / "hexane_only_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))
    out_md.write_text("# Hexane-only EGNN — OD_Murcko vs ID\n\n" + table + "\n")
    print(f"\nwrote {out_json}\nwrote {out_md}")


if __name__ == "__main__":
    main()

"""Run only the multi-scheme test evaluation for a finished training run.

Useful when training was interrupted before evaluate_all() got a chance to fire,
or when you want to re-evaluate a saved best.pt after changing eval_schemes.

Usage:
    python -m scripts.eval_only --config configs/v2_id.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from chameleonnet.training import Trainer, load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)
    # Avoid creating a new wandb run for a pure-eval pass — write only to
    # test_metrics.json on disk.
    cfg.wandb_mode = "disabled"
    trainer = Trainer(cfg)
    # Skip fit(); just load best.pt (if present) and evaluate test schemes.
    trainer.evaluate_all()


if __name__ == "__main__":
    main()

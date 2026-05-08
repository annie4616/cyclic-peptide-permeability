"""CLI entry point: train + multi-scheme test eval.

Usage:
    python -m scripts.train --config configs/default.yaml
    python -m scripts.train --config configs/default.yaml --epochs 30  # override

Any field on TrainConfig can be overridden via --field value.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file as both a module and a script.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from chameleonnet.training import Trainer, TrainConfig, load_config


def _override_config(cfg: TrainConfig, extra: list[str]) -> TrainConfig:
    """Parse --key value pairs after the known args."""
    it = iter(extra)
    for tok in it:
        if not tok.startswith("--"):
            continue
        key = tok.lstrip("-")
        try:
            val = next(it)
        except StopIteration:
            raise SystemExit(f"--{key} expected a value")
        if not hasattr(cfg, key):
            raise SystemExit(f"Unknown config field: {key}")
        current = getattr(cfg, key)
        if isinstance(current, bool):
            cast = val.lower() in {"1", "true", "yes"}
        elif isinstance(current, int):
            cast = int(val)
        elif isinstance(current, float):
            cast = float(val)
        elif isinstance(current, list):
            cast = val.split(",")
        else:
            cast = val
        setattr(cfg, key, cast)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/hdd0/sohyun/cyclic-peptide-permeability/ChameleonNet/configs/v2_local.yaml",
                        help="Optional YAML config; defaults baked into TrainConfig if omitted.")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip the test-set evaluation pass at end.")
    args, extra = parser.parse_known_args()

    # yaml 파일 안의 경로들과  반환
    cfg = load_config(args.config) if args.config else TrainConfig()
    # argparse를 정의하지 않고서도 config 넣을 수 있음
    cfg = _override_config(cfg, extra)

    trainer = Trainer(cfg)
    trainer.fit()
    if not args.no_eval:
        trainer.evaluate_all()


if __name__ == "__main__":
    main()

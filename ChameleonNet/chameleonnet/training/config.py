"""Dataclass-based config + a tiny YAML loader.

We avoid Hydra/OmegaConf for now — one small dataclass is easier to read.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TrainConfig:
    # paths
    csv_path: str = "/ssd0/sohyun/cyclic_peptide_permeability/data/CycPeptMPDB-4D_with_assay_descriptors.csv"
    pdb_root: str = "/ssd0/sohyun/cyclic_peptide_permeability/CycPeptMPDB-4D"
    splits_dir: str = "/ssd0/sohyun/cyclic_peptide_permeability/splits"
    vocab_csvs: List[str] = field(default_factory=lambda: [
        "/ssd0/sohyun/cyclic_peptide_permeability/water_residue_vocab.csv",
        "/ssd0/sohyun/cyclic_peptide_permeability/hexane_residue_vocab.csv",
    ])
    cache_dir: Optional[str] = "/ssd0/sohyun/cyclic_peptide_permeability/ChameleonNet/.cache_pdb"
    output_dir: str = "/ssd0/sohyun/cyclic_peptide_permeability/ChameleonNet/runs/default"

    # split scheme
    split_scheme: str = "ID"  # "ID" | "OD" | "Cliff"

    # data
    use_trajectory: bool = True
    max_conformers: int = 16
    descriptor_cols: Optional[List[str]] = None  # None → DEFAULT_DESCRIPTORS

    # model
    model_arch: str = "v1"  # "v1" | "v2" — v2 is residue-resolved chameleonic
    augment_delta_descriptors: bool = False  # append derived Δ-features (V2 default on)
    hidden_dim: int = 128
    conformer_layers: int = 3
    sequence_backend: str = "learned"  # "learned" | "peptideclm"
    peptideclm_name_or_path: Optional[str] = None
    head_hidden: int = 256
    dropout: float = 0.1

    # losses
    lambda_chameleonic: float = 0.1
    lambda_triplet: float = 0.05
    pampa_baseline: float = -8.0
    # V2 residue-resolved auxiliary that ties the per-residue chameleon weight
    # to the global ΔPSA — small weight; turn off with 0.
    lambda_residue_psa: float = 0.05

    # optim
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0

    # eval
    eval_schemes: List[str] = field(default_factory=lambda: ["ID", "OD", "Cliff"])
    seed: int = 42
    device: str = "cuda"

    # logging
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"  # "online" | "offline" | "disabled"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> TrainConfig:
    """Minimal YAML loader. Falls back to plain dict if PyYAML missing."""
    path = Path(path)
    text = path.read_text()
    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except ImportError:
        # Very small fallback parser: only supports `key: value` lines and
        # nested lists prefixed with `- `. Good enough for our tiny configs.
        data = _tiny_yaml_parse(text)
    cfg = TrainConfig(**{k: v for k, v in data.items() if k in TrainConfig.__dataclass_fields__})
    return cfg


def _tiny_yaml_parse(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    current_key: Optional[str] = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  - "):
            if current_key is None:
                continue
            out.setdefault(current_key, []).append(_coerce(line[4:].strip()))
            continue
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if val == "":
            current_key = key
            out[key] = []
        else:
            current_key = None
            out[key] = _coerce(val)
    return out


def _coerce(s: str):
    s = s.strip().strip('"').strip("'")
    if s.lower() in {"true", "false"}:
        return s.lower() == "true"
    if s.lower() in {"null", "none", "~"}:
        return None
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s

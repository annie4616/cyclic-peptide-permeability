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
    descriptor_cols: Optional[List[str]] = None  # None → DEFAULT_DESCRIPTORS (or EXTENDED if flag below)
    use_extended_descriptors: bool = False  # add 17 RDKit features after the 11 MD-derived ones
    use_full_descriptors: bool = False  # use ALL numeric non-leakage CSV columns (keeps MD-11 first)
    conformer_source: str = "trajectory"  # "trajectory" | "centroids"

    # model
    model_arch: str = "v1"  # "v1" | "v2" — v2 is residue-resolved chameleonic
    augment_delta_descriptors: bool = False  # append derived Δ-features (V2 default on)
    hidden_dim: int = 128
    conformer_layers: int = 3
    sequence_backend: str = "learned"  # "learned" | "peptideclm" | "helmbert"
    peptideclm_name_or_path: Optional[str] = None
    helmbert_name_or_path: Optional[str] = None
    head_hidden: int = 256
    dropout: float = 0.1
    # OOD-regularization levers (all no-ops at default; v1 model honours them).
    # modality_dropout: during training, randomly zero the conformer block and
    #   the sequence block independently with this prob (descriptors are never
    #   dropped) so the model can't lean solely on the scaffold-memorising
    #   learned branches. physics_residual: main prediction comes from the
    #   descriptor branch; the learned branches add only a gated residual with
    #   an L2 penalty (lambda_resid_l2) — so OOD gracefully falls back to
    #   transferable physics. info_bottleneck: variational bottleneck on the
    #   learned representation with KL weight lambda_ib.
    modality_dropout: float = 0.0
    physics_residual: bool = False
    lambda_resid_l2: float = 0.0
    info_bottleneck: bool = False
    lambda_ib: float = 0.0
    # gbr_residual: prediction = gbr_pred + gated learned residual. Needs a
    # precomputed pid->gbr_pred map (scripts/compute_gbr_preds.py); the deep
    # model only has to learn what the descriptor-GBR misses (the chameleonic
    # 3D signal), which is the OOD-robust framing. gbr_preds_path points at the
    # JSON; lambda_distill (optional) pulls a learned-only head toward gbr_pred.
    gbr_residual: bool = False
    gbr_preds_path: Optional[str] = None
    lambda_distill: float = 0.0
    # Optional path to a .pt file produced by `scripts/build_residue_lm_embeddings.py`.
    # When set, the conformer encoder's residue embedding is replaced by a
    # frozen lookup over PeptideCLM-2 vectors (one row per ResidueVocab token),
    # followed by a small learned Linear projection into `hidden_dim`. Leaving
    # it None preserves the original behavior (random embedding learned from
    # scratch) so existing baselines stay reproducible.
    residue_emb_path: Optional[str] = None

    # losses
    lambda_chameleonic: float = 0.1
    lambda_triplet: float = 0.05
    pampa_baseline: float = -8.0
    # V2 residue-resolved auxiliary that ties the per-residue chameleon weight
    # to the global ΔPSA — small weight; turn off with 0.
    lambda_residue_psa: float = 0.05
    # Triplet-mining knobs (used only when lambda_triplet > 0). Defaults match
    # the original behavior; raise resolution + lower sim_high for cyclic
    # peptides so mined triplets reflect *real* near-neighbors instead of
    # fingerprint-collision noise.
    triplet_sim_high: float = 0.7
    triplet_sim_low: float = 0.4
    triplet_morgan_radius: int = 2
    triplet_morgan_nbits: int = 2048

    # Scaffold adversary (gradient-reversal). Removes spurious scaffold-identity
    # signal from the learned representation to improve scaffold-OOD
    # generalization. Disabled when lambda_adv <= 0 or adv_n_groups <= 0.
    lambda_adv: float = 0.0           # weight on the adversary CE in total loss
    adv_n_groups: int = 32            # # train-only scaffold (KMeans) clusters
    adv_hidden: int = 128             # adversary MLP hidden width
    adv_lambda_max: float = 1.0       # peak GRL reversal strength
    adv_warmup_epochs: int = 10       # epochs to ramp GRL lambda 0 -> max

    # optim
    batch_size: int = 16
    num_workers: int = 4
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    early_stop_patience: int = 0  # 0 = disabled; otherwise stop if val MAE doesn't improve for N epochs

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
    except ImportError: # Pyyaml이 설치되어 있지 않은 경우, 간단한 YAML 파서를 사용하여 데이터를 로드
        # Very small fallback parser: only supports `key: value` lines and
        # nested lists prefixed with `- `. Good enough for our tiny configs.
        data = _tiny_yaml_parse(text)
    cfg = TrainConfig(**{k: v for k, v in data.items() if k in TrainConfig.__dataclass_fields__})
    # train config에 있는 필드만 사용하여 TrainConfig 객체를 생성
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

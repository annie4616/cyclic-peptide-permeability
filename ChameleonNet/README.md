# ChameleonNet

Dual-environment conformer-ensemble model for cyclic peptide permeability prediction on CycPeptMPDB-4D.

The core hypothesis: permeable cyclic peptides exhibit a *chameleonic* effect — they reorganize geometry between aqueous and lipid environments. Rather than ingesting one conformer per peptide, ChameleonNet ingests both the water and hexane conformer ensembles and feeds the **difference** between their pooled embeddings into the predictor.

## Architecture

```
Water conformers  ──► 3D EGNN ──► attention pool ──► h_water
                       (shared)                          │
Hexane conformers ──► 3D EGNN ──► attention pool ──► h_hexane
                                                         │
                              chameleonic vector =       ▼
                              [h_w, h_h, h_w − h_h]
Sequence/SMILES   ──► SequenceEncoder ─────────────► h_seq
4D descriptors    ──► DescriptorMLP   ─────────────► h_desc
                                                         │
                       concat ──► MLP head ──► PAMPA prediction
                                  └──► chameleonic-magnitude head (auxiliary)
```

### Three novelties vs. prior work

1. **Dual-environment conformer ensemble encoder.** MCPerm uses one conformer in one environment; ChameleonNet uses K water + K hexane conformers, attention-pooled, and feeds the explicit difference vector to the head. The chameleonic hypothesis is hard-wired into the architecture.

2. **Chameleonic-magnitude regression (auxiliary).** `||h_water − h_hexane||` is regressed against `(PAMPA − baseline)`. Higher-permeability peptides are pushed to have larger water/hexane geometric divergence, which is exactly the chameleonic signal in regression form (not the binary contrast a classification setup would give).

3. **PeptideCLM reuse + Tanimoto triplet loss.** SMILES branch reuses MCPerm's pretrained PeptideCLM (or falls back to a learned-from-scratch sequence encoder). The triplet loss from MultiCycPermea attacks permeability cliffs directly.

## Folder layout

```
ChameleonNet/
├── chameleonnet/
│   ├── data/
│   │   ├── pdb_parser.py         # multi-MODEL PDB → (K, N, 3) tensors
│   │   ├── residue_vocab.py      # 3-letter + non-canonical residue mapping
│   │   ├── dataset.py            # ChameleonDataset + collate
│   │   └── splits.py             # ID/OD/Cliff loaders
│   ├── models/
│   │   ├── conformer_encoder.py  # EGNN-style 3D encoder
│   │   ├── attention_pool.py     # per-sample attention over conformers
│   │   ├── sequence_encoder.py   # learned or PeptideCLM backend
│   │   ├── descriptor_mlp.py     # 4D descriptor MLP
│   │   └── chameleonnet.py       # main module wiring all branches
│   ├── losses/
│   │   ├── chameleonic.py        # ||h_w − h_h|| regression + aux head MSE
│   │   ├── triplet.py            # Tanimoto-mined triplet margin loss
│   │   └── composite.py          # weighted sum + sub-term logging
│   ├── training/
│   │   ├── config.py             # TrainConfig dataclass + tiny YAML loader
│   │   └── trainer.py            # main training loop + multi-scheme eval
│   └── utils/
│       ├── metrics.py            # MAE, RMSE, R², Pearson
│       ├── scaler.py             # train-stat descriptor standardizer
│       └── seed.py
├── configs/
│   ├── default.yaml              # ID-split baseline
│   ├── od_split.yaml             # out-of-distribution evaluation
│   └── cliff.yaml                # permeability-cliff evaluation
├── scripts/
│   ├── train.py                  # CLI entry point
│   └── build_pdb_cache.py        # one-shot PDB → .npz cache builder
└── README.md
```

## Usage

### 1. Pre-cache PDBs (recommended)

The first epoch is dominated by PDB parsing. Cache once:

```bash
cd ChameleonNet
python scripts/build_pdb_cache.py
```

This writes `.cache_pdb/{pid}_{Water|Hexane}_traj.npz` for every peptide in the assay CSV.

### 2. Train

```bash
# ID split (random 80/10/10)
python scripts/train.py --config configs/default.yaml

# Source-grouped OD split — tests generalization across literature sources
python scripts/train.py --config configs/od_split.yaml

# Permeability-cliff split — tests Tanimoto-similar / ΔPAMPA-large pairs
python scripts/train.py --config configs/cliff.yaml
```

Any field in `TrainConfig` can be overridden on the CLI:

```bash
python scripts/train.py --config configs/default.yaml --epochs 30 --lr 0.001
```

### 3. Evaluation

The trainer automatically evaluates the best checkpoint on every split scheme listed in `eval_schemes` (default: ID, OD, Cliff) and writes `runs/<name>/test_metrics.json`. Per-epoch logs land in `runs/<name>/history.json`.

## Configuration knobs that matter

- `max_conformers` — how many trajectory frames to subsample per environment. 16 is a reasonable default; raising it improves the chameleonic estimate but slows the encoder quadratically (the EGNN edge MLP is O(N² × K)).
- `lambda_chameleonic` — weight on the chameleonic-magnitude auxiliary loss. Start at 0.1; raise it if the diff vector seems uninformative (validate by tracking the auxiliary head's MAE).
- `lambda_triplet` — weight on the Tanimoto triplet loss. Set higher (~0.2) for the Cliff split where it matters most. Requires RDKit; degrades to 0 if RDKit is missing.
- `sequence_backend` — `"learned"` for a tiny self-contained Transformer; `"peptideclm"` to wrap a HuggingFace PeptideCLM checkpoint via `peptideclm_name_or_path`.

## Dependencies

Required: `torch`, `numpy`. Optional: `pyyaml` (configs), `rdkit` (Tanimoto triplets), `transformers` (PeptideCLM backend). The code degrades gracefully when optional deps are missing.

## Notes

- The 3D encoder shares weights across both environments — only the geometry differs between `h_water` and `h_hexane`, which is what makes the difference vector physically meaningful.
- The two attention poolers are *not* shared: solvents may weight conformers differently.
- `pampa_baseline = -8.0` matches the lower clip of CycPeptMPDB PAMPA values, so the chameleonic-magnitude target is non-negative.

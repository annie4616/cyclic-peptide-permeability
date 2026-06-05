"""Emit Round-1 candidate YAML configs for the OD_Murcko campaign.

All candidates share a fast screening backbone (learned sequence encoder,
60 epochs + early stop) and differ only by the orthogonal OOD levers we added.
Finalists are later re-run with the heavier peptideclm backbone + 3 seeds.
"""
import os
ROOT = "/hdd0/sohyun/cyclic-peptide-permeability"
CFGDIR = f"{ROOT}/ChameleonNet/configs/campaign"
os.makedirs(CFGDIR, exist_ok=True)

BASE = dict(
    csv_path=f"{ROOT}/data/CycPeptMPDB-4D_with_assay_descriptors_preprocessed.csv",
    pdb_root=f"{ROOT}/CycPeptMPDB-4D",
    splits_dir=f"{ROOT}/splits",
    cache_dir=f"{ROOT}/ChameleonNet/.cache_pdb",
    split_scheme="OD_Murcko",
    use_trajectory="true", max_conformers=10,   # screening: fewer conformers ~ faster EGNN
    model_arch="v1", augment_delta_descriptors="false",
    hidden_dim=128, conformer_layers=3,
    sequence_backend="learned",          # fast screening backbone
    head_hidden=256, dropout=0.1,
    lambda_chameleonic=0.1, lambda_triplet=0.0, lambda_residue_psa=0.0,
    pampa_baseline=-8.0,
    batch_size=16, num_workers=8, epochs=50,
    lr=5e-4, weight_decay=1e-5, warmup_epochs=5, grad_clip=1.0,
    early_stop_patience=10,
    seed=42, device="cuda",
    wandb_project="chameleonnet-campaign", wandb_mode="offline",
)
VOCAB = [f"{ROOT}/eda/water_residue_vocab.csv", f"{ROOT}/eda/hexane_residue_vocab.csv"]

# name -> overrides
CANDS = {
    "c0_baseline":            {},
    "c1_fulldesc":            dict(use_full_descriptors="true", augment_delta_descriptors="true"),
    "c2_fulldesc_moddrop":    dict(use_full_descriptors="true", augment_delta_descriptors="true", modality_dropout=0.3),
    "c3_fulldesc_physres":    dict(use_full_descriptors="true", augment_delta_descriptors="true", physics_residual="true", lambda_resid_l2=0.01),
    "c4_fulldesc_ib":         dict(use_full_descriptors="true", augment_delta_descriptors="true", info_bottleneck="true", lambda_ib=1e-3),
    "c5_fulldesc_reg":        dict(use_full_descriptors="true", augment_delta_descriptors="true", dropout=0.3, weight_decay=1e-3, hidden_dim=96),
    "c6_extdesc":             dict(use_extended_descriptors="true", augment_delta_descriptors="true"),
    "c7_fulldesc_physres_moddrop": dict(use_full_descriptors="true", augment_delta_descriptors="true", physics_residual="true", lambda_resid_l2=0.01, modality_dropout=0.3),
    "c8_fulldesc_physres_reg": dict(use_full_descriptors="true", augment_delta_descriptors="true", physics_residual="true", lambda_resid_l2=0.01, dropout=0.3, weight_decay=1e-3),
}

def fmt(v):
    # Avoid bare-exponent floats (e.g. 1e-05) — PyYAML parses them as strings.
    if isinstance(v, float):
        s = ("%.12f" % v).rstrip("0").rstrip(".")
        return s if s else "0"
    return v


def emit(name, ov):
    cfg = dict(BASE); cfg.update(ov)
    cfg["output_dir"] = f"{ROOT}/ChameleonNet/runs/campaign/{name}"
    cfg["wandb_run_name"] = name
    lines = []
    for k, v in cfg.items():
        lines.append(f"{k}: {fmt(v)}")
    lines.append("vocab_csvs:")
    for v in VOCAB:
        lines.append(f"  - {v}")
    lines.append("eval_schemes:")
    lines.append("  - OD_Murcko")
    path = f"{CFGDIR}/{name}.yaml"
    open(path, "w").write("\n".join(lines) + "\n")
    return path

for name, ov in CANDS.items():
    print(emit(name, ov))

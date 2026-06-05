"""Round-2 configs: build on the Round-1 winner (extended 28 descriptors)."""
import os
ROOT = "/hdd0/sohyun/cyclic-peptide-permeability"
CFGDIR = f"{ROOT}/ChameleonNet/configs/campaign"
GBR = f"{ROOT}/ChameleonNet/runs/_logs/gbr_preds_OD_Murcko.json"

BASE = dict(
    csv_path=f"{ROOT}/data/CycPeptMPDB-4D_with_assay_descriptors_preprocessed.csv",
    pdb_root=f"{ROOT}/CycPeptMPDB-4D", splits_dir=f"{ROOT}/splits",
    cache_dir=f"{ROOT}/ChameleonNet/.cache_pdb",
    split_scheme="OD_Murcko", use_trajectory="true", max_conformers=10,
    model_arch="v1",
    use_extended_descriptors="true", augment_delta_descriptors="true",
    hidden_dim=128, conformer_layers=3, sequence_backend="learned",
    head_hidden=256, dropout=0.1,
    lambda_chameleonic=0.1, lambda_triplet=0.0, lambda_residue_psa=0.0, pampa_baseline=-8.0,
    batch_size=16, num_workers=8, epochs=50, lr=5e-4, weight_decay=1e-5,
    warmup_epochs=5, grad_clip=1.0, early_stop_patience=10,
    seed=42, device="cuda", wandb_project="chameleonnet-campaign", wandb_mode="offline",
)
VOCAB = [f"{ROOT}/eda/water_residue_vocab.csv", f"{ROOT}/eda/hexane_residue_vocab.csv"]

CANDS = {
    "e1_extdesc":          {},                                                          # control (= c6, 50ep)
    "e2_extdesc_gbrres":   dict(gbr_residual="true", gbr_preds_path=GBR, lambda_resid_l2=0.01),
    "e3_extdesc_physres":  dict(physics_residual="true", lambda_resid_l2=0.1),
    "e4_extdesc_moddrop":  dict(modality_dropout=0.15),
    "e5_extdesc_distill":  dict(lambda_distill=0.2, gbr_preds_path=GBR),
    "e6_extdesc_big":      dict(hidden_dim=160, head_hidden=320, dropout=0.15, weight_decay=5e-5),
}

def fmt(v):
    if isinstance(v, float):
        s = ("%.12f" % v).rstrip("0").rstrip("."); return s or "0"
    return v

def emit(name, ov):
    cfg = dict(BASE); cfg.update(ov)
    cfg["output_dir"] = f"{ROOT}/ChameleonNet/runs/campaign/{name}"
    cfg["wandb_run_name"] = name
    lines = [f"{k}: {fmt(v)}" for k, v in cfg.items()]
    lines += ["vocab_csvs:"] + [f"  - {v}" for v in VOCAB] + ["eval_schemes:", "  - OD_Murcko"]
    open(f"{CFGDIR}/{name}.yaml", "w").write("\n".join(lines) + "\n")
    print(f"{CFGDIR}/{name}.yaml")

for n, o in CANDS.items():
    emit(n, o)

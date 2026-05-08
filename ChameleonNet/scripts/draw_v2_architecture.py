"""Render the ChameleonNet V2 architecture as a PNG.

Boxes correspond to nn.Modules in chameleonnet/models/chameleonnet_v2.py.
Tensor-shape annotations match the actual code (F=hidden_dim=128 by default,
H=head_hidden=256, B=batch, K=conformers/sample, R=residues, N=atoms).

Usage:
    python scripts/draw_v2_architecture.py --out runs/v2_architecture.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COLORS = {
    "input":   "#cfe2f3",
    "encoder": "#fff2cc",
    "pool":    "#fce5cd",
    "fuse":    "#d9ead3",
    "head":    "#f4cccc",
    "loss":    "#ead1dc",
    "shared":  "#e6e6e6",
}


def box(ax, x, y, w, h, text, color, fontsize=9, weight="normal"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.0, edgecolor="#333", facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=fontsize, weight=weight)


def arrow(ax, x1, y1, x2, y2, color="#444", lw=1.0, style="-|>", text=None,
          text_offset=(0.0, 0.08), fontsize=8):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=12,
        color=color, linewidth=lw,
    ))
    if text:
        mx, my = (x1 + x2) / 2 + text_offset[0], (y1 + y2) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center",
                fontsize=fontsize, style="italic", color="#222")


def render(out_path: Path) -> None:
    W, H = 18, 13
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    ax.text(W / 2, H - 0.3,
            "ChameleonNet V2 — residue-resolved chameleonic dual-environment model",
            ha="center", fontsize=14, weight="bold")
    ax.text(W / 2, H - 0.65,
            "B = batch · K = conformers/sample · N = atoms · R = residues · "
            "F = hidden_dim = 128 · H = head_hidden = 256",
            ha="center", fontsize=9, style="italic", color="#444")

    # ---------- Inputs (top) ----------
    box(ax, 0.5,  11.2, 4.0, 0.7,
        "Water conformer ensemble\n(K_w, N, 3) coords + atom z + res_pos",
        COLORS["input"], 9, "bold")
    box(ax, 5.0,  11.2, 4.0, 0.7,
        "Hexane conformer ensemble\n(K_h, N, 3) coords + atom z + res_pos",
        COLORS["input"], 9, "bold")
    box(ax, 9.5,  11.2, 3.7, 0.7,
        "Sequence (residue-name list)\nSMILES string",
        COLORS["input"], 9, "bold")
    box(ax, 13.7, 11.2, 4.0, 0.7,
        "Descriptors (4D)\n11 raw  →  +10 Δ-features  =  21d",
        COLORS["input"], 9, "bold")

    # ---------- Shared 3D EGNN encoder ----------
    box(ax, 1.5, 9.7, 7.0, 0.9,
        "Shared 3D EGNN encoder (ConformerEncoder)\n"
        "atom_embed + res_embed → input_mix → 3 × _EGNNLayer → LayerNorm   "
        "→  h_atom ∈ ℝ^(M, N, F)",
        COLORS["shared"], 9, "bold")

    # ---------- Per-environment branches ----------
    # Water column
    box(ax, 0.5, 8.1, 4.0, 0.6,
        "mean over atoms  →  h_conf_w ∈ ℝ^(K_w, F)",
        COLORS["pool"])
    box(ax, 0.5, 7.1, 4.0, 0.6,
        "ConformerAttentionPool  →  h_water ∈ ℝ^(B, F)",
        COLORS["pool"])
    box(ax, 0.5, 6.0, 4.0, 0.7,
        "AtomToResiduePool (mean + attention over atoms,\n"
        "per (conformer, residue))",
        COLORS["pool"])
    box(ax, 0.5, 4.95, 4.0, 0.6,
        "pool_residues_over_conformers  →  h_water_res ∈ ℝ^(B, R, F)",
        COLORS["pool"])

    # Hexane column
    box(ax, 5.0, 8.1, 4.0, 0.6,
        "mean over atoms  →  h_conf_h ∈ ℝ^(K_h, F)",
        COLORS["pool"])
    box(ax, 5.0, 7.1, 4.0, 0.6,
        "ConformerAttentionPool  →  h_hexane ∈ ℝ^(B, F)",
        COLORS["pool"])
    box(ax, 5.0, 6.0, 4.0, 0.7,
        "AtomToResiduePool   (shared atom_to_residue,\n"
        "weights tied across envs)",
        COLORS["pool"])
    box(ax, 5.0, 4.95, 4.0, 0.6,
        "pool_residues_over_conformers  →  h_hexane_res ∈ ℝ^(B, R, F)",
        COLORS["pool"])

    # ---------- Sequence / descriptor branches ----------
    box(ax, 9.5, 9.0, 3.7, 0.9,
        "SequenceEncoder (peptideclm)\n"
        "AutoModel(peptideclm-2-mlm-base) →\n"
        "mean-pool tokens → Linear(768 → F)",
        COLORS["encoder"])
    box(ax, 9.5, 7.9, 3.7, 0.6,
        "h_seq ∈ ℝ^(B, F)", COLORS["pool"])

    box(ax, 13.7, 9.0, 4.0, 0.9,
        "DescriptorScaler (train-fit μ/σ)\n"
        "→ DescriptorMLP\n"
        "Linear(21 → F) → SiLU → Linear(F → F) → SiLU",
        COLORS["encoder"])
    box(ax, 13.7, 7.9, 4.0, 0.6,
        "h_desc ∈ ℝ^(B, F)", COLORS["pool"])

    # ---------- Chameleonic Δ branches (center) ----------
    box(ax, 9.5, 6.4, 3.7, 0.7,
        "h_diff = h_water − h_hexane\n∈ ℝ^(B, F)   (V1-style global Δ)",
        COLORS["fuse"])

    box(ax, 9.5, 4.5, 3.7, 1.6,
        "ResidueChameleonHead\n\n"
        "Δ_res = h_water_res − h_hexane_res\n"
        "score = MLP([h_w | h_h | Δ])  ∈ ℝ^(B, R)\n"
        "weight = softmax_R(score)\n"
        "h_chameleon = Σ_R weight · Δ_res  ∈ ℝ^(B, F)",
        COLORS["fuse"], 8.5)

    box(ax, 13.7, 4.5, 4.0, 1.6,
        "Auxiliary tensors carried to the loss\n\n"
        "delta_residue ∈ ℝ^(B, R, F)\n"
        "chameleon_weight ∈ ℝ^(B, R)\n"
        "res_mask ∈ {0,1}^(B, R)",
        COLORS["head"], 8.5)

    # ---------- Fusion ----------
    box(ax, 4.5, 3.4, 9.0, 0.8,
        "Fused = [ h_water  |  h_hexane  |  h_diff  |  h_chameleon  |  h_seq  |  h_desc ]"
        "    ∈ ℝ^(B, 6F = 768)",
        COLORS["fuse"], 10, "bold")

    # ---------- Heads ----------
    box(ax, 1.5, 2.0, 5.5, 1.0,
        "Main head\n"
        "Linear(6F → H) → SiLU → Drop → Linear(H → H) → SiLU → Drop → Linear(H → 1)\n"
        "→  pampa_pred ∈ ℝ^(B,)",
        COLORS["head"], 9, "bold")
    box(ax, 7.5, 2.0, 4.0, 1.0,
        "Auxiliary head\n"
        "Linear(F → F) → SiLU → Linear(F → 1)\n"
        "→  chameleonic_pred (aux)",
        COLORS["head"], 9, "bold")
    box(ax, 12.0, 2.0, 5.7, 1.0,
        "Triplet branch\n"
        "fused_embedding = [h_water | h_hexane | h_diff]\n"
        "passed to Tanimoto triplet mining (RDKit Morgan FP)",
        COLORS["head"], 9, "bold")

    # ---------- Loss ----------
    box(ax, 0.5, 0.4, 17.2, 1.2,
        "CompositeLoss\n"
        "= MSE(pampa_pred, target)\n"
        " + λ_cham · chameleonic_loss(h_diff, chameleonic_pred, pampa)"
        "   + λ_trip · TanimotoTripletLoss(fused_emb, smiles, pampa)"
        "   + λ_psa  · ResidueChameleonLoss(delta_residue, |ΔPSA|)",
        COLORS["loss"], 9, "bold")

    # ---------- Arrows ----------
    # Inputs → encoder
    arrow(ax, 2.5, 11.2, 3.5, 10.6)
    arrow(ax, 7.0, 11.2, 6.5, 10.6)

    # Encoder → per-env mean (top of each column)
    arrow(ax, 3.5, 9.7, 2.5, 8.7)
    arrow(ax, 6.5, 9.7, 7.0, 8.7)

    # Vertical chains in each column
    for x in (2.5, 7.0):
        arrow(ax, x, 8.1, x, 7.7)   # mean → AttentionPool
        arrow(ax, x, 7.1, x, 6.7)   # AttentionPool → AtomToResiduePool
        arrow(ax, x, 6.0, x, 5.55)  # AtomToResiduePool → pool_residues

    # Encoder also feeds AtomToResiduePool directly (atom-level features pre-pool)
    arrow(ax, 3.5, 9.7, 2.5, 6.7, color="#888", lw=0.8)
    arrow(ax, 6.5, 9.7, 7.0, 6.7, color="#888", lw=0.8)

    # h_water / h_hexane → h_diff
    arrow(ax, 4.5, 7.4, 9.5, 6.75)
    arrow(ax, 9.0, 7.4, 9.5, 6.75)

    # h_water_res / h_hexane_res → ResidueChameleonHead
    arrow(ax, 4.5, 5.25, 9.5, 5.3)
    arrow(ax, 9.0, 5.25, 9.5, 5.3)

    # ResidueChameleonHead → aux tensors panel
    arrow(ax, 13.2, 5.3, 13.7, 5.3)

    # Sequence/descriptor branches
    arrow(ax, 11.35, 11.2, 11.35, 9.9)
    arrow(ax, 15.7,  11.2, 15.7,  9.9)
    arrow(ax, 11.35,  9.0, 11.35, 8.5)
    arrow(ax, 15.7,   9.0, 15.7,  8.5)

    # Everything → Fused
    arrow(ax, 2.5, 7.1, 6.5, 4.2, color="#666")
    arrow(ax, 7.0, 7.1, 8.0, 4.2, color="#666")
    arrow(ax, 11.35, 6.4, 9.5, 4.2)
    arrow(ax, 11.35, 4.5, 10.0, 4.2)
    arrow(ax, 11.35, 7.9, 11.0, 4.2, color="#666")
    arrow(ax, 15.7,  7.9, 13.0, 4.2, color="#666")

    # Fused → heads
    arrow(ax, 6.0,  3.4, 4.0, 3.0)
    arrow(ax, 9.0,  3.4, 9.5, 3.0)
    arrow(ax, 12.0, 3.4, 14.5, 3.0, color="#666")

    # Heads → loss
    arrow(ax, 4.0,  2.0, 4.0, 1.6)
    arrow(ax, 9.5,  2.0, 9.5, 1.6)
    arrow(ax, 14.5, 2.0, 14.5, 1.6)
    # Aux tensors → loss directly
    arrow(ax, 15.7, 4.5, 15.7, 1.6, color="#888", lw=0.8)

    # ---------- Legend ----------
    legend_x, legend_y = 0.5, H - 1.05
    legend = [
        ("input/target", COLORS["input"]),
        ("encoder",      COLORS["encoder"]),
        ("shared trunk", COLORS["shared"]),
        ("pool / Δ",     COLORS["pool"]),
        ("fusion",       COLORS["fuse"]),
        ("head",         COLORS["head"]),
        ("loss",         COLORS["loss"]),
    ]
    for i, (label, color) in enumerate(legend):
        ax.add_patch(patches.Rectangle(
            (legend_x + i * 1.95, legend_y), 0.35, 0.22,
            facecolor=color, edgecolor="#333", linewidth=0.6,
        ))
        ax.text(legend_x + i * 1.95 + 0.45, legend_y + 0.11, label,
                fontsize=8.5, va="center")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="runs/v2_architecture.png")
    args = p.parse_args()
    render(Path(args.out))


if __name__ == "__main__":
    main()

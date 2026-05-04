"""
Conformational analysis of chameleonic cyclic peptide CycPeptMPDB_ID 6264
(2021_Kelly L1_2.1.3.2.4.1, PAMPA = -5.02, 10-mer cyclic peptide).

Goals:
  1) How different is the peptide between water and hexane? (cross-solvent)
  2) How variable is it within water?  Within hexane? (per-solvent)

Metrics:
  - per-frame backbone RMSD vs each medoid
  - radius of gyration (Rg)
  - intramolecular hydrogen bond count
  - solvent-accessible polar area proxy: backbone NH/CO buried fraction
  - 2D PCA over backbone phi/psi dihedrals (combined fit)
  - per-solvent clustering (hierarchical) of backbone RMSD matrix
"""
import os, warnings, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.dihedrals import Dihedral
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA

ROOT = "/ssd0/sohyun/cyclic_peptide/cyclic_peptide_permeability/CycPeptMPDB-4D"
OUT  = "/ssd0/sohyun/cyclic_peptide/cyclic_peptide_permeability/eda/chameleonic_6264"
PEP  = "2021_Kelly_6264"

water_traj  = f"{ROOT}/Water/Trajectories/{PEP}_H2O_Traj.pdb"
hexane_traj = f"{ROOT}/Hexane/Trajectories/{PEP}_Hexane_Traj.pdb"

def load(top_or_traj):
    # The trajectory PDB is a multi-MODEL file containing only the peptide.
    u = mda.Universe(top_or_traj, top_or_traj)
    return u

uW = load(water_traj)
uH = load(hexane_traj)

print(f"Water:  {len(uW.trajectory)} frames, {uW.atoms.n_atoms} atoms, {uW.residues.n_residues} residues")
print(f"Hexane: {len(uH.trajectory)} frames, {uH.atoms.n_atoms} atoms, {uH.residues.n_residues} residues")
print("Residues (water):", [r.resname for r in uW.residues])

# Backbone selection — cyclic peptide may have non-standard residues; use name CA + N + C
sel_bb = "name CA or name N or name C"
bbW = uW.select_atoms(sel_bb)
bbH = uH.select_atoms(sel_bb)
assert bbW.n_atoms == bbH.n_atoms, "Backbone atom count mismatch"
n_bb = bbW.n_atoms
print(f"Backbone atoms: {n_bb}")

# ---------- gather per-frame backbone coordinates ----------
def gather_bb_coords(u, sel):
    grp = u.select_atoms(sel)
    coords = np.zeros((len(u.trajectory), grp.n_atoms, 3))
    for i, ts in enumerate(u.trajectory):
        coords[i] = grp.positions
    return coords

cW = gather_bb_coords(uW, sel_bb)
cH = gather_bb_coords(uH, sel_bb)

# Center every frame
def center(coords):
    return coords - coords.mean(axis=1, keepdims=True)
cW_c = center(cW)
cH_c = center(cH)

# Kabsch alignment of every frame to a single reference (water frame 0) so RMSDs are comparable.
def kabsch(P, Q):
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R

ref = cW_c[0]
def align_all(coords):
    out = np.empty_like(coords)
    for i in range(coords.shape[0]):
        R = kabsch(coords[i], ref)
        out[i] = coords[i] @ R.T
    return out
cWa = align_all(cW_c)
cHa = align_all(cH_c)

# ---------- combined RMSD matrix ----------
all_c = np.concatenate([cWa, cHa], axis=0)  # (200, n_bb, 3)
nW = cWa.shape[0]; nH = cHa.shape[0]
N = nW + nH
rmsd_mat = np.zeros((N, N))
# pairwise RMSD with optimal Kabsch on each pair
for i in range(N):
    for j in range(i+1, N):
        R = kabsch(all_c[j], all_c[i])
        d = all_c[j] @ R.T - all_c[i]
        r = np.sqrt((d**2).sum() / d.shape[0])
        rmsd_mat[i, j] = rmsd_mat[j, i] = r
np.save(f"{OUT}/rmsd_matrix.npy", rmsd_mat)

print("\n--- BACKBONE RMSD STATS (Å) ---")
WW = rmsd_mat[:nW, :nW]; WH = rmsd_mat[:nW, nW:]; HH = rmsd_mat[nW:, nW:]
def stat(name, M, exclude_diag=False):
    if exclude_diag:
        iu = np.triu_indices_from(M, k=1)
        v = M[iu]
    else:
        v = M.ravel()
    print(f"{name:18s} n={len(v):6d}  mean={v.mean():.2f}  median={np.median(v):.2f}  max={v.max():.2f}")
stat("water-water",  WW, exclude_diag=True)
stat("hexane-hexane",HH, exclude_diag=True)
stat("water-hexane", WH)

# ---------- medoid ----------
def medoid(M):
    s = M.sum(axis=1)
    return int(np.argmin(s))
mW = medoid(WW)
mH = medoid(HH)
medW_global = mW
medH_global = nW + mH
cross_med_rmsd = rmsd_mat[medW_global, medH_global]
print(f"\nWater  medoid frame: {mW}, hexane medoid frame: {mH}")
print(f"RMSD(water medoid, hexane medoid) = {cross_med_rmsd:.2f} Å")

# ---------- radius of gyration & intramolecular hbonds ----------
def rg_traj(u, sel="protein or name * "):
    # all heavy atoms
    grp = u.select_atoms("not name H*")
    out = np.zeros(len(u.trajectory))
    for i, ts in enumerate(u.trajectory):
        out[i] = grp.radius_of_gyration()
    return out
rgW = rg_traj(uW); rgH = rg_traj(uH)

# intramolecular H-bonds: distance + angle test directly (no topology bonds needed).
# Donor heavy = backbone/side-chain N or O-H oxygen; we use amide N-H here (peptide bonds).
def hb_traj(u):
    # All amide hydrogens (named H, HN, H1...) bonded to a nitrogen by virtue of name.
    # In these PDBs amide hydrogen is "H" attached to backbone N. Use within-residue pairing:
    # Find for each residue: N atom and the H attached to it (name H or HN).
    NH_pairs = []  # list of (N_index, H_index)
    for res in u.residues:
        atoms = {a.name: a for a in res.atoms}
        if "N" in atoms:
            for hn in ("H", "HN", "H1"):
                if hn in atoms:
                    NH_pairs.append((atoms["N"].index, atoms[hn].index))
                    break
    acceptors = u.select_atoms("name O or name OXT or name OD* or name OE* or name OG* or name OH")
    counts = np.zeros(len(u.trajectory))
    for fi, ts in enumerate(u.trajectory):
        n_hb = 0
        for n_idx, h_idx in NH_pairs:
            n_pos = u.atoms[n_idx].position
            h_pos = u.atoms[h_idx].position
            for o in acceptors:
                # skip same-residue O of the amide (not a real H-bond)
                if o.resindex == u.atoms[n_idx].resindex:
                    continue
                o_pos = o.position
                d_HA = np.linalg.norm(o_pos - h_pos)
                if d_HA > 2.5: continue
                # angle N-H...O
                v1 = h_pos - n_pos; v2 = o_pos - h_pos
                cosang = (v1*v2).sum() / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-9)
                ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
                if ang < 50:  # supplementary; means N-H...O angle > 130
                    n_hb += 1
        counts[fi] = n_hb
    return counts
hbW = hb_traj(uW); hbH = hb_traj(uH)

print(f"\nRg (Å)        water: {rgW.mean():.2f} ± {rgW.std():.2f}   hexane: {rgH.mean():.2f} ± {rgH.std():.2f}")
print(f"Intramol H-bonds  water: {hbW.mean():.2f} ± {hbW.std():.2f}   hexane: {hbH.mean():.2f} ± {hbH.std():.2f}")

# ---------- backbone phi/psi dihedrals + PCA ----------
def collect_phipsi(u):
    phis, psis = [], []
    for res in u.residues:
        ph = res.phi_selection()
        ps = res.psi_selection()
        if ph is not None: phis.append(ph)
        if ps is not None: psis.append(ps)
    angles = []
    for atomgrp in phis + psis:
        d = Dihedral([atomgrp]).run()
        angles.append(d.results.angles[:,0])
    return np.array(angles).T  # (frames, n_dihedrals)

phipsiW = collect_phipsi(uW)
phipsiH = collect_phipsi(uH)
print(f"phi/psi shape water/hexane: {phipsiW.shape} {phipsiH.shape}")

# encode as sin/cos to handle wrap
def sc(a):
    rad = np.deg2rad(a)
    return np.concatenate([np.sin(rad), np.cos(rad)], axis=1)
X = np.vstack([sc(phipsiW), sc(phipsiH)])
pca = PCA(n_components=3).fit(X)
emb = pca.transform(X)
print("PCA explained variance:", pca.explained_variance_ratio_[:3])
embW = emb[:nW]; embH = emb[nW:]

# ---------- hierarchical clustering per solvent ----------
def cluster(M, cutoff_A=1.0):
    cond = squareform(M, checks=False)
    Z = linkage(cond, method="average")
    labels = fcluster(Z, t=cutoff_A, criterion="distance")
    return labels
labW = cluster(WW, 1.0)
labH = cluster(HH, 1.0)
print(f"\n# clusters (RMSD<=1.0 Å, average linkage)  water:{labW.max()}  hexane:{labH.max()}")
def hist(lab):
    _, c = np.unique(lab, return_counts=True)
    return sorted(c, reverse=True)
print("Water cluster sizes:", hist(labW))
print("Hexane cluster sizes:", hist(labH))

# ---------- save metrics CSV ----------
df = pd.DataFrame({
    "frame": np.arange(N),
    "solvent": ["water"]*nW + ["hexane"]*nH,
    "rg":   np.concatenate([rgW, rgH]),
    "hbonds": np.concatenate([hbW, hbH]),
    "rmsd_to_water_medoid":  rmsd_mat[medW_global],
    "rmsd_to_hexane_medoid": rmsd_mat[medH_global],
    "pc1": emb[:,0], "pc2": emb[:,1], "pc3": emb[:,2],
    "cluster": np.concatenate([labW, labH + labW.max()]),
})
df.to_csv(f"{OUT}/per_frame_metrics.csv", index=False)
print(f"\nSaved per-frame metrics to {OUT}/per_frame_metrics.csv")

# ---------- plots ----------
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

ax = axes[0,0]
im = ax.imshow(rmsd_mat, cmap="viridis", origin="lower")
ax.axhline(nW-0.5, color="white", lw=0.8); ax.axvline(nW-0.5, color="white", lw=0.8)
ax.set_title("Backbone RMSD matrix\n(left/bottom: water; right/top: hexane)")
ax.set_xlabel("frame"); ax.set_ylabel("frame")
plt.colorbar(im, ax=ax, label="RMSD (Å)")

ax = axes[0,1]
ax.hist(WW[np.triu_indices_from(WW,1)], bins=30, alpha=0.6, label=f"water-water (μ={WW[np.triu_indices_from(WW,1)].mean():.2f})", color="steelblue", density=True)
ax.hist(HH[np.triu_indices_from(HH,1)], bins=30, alpha=0.6, label=f"hexane-hexane (μ={HH[np.triu_indices_from(HH,1)].mean():.2f})", color="darkorange", density=True)
ax.hist(WH.ravel(), bins=30, alpha=0.4, label=f"water-hexane (μ={WH.mean():.2f})", color="firebrick", density=True)
ax.set_xlabel("backbone RMSD (Å)"); ax.set_ylabel("density")
ax.set_title("Pairwise RMSD distributions")
ax.legend(fontsize=8)

ax = axes[0,2]
ax.scatter(embW[:,0], embW[:,1], s=24, alpha=0.7, color="steelblue", label=f"water (n={nW})")
ax.scatter(embH[:,0], embH[:,1], s=24, alpha=0.7, color="darkorange", label=f"hexane (n={nH})")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
ax.set_title("PCA over backbone (sin/cos φ,ψ)")
ax.legend()

ax = axes[1,0]
ax.hist(rgW, bins=20, alpha=0.6, color="steelblue", label=f"water  μ={rgW.mean():.2f}")
ax.hist(rgH, bins=20, alpha=0.6, color="darkorange", label=f"hexane μ={rgH.mean():.2f}")
ax.set_xlabel("Radius of gyration (Å)"); ax.set_ylabel("count")
ax.set_title("Compactness")
ax.legend()

ax = axes[1,1]
bins = np.arange(0, max(hbW.max(), hbH.max())+2)-0.5
ax.hist(hbW, bins=bins, alpha=0.6, color="steelblue", label=f"water μ={hbW.mean():.2f}")
ax.hist(hbH, bins=bins, alpha=0.6, color="darkorange", label=f"hexane μ={hbH.mean():.2f}")
ax.set_xlabel("# intramolecular H-bonds")
ax.set_title("Self-shielding (intramolecular HB)")
ax.legend()

ax = axes[1,2]
ax.plot(rmsd_mat[medW_global][:nW], color="steelblue", label="water frames vs water medoid")
ax.plot(np.arange(nW, N), rmsd_mat[medW_global][nW:], color="steelblue", ls="--", alpha=0.6, label="hexane frames vs water medoid")
ax.plot(rmsd_mat[medH_global][:nW], color="darkorange", ls="--", alpha=0.6, label="water frames vs hexane medoid")
ax.plot(np.arange(nW, N), rmsd_mat[medH_global][nW:], color="darkorange", label="hexane frames vs hexane medoid")
ax.axvline(nW-0.5, color="k", lw=0.5)
ax.set_xlabel("frame index (water | hexane)"); ax.set_ylabel("backbone RMSD (Å)")
ax.set_title("Per-frame RMSD to medoids")
ax.legend(fontsize=7)

plt.suptitle("Chameleonic peptide CycPeptMPDB_ID 6264 (10-mer, PAMPA=-5.02)\n"
             f"Cross-medoid RMSD water↔hexane = {cross_med_rmsd:.2f} Å", fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUT}/conformation_summary.png", dpi=140)
print(f"Saved plot: {OUT}/conformation_summary.png")

# ---------- save numeric summary ----------
summary = {
    "peptide_id": 6264,
    "name": "L1_2.1.3.2.4.1",
    "PAMPA": -5.02,
    "n_residues": int(uW.residues.n_residues),
    "n_backbone_atoms": int(n_bb),
    "rmsd_water_water_mean": float(WW[np.triu_indices_from(WW,1)].mean()),
    "rmsd_hexane_hexane_mean": float(HH[np.triu_indices_from(HH,1)].mean()),
    "rmsd_water_hexane_mean": float(WH.mean()),
    "rmsd_water_medoid_to_hexane_medoid": float(cross_med_rmsd),
    "rg_water_mean":  float(rgW.mean()), "rg_water_std":  float(rgW.std()),
    "rg_hexane_mean": float(rgH.mean()), "rg_hexane_std": float(rgH.std()),
    "hb_water_mean":  float(hbW.mean()), "hb_water_std":  float(hbW.std()),
    "hb_hexane_mean": float(hbH.mean()), "hb_hexane_std": float(hbH.std()),
    "n_clusters_water_1A":  int(labW.max()),
    "n_clusters_hexane_1A": int(labH.max()),
    "water_cluster_sizes":  [int(x) for x in hist(labW)],
    "hexane_cluster_sizes": [int(x) for x in hist(labH)],
    "pca_explained_variance_top3": pca.explained_variance_ratio_[:3].tolist(),
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

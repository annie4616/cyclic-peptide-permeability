"""Conformer-ENERGY-regression SSL pretraining of ConformerEncoder on CREMP.

Our ConformerEncoder is rotation/translation-INVARIANT (distance-based, SchNet-
like). Coordinate-noise-vector denoising is ill-posed for an invariant net (the
target is equivariant; a flat loss confirmed it), so we use an invariant-friendly
objective: from the pooled conformer embedding, regress the conformer's RELATIVE
ENERGY (kcal/mol, from CREMP). This teaches a force-field-like potential — a
physically meaningful 3D representation — that transfers into ChameleonNet's 3D
encoder. Energies are standardized per molecule so the encoder learns
within-molecule geometry->energy structure, not absolute scale.

Saves the encoder state_dict to <out>. GPU from CUDA_VISIBLE_DEVICES.
"""
from __future__ import annotations
import glob, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from chameleonnet.models.conformer_encoder import ConformerEncoder
from chameleonnet.data.residue_vocab import ResidueVocab

ROOT = "/hdd0/sohyun/cyclic-peptide-permeability"
CACHE = f"{ROOT}/ChameleonNet/.cache_cremp"
OUT = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/ChameleonNet/runs/_logs/cremp_encoder.pt"
EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
HIDDEN = 128
LAYERS = 3
device = "cuda" if torch.cuda.is_available() else "cpu"


class CrempConfs(Dataset):
    """One sample = one (molecule, conformer). Energy standardized per molecule."""
    def __init__(self):
        self.items = []
        for f in sorted(glob.glob(f"{CACHE}/*.npz")):
            d = np.load(f)
            e = np.asarray(d["relE"], np.float32)
            mu, sd = np.nanmean(e), np.nanstd(e)
            ez = (e - mu) / (sd + 1e-6)
            ez = np.nan_to_num(ez, nan=0.0)
            self.items.append((d["coords"], d["z"], d["res"], ez))
        self.index = [(mi, ci) for mi, (c, *_ ) in enumerate(self.items) for ci in range(c.shape[0])]
    def __len__(self): return len(self.index)
    def __getitem__(self, i):
        mi, ci = self.index[i]
        coords, z, res, ez = self.items[mi]
        return coords[ci], z, res, np.float32(ez[ci])


def collate(batch):
    Nmax = max(c.shape[0] for c, *_ in batch); B = len(batch)
    coords = torch.zeros(B, Nmax, 3); z = torch.zeros(B, Nmax, dtype=torch.long)
    res = torch.zeros(B, Nmax, dtype=torch.long); pad = torch.ones(B, Nmax, dtype=torch.bool)
    y = torch.zeros(B)
    for i, (c, zz, rr, ee) in enumerate(batch):
        n = c.shape[0]
        coords[i, :n] = torch.from_numpy(np.asarray(c, np.float32))
        z[i, :n] = torch.from_numpy(np.asarray(zz, np.int64))
        res[i, :n] = torch.from_numpy(np.asarray(rr, np.int64))
        pad[i, :n] = False; y[i] = float(ee)
    return coords, z, res, pad, y


class EnergyReg(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.head = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 1))
    def forward(self, coords, z, res, pad):
        pooled = self.enc(coords=coords, z=z, res=res, pad_mask=pad)  # (B, hidden), invariant
        return self.head(pooled).squeeze(-1)


def main():
    vocab = ResidueVocab.from_csvs(f"{ROOT}/eda/water_residue_vocab.csv", f"{ROOT}/eda/hexane_residue_vocab.csv")
    enc = ConformerEncoder(hidden_dim=HIDDEN, num_layers=LAYERS, residue_vocab_size=len(vocab),
                           residue_vocab_tokens=list(vocab._tokens))
    model = EnergyReg(enc).to(device)
    ds = CrempConfs()
    print(f"[pretrain] {len(ds)} conformers, device={device}, epochs={EPOCHS}, objective=relE-regression", flush=True)
    loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=8, collate_fn=collate, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0.0; nb = 0
        for coords, z, res, pad, y in loader:
            coords, z, res, pad, y = (coords.to(device), z.to(device), res.to(device),
                                      pad.to(device), y.to(device))
            pred = model(coords, z, res, pad)
            loss = nn.functional.mse_loss(pred, y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss); nb += 1
        print(f"  ep{ep} energy_mse={tot/max(1,nb):.4f} ({time.time()-t0:.0f}s)", flush=True)
    torch.save({"encoder": enc.state_dict(), "hidden_dim": HIDDEN, "num_layers": LAYERS,
                "vocab_size": len(vocab)}, OUT)
    print(f"[pretrain] saved encoder -> {OUT}", flush=True)


if __name__ == "__main__":
    main()

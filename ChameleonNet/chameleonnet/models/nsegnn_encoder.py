"""NS-EGNN conformer encoder (Non-Stationary EGNN, MaojiWEN/NS-EGNN, NeurIPS'25).

Faithful adaptation of NS-EGNN's two ingredients to our setting:
  1. E_GCL  — the equivariant EGNN layer (Satorras 2021) used in the repo:
       radial = ||x_i - x_j||^2 (rotation/translation invariant)
       m_ij   = edge_mlp([h_i, h_j, radial]) (optional sigmoid attention)
       x_i   += coords_weight * mean_j ( (x_i-x_j) * coord_mlp(m_ij) )   (equivariant)
       h_i    = h_i + node_mlp([h_i, sum_j m_ij])                         (recurrent)
     with norm_diff (unit (x_i-x_j)) and clamp, exactly as the repo.
  2. Non-Stationary spectral features — our K conformers come from a temporally
     ordered trajectory, so each atom has a length-K xyz trajectory. We take a
     multi-scale STFT magnitude of that trajectory (RMS over x/y/z), pooled over
     STFT time-frames, and inject it (Linear -> hidden//2) into the node feature
     before message passing. This is NS-EGNN's "non-stationary" idea applied
     along the conformer/trajectory axis. (Repo hops were tuned for long MD
     trajectories; we scale window sizes to the available K frames.)

Output: per-molecule (B, hidden) env embedding — replaces the per-conformer
encode + AttentionPool path for this arch. Atoms within a molecule are treated
as a fully-connected graph (pad atoms masked).
"""
from __future__ import annotations
from typing import List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EGCLDense(nn.Module):
    """E_GCL over a per-molecule dense graph with a leading frame axis.

    h:    (T, N, F)    coord: (T, N, 3)    valid: (N,) bool (True = real atom)
    Messages are computed per frame; the graph (edges) is shared across frames.
    """

    def __init__(self, hidden_nf: int, act=nn.SiLU(), coords_weight: float = 1.0,
                 attention: bool = True, clamp: bool = True, norm_diff: bool = True):
        super().__init__()
        self.coords_weight = coords_weight
        self.attention = attention
        self.clamp = clamp
        self.norm_diff = norm_diff
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2 + 1, hidden_nf), act,
            nn.Linear(hidden_nf, hidden_nf), act)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf * 2, hidden_nf), act,
            nn.Linear(hidden_nf, hidden_nf))
        coord_last = nn.Linear(hidden_nf, 1, bias=False)
        nn.init.xavier_uniform_(coord_last.weight, gain=0.001)
        self.coord_mlp = nn.Sequential(nn.Linear(hidden_nf, hidden_nf), act, coord_last)
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def forward(self, h, coord, valid):
        T, N, Fdim = h.shape
        diff = coord.unsqueeze(2) - coord.unsqueeze(1)            # (T,N,N,3)
        radial = (diff * diff).sum(-1, keepdim=True)              # (T,N,N,1)
        if self.norm_diff:
            diff = F.normalize(diff, p=2, dim=-1)
        h_i = h.unsqueeze(2).expand(T, N, N, Fdim)
        h_j = h.unsqueeze(1).expand(T, N, N, Fdim)
        m = self.edge_mlp(torch.cat([h_i, h_j, radial], dim=-1))  # (T,N,N,F)
        if self.attention:
            m = m * self.att_mlp(m)
        # mask: kill edges to pad atoms (col j) and self-edges
        col_valid = valid.view(1, 1, N, 1).float()
        eye = torch.eye(N, device=h.device, dtype=torch.bool).view(1, N, N, 1)
        mask = col_valid * (~eye).float()
        m = m * mask
        # equivariant coordinate update (mean over neighbours)
        trans = diff * self.coord_mlp(m)                          # (T,N,N,3)
        if self.clamp:
            trans = trans.clamp(-100.0, 100.0)
        denom = mask.sum(dim=2).clamp_min(1.0)                    # (T,N,1)
        coord = coord + self.coords_weight * (trans.sum(dim=2) / denom)
        # node update (sum over neighbours)
        agg = m.sum(dim=2)                                        # (T,N,F)
        h = h + self.node_mlp(torch.cat([h, agg], dim=-1))
        h = h * valid.view(1, N, 1).float()
        return h, coord


class NSEGNNConformerEncoder(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, max_z: int = 36,
                 residue_vocab_size: int = 64, edge_hidden: int = 64,
                 residue_emb_path: Optional[Union[str, Path]] = None,
                 residue_vocab_tokens: Optional[List[str]] = None,
                 hops: tuple = (1, 2, 4)):
        super().__init__()
        # Same atom/residue priors as the base encoder (residue_emb_path path
        # kept minimal: NS-EGNN PoC uses the learned residue embedding).
        self.hidden_dim = hidden_dim
        self.atom_embed = nn.Embedding(max_z, hidden_dim, padding_idx=0)
        self.res_embed = nn.Embedding(residue_vocab_size, hidden_dim, padding_idx=0)
        self.res_proj: nn.Module = nn.Identity()
        self.input_mix = nn.Linear(hidden_dim, hidden_dim)
        self.hops = hops
        # spectral_embedding input = sum_s (freq_bins_s); freq_bins = n_fft//2+1,
        # n_fft = 2*hop -> hop+1 bins. Sum over scales = sum(hop)+len(hops).
        spec_in = sum(hops) + len(hops)
        self.spectral_embedding = nn.Linear(spec_in, hidden_dim // 2)
        enc_dim = hidden_dim + hidden_dim // 2
        self.time_embedding = nn.Embedding(512, enc_dim)  # frame (conformer) index
        self.layers = nn.ModuleList([
            _EGCLDense(enc_dim) for _ in range(num_layers)])
        self.out_norm = nn.LayerNorm(enc_dim)
        self.out_proj = nn.Linear(enc_dim, hidden_dim)

    def atom_features(self, z: torch.Tensor, res: torch.Tensor) -> torch.Tensor:
        return self.atom_embed(z) + self.res_proj(self.res_embed(res))

    def _spectral(self, coords_TN3: torch.Tensor) -> torch.Tensor:
        """Multi-scale STFT node features. coords: (T,N,3) -> (N, spec_in)."""
        T, N, _ = coords_TN3.shape
        x = coords_TN3.permute(1, 2, 0)  # (N,3,T)
        feats = []
        for hop in self.hops:
            win = max(2, min(2 * hop, T))
            nfft = 2 * hop
            window = torch.hann_window(win, device=x.device)
            mags = []
            for c in range(3):
                st = torch.stft(x[:, c, :], n_fft=nfft, hop_length=hop,
                                win_length=win, window=window, return_complex=True,
                                center=True)
                mags.append(st.abs())  # (N, freq, frames)
            rms = torch.sqrt(sum(m ** 2 for m in mags) / 3.0)  # (N, freq, frames)
            feats.append(rms.mean(dim=2))                      # (N, freq) pool frames
        return torch.cat(feats, dim=1)  # (N, sum(hop)+len(hops))

    def forward_env(self, coords, z, res, pad_mask, batch_index, batch_size):
        """coords (M,Nmax,3), z/res/pad_mask (M,Nmax), batch_index (M,) -> (B,hidden).

        Conformers of a molecule share atoms (z/res/pad identical); only coords
        differ across the K frames. Processed molecule-by-molecule.
        """
        dev = coords.device
        outs = []
        for b in range(batch_size):
            sel = (batch_index == b).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                outs.append(torch.zeros(self.hidden_dim, device=dev)); continue
            c_all = coords[sel]                   # (K, Nmax, 3)
            z0, r0, pad0 = z[sel[0]], res[sel[0]], pad_mask[sel[0]]
            # Heavy-atom graph only: exclude pad AND hydrogens (z<=1). Cuts the
            # dense (T,N,N,F) message tensor ~10x; the chameleonic shape signal
            # lives in the heavy-atom geometry. Gather by index (H is interspersed).
            keep = (~pad0) & (z0 > 1)             # (Nmax,)
            idx = keep.nonzero(as_tuple=True)[0]
            n = int(idx.numel())
            if n == 0:
                outs.append(torch.zeros(self.hidden_dim, device=dev)); continue
            c = c_all[:, idx]                     # (K, n, 3)
            valid_n = torch.ones(n, dtype=torch.bool, device=dev)
            K = c.shape[0]
            h0 = self.input_mix(self.atom_features(z0[idx], r0[idx]))  # (n, hidden)
            spec = self.spectral_embedding(self._spectral(c))        # (n, hidden//2)
            node = torch.cat([h0, spec], dim=-1)                     # (n, enc_dim)
            h = node.unsqueeze(0).expand(K, n, node.shape[-1]).contiguous()
            t_idx = torch.arange(K, device=dev).clamp_max(511)
            h = h + self.time_embedding(t_idx).unsqueeze(1)          # (K,n,enc)
            x = c
            for layer in self.layers:
                h, x = layer(h, x, valid_n)
            h = self.out_norm(h)
            # pool over atoms then frames
            vmask = valid_n.view(1, n, 1).float()
            per_frame = (h * vmask).sum(1) / vmask.sum(1).clamp_min(1.0)  # (K,enc)
            mol = per_frame.mean(0)                                       # (enc,)
            outs.append(self.out_proj(mol))
        return torch.stack(outs, dim=0)  # (B, hidden)

    # Compatibility: a plain forward is not used for this arch (per-molecule path).
    def forward(self, *a, **k):
        raise RuntimeError("NSEGNNConformerEncoder uses forward_env (per-molecule).")

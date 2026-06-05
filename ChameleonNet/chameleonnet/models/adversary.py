"""Gradient-reversal scaffold adversary for OOD (unseen-scaffold) generalization.

Motivation
----------
On the scaffold-OOD split (OD_Murcko), train and test molecules are forced to
have dissimilar ring systems. A model that leans on *scaffold identity* — a
feature that happens to correlate with permeability inside the training
sources (e.g. lariat peptides all come from one source) — collapses when that
correlation breaks at test time. The scaffold identity is a **spurious**
correlate; the chameleonic solvation mechanism is the **causal** signal.

We attach a small classifier that tries to predict which (train-only) scaffold
cluster a peptide belongs to, fed through a gradient-reversal layer (DANN,
Ganin & Lempitsky 2015). The classifier minimises its cross-entropy normally,
but the reversed gradient pushes the *encoder* to make its representation
scaffold-invariant — so the main head is forced onto the transferable
mechanism instead of memorised ring systems.

The adversary reads only the *learned* representation
(``h_water | h_hexane | h_diff | h_seq``); the physics descriptors ``h_desc``
are deliberately left out of the adversarial path so the transferable
descriptor signal is preserved.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Identity in forward; flip + scale the gradient on the way back so the
        # upstream encoder is trained to *fool* the adversary.
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_)


class ScaffoldAdversary(nn.Module):
    """MLP that predicts the scaffold-cluster id from the learned embedding.

    The gradient-reversal happens *before* this head, so the head's own
    parameters always receive a normal (descending) gradient — only the
    gradient that flows further back into the encoder is reversed and scaled by
    ``lambda_``. With ``lambda_ == 0`` the head still learns to read scaffold
    identity (useful as a warmup / diagnostic) while exerting no pressure on the
    encoder.
    """

    def __init__(self, in_dim: int, num_groups: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_groups),
        )

    def forward(self, emb: torch.Tensor, lambda_: float) -> torch.Tensor:
        return self.net(grad_reverse(emb, lambda_))

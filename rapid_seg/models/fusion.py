"""RAPiD Fusion with Attention (FuAtten), paper Sec. 5.1, Fig. 5.

Channel-wise (Squeeze-and-Excitation style) fusion of the complementary
voxel-wise features E_C (coordinates), E_I (intensity/reflectivity) and E_R
(RAPiD embedding):

    E   = concat([E_C, E_I, E_R], dim=channels)      # (V, f*)
    z   = GAP(E)                                      # squeeze  -> (f*,)
    a_z = sigmoid(W2 . ReLU(W1 . z))                  # excitation
    E'  = a_z (.) E                                   # re-weighted features

Here voxel features are already pooled to (V, f) vectors, so the GAP reduces
over the batch/voxel axis to form the channel descriptor.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class FuAtten(nn.Module):
    def __init__(self, total_dim: int, reduction: int = 8):
        super().__init__()
        hidden = max(total_dim // reduction, 4)
        self.fc1 = nn.Linear(total_dim, hidden)
        self.fc2 = nn.Linear(hidden, total_dim)
        self.act = nn.ReLU(inplace=True)
        self.gate = nn.Sigmoid()

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        e = torch.cat(list(feats), dim=1)          # (V, f*)
        z = e.mean(dim=0)                          # squeeze / GAP -> (f*,)
        a = self.gate(self.fc2(self.act(self.fc1(z))))  # excitation -> (f*,)
        return e * a[None, :]                      # channel-wise re-weighting

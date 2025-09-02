"""Voxel-based Set Attention (VSA), the building block of the outer AE.

Follows the induced set-attention formulation of VoxSeT [he2022voxela] used in
the paper (Sec. 4, Eq. 11-14):

  Encoder (point -> voxel):
      (K, V) = Proj(G)                      linear projections of point feats
      A~      = softmax_scatter(K L^T, I^v) attention over points *within a voxel*
      H       = A~ (.) V                    weighted values
      H^v     = sum_scatter(H, I^v)         aggregate to `l` latent codes / voxel

  Decoder (voxel -> point):
      (K*, V*) = Proj(H^)                   projections of enriched latent codes
      A*       = K*_i Q_i^T                 per-point attention over the l codes
      G^       = softmax(A*)^T V*           reconstruct point features

`I^v` is the point-to-voxel index; latent codes `L` are learnable inducing
points shared across voxels. All grouping uses the pure-torch scatter ops.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..utils.scatter import scatter_add, scatter_softmax


class VSAEncoder(nn.Module):
    """Point features -> per-voxel latent representation H^v of shape (V, l, d)."""

    def __init__(self, in_dim: int, dim: int, num_latents: int):
        super().__init__()
        self.num_latents = num_latents
        self.dim = dim
        self.proj_k = nn.Linear(in_dim, num_latents)      # K L^T -> (m, l) scores
        self.proj_v = nn.Linear(in_dim, dim)              # values (m, d)

    def forward(self, feats: torch.Tensor, voxel_idx: torch.Tensor, num_voxels: int):
        # attention scores of each point to each of the l latent codes
        scores = self.proj_k(feats)                        # (m, l)
        values = self.proj_v(feats)                        # (m, d)
        hv = feats.new_zeros((num_voxels, self.num_latents, self.dim))
        for j in range(self.num_latents):
            # softmax over points sharing a voxel, for latent code j
            attn = scatter_softmax(scores[:, j], voxel_idx, num_voxels)   # (m,)
            weighted = values * attn[:, None]                              # (m, d)
            hv[:, j, :] = scatter_add(weighted, voxel_idx, num_voxels)     # (V, d)
        return hv


class VSADecoder(nn.Module):
    """Per-voxel latent H^v + point queries -> reconstructed point features."""

    def __init__(self, in_dim: int, latent_dim: int, out_dim: int, num_latents: int):
        super().__init__()
        self.num_latents = num_latents
        self.proj_q = nn.Linear(in_dim, latent_dim)        # point queries (m, d)
        self.proj_k = nn.Linear(latent_dim, latent_dim)    # latent keys
        self.proj_v = nn.Linear(latent_dim, out_dim)       # latent values

    def forward(self, hv: torch.Tensor, query_feats: torch.Tensor, voxel_idx: torch.Tensor):
        # broadcast per-voxel latent codes back to their points: (m, l, d)
        h_pt = hv[voxel_idx]
        q = self.proj_q(query_feats)                       # (m, d)
        k = self.proj_k(h_pt)                              # (m, l, d)
        v = self.proj_v(h_pt)                              # (m, l, out)
        # per-point attention over the l latent codes
        attn = torch.softmax((k * q[:, None, :]).sum(-1) / (q.shape[-1] ** 0.5), dim=1)  # (m, l)
        out = (attn[:, :, None] * v).sum(dim=1)            # (m, out)
        return out

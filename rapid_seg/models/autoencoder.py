"""Double-nested RAPiD AutoEncoder (paper Sec. 4, Fig. 3).

Outer module  : a VSA AutoEncoder that performs point <-> voxel conversion.
Inner module  : a conv AutoEncoder that reduces the voxel feature dim d -> d'
                and exchanges voxel-level information via a ConvFFN with dual
                depth-wise convolutions.

Pipeline (m points, d_in features per point):
    G (m, d_in)
      --outer VSA encoder-->  H^v (V, l, d)
      --inner encoder------>  h_bar (V, l, d')      # compressed embedding
      --ConvFFN------------>  h_bar (V, l, d')      # spatial interactivity
      --inner decoder------>  H^v_hat (V, l, d)
      --outer VSA decoder-->  G_hat (m, d_in)       # reconstruction

Outputs used downstream:
    h_bar   : compressed voxel-wise RAPiD representation (fed to FuAtten)
    G_hat   : reconstruction (MSE recon loss, Eq. 10)
    h_point : per-point embedding pooled from h_bar and broadcast by voxel
              index (drives the class-aware contrastive loss, Eq. 9)

The ConvFFN performs voxel-level information exchange after dimensionality
reduction via dual depth-wise convolutions (Conv1d, groups=channels) over the
latent-code axis.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vsa import VSAEncoder, VSADecoder


class InnerEncoder(nn.Module):
    """Conv + BatchNorm dimensionality reduction d -> d' over the l latent axis."""

    def __init__(self, dim: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        chans = torch.linspace(dim, latent_dim, num_layers + 1).round().long().tolist()
        layers = []
        for i in range(num_layers):
            layers += [
                nn.Conv1d(chans[i], chans[i + 1], kernel_size=3, padding=1),
                nn.BatchNorm1d(chans[i + 1]),
                nn.GELU(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, hv: torch.Tensor) -> torch.Tensor:      # (V, l, d)
        x = hv.transpose(1, 2)                                # (V, d, l)
        x = self.net(x)                                       # (V, d', l)
        return x.transpose(1, 2)                              # (V, l, d')


class ConvFFN(nn.Module):
    """Dual depth-wise convolutions for voxel-level information exchange."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.dw1 = nn.Conv1d(latent_dim, latent_dim, 3, padding=1, groups=latent_dim)
        self.act = nn.GELU()
        self.dw2 = nn.Conv1d(latent_dim, latent_dim, 3, padding=1, groups=latent_dim)

    def forward(self, h_bar: torch.Tensor) -> torch.Tensor:   # (V, l, d')
        x = h_bar.transpose(1, 2)                             # (V, d', l)
        x = self.dw2(self.act(self.dw1(x)))
        return x.transpose(1, 2) + h_bar                      # residual


class InnerDecoder(nn.Module):
    """DeConv layers reconstructing d' -> d over the l latent axis."""

    def __init__(self, latent_dim: int, dim: int, num_layers: int = 2):
        super().__init__()
        chans = torch.linspace(latent_dim, dim, num_layers + 1).round().long().tolist()
        layers = []
        for i in range(num_layers):
            layers += [
                nn.Conv1d(chans[i], chans[i + 1], kernel_size=3, padding=1),
                nn.GELU() if i < num_layers - 1 else nn.Identity(),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, h_bar: torch.Tensor) -> torch.Tensor:   # (V, l, d')
        x = h_bar.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)                              # (V, l, d)


class RAPiDAutoEncoder(nn.Module):
    """Double-nested AE producing the compressed voxel-wise RAPiD embedding."""

    def __init__(
        self,
        in_dim: int,
        dim: int = 64,
        latent_dim: int = 16,
        num_latents: int = 8,
        inner_layers: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.dim = dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents

        self.outer_encoder = VSAEncoder(in_dim, dim, num_latents)
        self.inner_encoder = InnerEncoder(dim, latent_dim, inner_layers)
        self.conv_ffn = ConvFFN(latent_dim)
        self.inner_decoder = InnerDecoder(latent_dim, dim, inner_layers)
        self.outer_decoder = VSADecoder(in_dim, dim, in_dim, num_latents)

    def encode(self, feats, voxel_idx, num_voxels):
        hv = self.outer_encoder(feats, voxel_idx, num_voxels)   # (V, l, d)
        h_bar = self.inner_encoder(hv)                          # (V, l, d')
        h_bar = self.conv_ffn(h_bar)                            # (V, l, d')
        return h_bar

    def forward(self, feats, voxel_idx, num_voxels):
        hv = self.outer_encoder(feats, voxel_idx, num_voxels)   # (V, l, d)
        h_bar = self.inner_encoder(hv)                          # (V, l, d')
        h_bar = self.conv_ffn(h_bar)                            # (V, l, d')
        hv_hat = self.inner_decoder(h_bar)                      # (V, l, d)
        recon = self.outer_decoder(hv_hat, feats, voxel_idx)    # (m, d_in)

        # per-point embedding for the class-aware contrastive loss:
        # pool the compressed embedding over latent codes, broadcast by voxel.
        h_point = h_bar.mean(dim=1)[voxel_idx]                  # (m, d')
        return {"h_bar": h_bar, "recon": recon, "h_point": h_point}

"""RAPiD-Seg network (paper Sec. 5).

Single-modal (LiDAR-only) 3D semantic segmentation with three complementary
point-wise feature streams (Sec. 5):
    (a) coordinate features  F_C = (x, y, z)
    (b) intensity features   F_I = (intensity, reflectivity)
    (c) RAPiD features       F_R = RAPiD(P_RoI; k)

F_C (+) F_I are voxelised by VSA voxel encoders -> E_C, E_I.
F_R is embedded by the double-nested RAPiD AE -> E_R (compressed h_bar).
E_C, E_I, E_R are fused by FuAtten and fed to the backbone for segmentation.

Two variants (Sec. 5.2, Fig. 6):
    R-RAPiD-Seg : early fusion of E_C, E_I, E_R (R-RAPiD features).
    C-RAPiD-Seg : additionally fuses E_R^C from C-RAPiD features (needs GT
                  labels at train time / pseudo-labels at test time).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .vsa import VSAEncoder
from .autoencoder import RAPiDAutoEncoder
from .fusion import FuAtten
from .backbone import build_backbone
from ..utils.scatter import scatter_mean


def voxelize(xyz: torch.Tensor, voxel_size: float):
    """Grid voxelisation -> (voxel_idx per point, voxel coords, num_voxels)."""
    coords = torch.floor(xyz / voxel_size).long()               # (m, 3)
    uniq, inv = torch.unique(coords, dim=0, return_inverse=True)
    return inv, uniq, uniq.shape[0]


class RapidSeg(nn.Module):
    def __init__(
        self,
        num_classes: int,
        rapid_dim: int,                 # k_max (RAPiD feature dim)
        coord_dim: int = 3,
        intensity_dim: int = 2,
        voxel_size: float = 0.10,
        vsa_dim: int = 32,
        vsa_latents: int = 8,
        ae_dim: int = 64,
        ae_latent_dim: int = 16,
        backbone_dim: int = 96,
        backbone_blocks: int = 4,
        variant: str = "R",             # "R" or "C"
        backbone: str = "auto",         # "auto"|"minkunet"|"voxel"
    ):
        super().__init__()
        assert variant in ("R", "C")
        self.variant = variant
        self.voxel_size = voxel_size
        self.vsa_latents = vsa_latents

        # coordinate & intensity voxel encoders (VSA voxelisation)
        self.enc_c = VSAEncoder(coord_dim, vsa_dim, vsa_latents)
        self.enc_i = VSAEncoder(intensity_dim, vsa_dim, vsa_latents)

        # shared RAPiD AE for R- (and C-) RAPiD features
        self.rapid_ae = RAPiDAutoEncoder(
            in_dim=rapid_dim, dim=ae_dim, latent_dim=ae_latent_dim, num_latents=vsa_latents
        )

        fused_dim = vsa_dim + vsa_dim + ae_latent_dim               # E_C + E_I + E_R
        if variant == "C":
            fused_dim += ae_latent_dim                              # + E_R^C
        self.fuatten = FuAtten(fused_dim)
        self.backbone = build_backbone(backbone, fused_dim, num_classes,
                                       backbone_dim, backbone_blocks)

    def _voxel_pool(self, hv: torch.Tensor) -> torch.Tensor:
        """Pool a (V, l, d) latent tensor over the latent axis -> (V, d)."""
        return hv.mean(dim=1)

    def forward(
        self,
        xyz: torch.Tensor,             # (m, 3)
        intensity: torch.Tensor,       # (m, intensity_dim)
        rapid_r: torch.Tensor,         # (m, rapid_dim)  R-RAPiD features
        rapid_c: Optional[torch.Tensor] = None,   # (m, rapid_dim) C-RAPiD (variant C)
    ):
        voxel_idx, voxel_coords, num_voxels = voxelize(xyz, self.voxel_size)

        # branch C & I: VSA voxel encoders -> per-voxel features
        e_c = self._voxel_pool(self.enc_c(xyz, voxel_idx, num_voxels))       # (V, vsa_dim)
        e_i = self._voxel_pool(self.enc_i(intensity, voxel_idx, num_voxels)) # (V, vsa_dim)

        # branch R: RAPiD AE
        ae_r = self.rapid_ae(rapid_r, voxel_idx, num_voxels)
        e_r = self._voxel_pool(ae_r["h_bar"])                                # (V, d')

        feats = [e_c, e_i, e_r]
        aux = {"ae_r": ae_r}

        if self.variant == "C":
            if rapid_c is None:
                rapid_c = rapid_r                # fall back until pseudo-labels ready
            ae_c = self.rapid_ae(rapid_c, voxel_idx, num_voxels)
            e_rc = self._voxel_pool(ae_c["h_bar"])
            feats.append(e_rc)
            aux["ae_c"] = ae_c

        fused = self.fuatten(feats)                                          # (V, fused_dim)
        point_logits = self.backbone(fused, voxel_coords, voxel_idx)         # (m, C)
        aux["voxel_idx"] = voxel_idx
        aux["num_voxels"] = num_voxels
        return point_logits, aux

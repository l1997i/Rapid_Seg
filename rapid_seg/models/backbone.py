"""Lightweight voxel-wise segmentation backbone.

The default backbone is Minkowski-UNet34 (see ``minkunet.py``), which needs
MinkowskiEngine + CUDA. This module is a dependency-free alternative for CPU /
Apple-Silicon runs and quick experiments: per-voxel residual MLP blocks
interleaved with k-NN neighbourhood aggregation over voxel centres provide a
spatial receptive field without sparse convolutions. It consumes the
FuAtten-fused voxel features and predicts per-voxel logits, broadcast back to
points. Selected automatically via ``backbone="auto"`` when MinkowskiEngine is
unavailable, or explicitly with ``backbone="voxel"``.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class NeighbourMix(nn.Module):
    """Aggregate features over k nearest voxels (spatial context, ~sparse conv)."""

    def __init__(self, dim: int, k: int = 8, max_voxels: int = 12000):
        super().__init__()
        self.k = k
        self.max_voxels = max_voxels
        self.merge = nn.Linear(2 * dim, dim)

    def forward(self, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        v = feats.shape[0]
        if v <= 1 or v > self.max_voxels:      # skip O(V^2) mixing on huge scenes
            return feats
        d = torch.cdist(coords.float(), coords.float())
        kk = min(self.k + 1, v)
        _, idx = torch.topk(d, kk, dim=1, largest=False)   # includes self
        neigh = feats[idx].mean(dim=1)                     # (V, dim)
        return self.merge(torch.cat([feats, neigh], dim=1))


class VoxelBackbone(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dim: int = 96, num_blocks: int = 4):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(in_dim, dim), nn.BatchNorm1d(dim), nn.GELU())
        self.blocks = nn.ModuleList()
        self.mixers = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResBlock(dim))
            self.mixers.append(NeighbourMix(dim) if i % 2 == 1 else nn.Identity())
        self.head = nn.Linear(dim, num_classes)

    def forward(self, voxel_feats: torch.Tensor, voxel_coords: torch.Tensor, voxel_idx: torch.Tensor):
        x = self.stem(voxel_feats)
        for blk, mix in zip(self.blocks, self.mixers):
            x = blk(x)
            x = mix(x, voxel_coords) if isinstance(mix, NeighbourMix) else x
        voxel_logits = self.head(x)                # (V, num_classes)
        return voxel_logits[voxel_idx]             # broadcast to points (m, C)


def build_backbone(kind: str, in_dim: int, num_classes: int,
                   dim: int = 96, num_blocks: int = 4):
    """Backbone factory.

    kind:
        "auto"     -> MinkowskiUNet34 when CUDA + MinkowskiEngine are available
                      (the paper's default), else the light voxel backbone.
        "minkunet" -> force MinkowskiUNet34 (errors if ME/CUDA missing).
        "voxel"    -> force the light voxel backbone (CPU/MPS-friendly).
    """
    from .minkunet import minkowski_available, MinkUNetBackbone

    if kind == "auto":
        kind = "minkunet" if minkowski_available() else "voxel"

    if kind == "minkunet":
        print("[backbone] MinkowskiUNet34 (sparse conv, NVIDIA-GPU default)")
        return MinkUNetBackbone(in_dim, num_classes)
    if kind == "voxel":
        if minkowski_available():
            print("[backbone] light voxel-UNet (forced; MinkUNet34 available)")
        else:
            print("[backbone] light voxel-UNet (CPU/MPS fallback; MinkowskiEngine/CUDA unavailable)")
        return VoxelBackbone(in_dim, num_classes, dim, num_blocks)
    raise ValueError(f"unknown backbone kind: {kind}")

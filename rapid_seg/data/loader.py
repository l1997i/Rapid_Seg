"""PointCloudLoader — read and preprocess raw LiDAR scans.

Reads KITTI-format ``.bin`` scans (``float32`` ``[x, y, z, remission]``) and
returns coordinates and reflectivity ready for :class:`RAPiDCalculator`::

    from rapid_seg import PointCloudLoader
    loader = PointCloudLoader()
    coordinates, reflectivity = loader.load_and_preprocess("scan.bin")
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


class PointCloudLoader:
    def __init__(self, device: str = "cpu", max_points: Optional[int] = None, seed: int = 0):
        self.device = device
        self.max_points = max_points
        self.seed = seed

    def load_bin(self, path: str) -> np.ndarray:
        """Read a KITTI ``.bin`` scan -> ``(N, 4)`` ``[x, y, z, remission]``."""
        return np.fromfile(path, dtype=np.float32).reshape(-1, 4)

    def load_and_preprocess(self, path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load a scan and return ``(coordinates (N,3), reflectivity (N,))`` tensors."""
        scan = self.load_bin(path)
        coords = scan[:, :3]
        refl = scan[:, 3]
        if self.max_points is not None and coords.shape[0] > self.max_points:
            rng = np.random.default_rng(self.seed)
            sel = rng.choice(coords.shape[0], self.max_points, replace=False)
            coords, refl = coords[sel], refl[sel]
        coordinates = torch.from_numpy(np.ascontiguousarray(coords)).to(self.device)
        reflectivity = torch.from_numpy(np.ascontiguousarray(refl)).to(self.device)
        return coordinates, reflectivity

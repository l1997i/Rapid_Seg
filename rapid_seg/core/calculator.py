"""RAPiDCalculator — the public API for computing RAPiD features.

Wraps the RAPiD feature functions behind a small, stable interface::

    from rapid_seg import RAPiDCalculator
    calc = RAPiDCalculator(device="cuda")
    feats = calc.compute_rapid_features(coords, reflectivity, k=10)
    ra    = calc.compute_range_aware_rapid(coords, reflectivity,
                                           k_close=10, k_mid=7, k_far=5)
    r     = calc.compute_r_rapid(coords, reflectivity)          # intra-ring
    c     = calc.compute_c_rapid(coords, reflectivity, labels)  # intra-class
"""
from __future__ import annotations

from typing import Optional

import torch

from ..models.rapid_features import (
    RAPiDConfig,
    compute_rapid_frame,
    compute_r_rapid,
    compute_c_rapid,
)
from ..utils.device import get_device


class RAPiDCalculator:
    def __init__(self, device: str = "auto", config: Optional[RAPiDConfig] = None):
        self.device = get_device() if device == "auto" else torch.device(device)
        self.config = config or RAPiDConfig()

    # -- helpers -------------------------------------------------------------
    def _prep(self, coordinates, reflectivity):
        coords = torch.as_tensor(coordinates, dtype=torch.float32, device=self.device)
        refl = torch.as_tensor(reflectivity, dtype=torch.float32, device=self.device).reshape(-1)
        return coords, refl

    def _cfg(self, **overrides) -> RAPiDConfig:
        cfg = RAPiDConfig(**{**self.config.__dict__, **overrides})
        return cfg

    # -- public API ----------------------------------------------------------
    def compute_rapid_features(self, coordinates, reflectivity, k: int) -> torch.Tensor:
        """Standard RAPiD: sorted 4D neighbour distances with a single ``k``.

        The whole cloud is treated as one RoI. Returns ``(N, k)``.
        """
        coords, refl = self._prep(coordinates, reflectivity)
        cfg = self._cfg(k_near=k, k_mid=k, k_far=k)
        group = torch.zeros(coords.shape[0], dtype=torch.long, device=self.device)
        return compute_rapid_frame(coords, refl, group, cfg)

    def compute_range_aware_rapid(self, coordinates, reflectivity,
                                  k_close: int, k_mid: int, k_far: int) -> torch.Tensor:
        """Range-aware RAPiD: ``k`` adapts to range. Returns ``(N, max(k))``."""
        coords, refl = self._prep(coordinates, reflectivity)
        cfg = self._cfg(k_near=k_close, k_mid=k_mid, k_far=k_far)
        group = torch.zeros(coords.shape[0], dtype=torch.long, device=self.device)
        return compute_rapid_frame(coords, refl, group, cfg)

    def compute_r_rapid(self, coordinates, reflectivity) -> torch.Tensor:
        """Intra-Ring RAPiD (RoI = the LiDAR ring of each point)."""
        coords, refl = self._prep(coordinates, reflectivity)
        return compute_r_rapid(coords, refl, self.config)

    def compute_c_rapid(self, coordinates, reflectivity, labels, ignore_index: int = -1) -> torch.Tensor:
        """Intra-Class RAPiD (RoI = the semantic class of each point)."""
        coords, refl = self._prep(coordinates, reflectivity)
        lab = torch.as_tensor(labels, dtype=torch.long, device=self.device).reshape(-1)
        return compute_c_rapid(coords, refl, lab, self.config, ignore_index)

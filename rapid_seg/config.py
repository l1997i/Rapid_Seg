"""Configuration profiles for RAPiD.

Exposes :class:`RAPiDConfig` and :func:`create_config`, the entry point used
throughout the examples::

    from rapid_seg.config import create_config, RAPiDConfig
    cfg = create_config("balanced", k_mid=8)

Three predefined profiles trade speed for accuracy:

    * ``fast``      - smaller ``k`` / fewer rings, lowest latency
    * ``balanced``  - default, the paper's SemanticKITTI setting
    * ``accurate``  - larger ``k``, best feature quality
"""
from __future__ import annotations

from typing import Any, Dict

from .models.rapid_features import RAPiDConfig

PROFILES: Dict[str, Dict[str, Any]] = {
    "fast": dict(k_near=7, k_mid=5, k_far=3, num_beams=64, batch_size=64, precision="float16"),
    "balanced": dict(k_near=10, k_mid=7, k_far=5, num_beams=64, batch_size=32, precision="float32"),
    "accurate": dict(k_near=13, k_mid=10, k_far=7, num_beams=64, batch_size=16, precision="float32"),
}


def create_config(profile: str = "balanced", **overrides: Any) -> RAPiDConfig:
    """Build a :class:`RAPiDConfig` from a named profile with optional overrides.

    Args:
        profile: one of ``"fast"``, ``"balanced"``, ``"accurate"``.
        **overrides: any :class:`RAPiDConfig` field (e.g. ``k_mid=8``,
            ``device="cuda"``).
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {list(PROFILES)}")
    params = dict(PROFILES[profile])
    params.update(overrides)
    return RAPiDConfig(**params)


__all__ = ["RAPiDConfig", "create_config", "PROFILES"]

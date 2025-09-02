"""Device selection.

The paper trains on 4x NVIDIA A100 GPUs. Following that, this reproduction
*defaults to NVIDIA CUDA*: `get_device()` returns a CUDA device whenever one is
available. When no NVIDIA GPU is present (e.g. on an Apple-Silicon Mac used for
local verification) it falls back to Apple MPS, and finally to CPU.
"""
from __future__ import annotations

import torch


def get_device(prefer: str = "cuda") -> torch.device:
    """Return the compute device.

    Priority (the NVIDIA GPU is always preferred when present):
        cuda  ->  mps  ->  cpu

    Args:
        prefer: "cuda" (default) enforces the CUDA-first order above. Pass
            "cpu" to force CPU (useful for deterministic debugging).
    """
    if prefer == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device):
    """Preferred autocast dtype per backend (bf16 on CUDA, else fp16/none)."""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return None

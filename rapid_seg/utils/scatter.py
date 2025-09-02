"""Pure-PyTorch scatter operations.

The paper relies on `torch_scatter` (pytorch/scatter) inside the Voxel Set
Attention module (scatter-sum / scatter-softmax over point-to-voxel indices).
`torch_scatter` ships CUDA/C++ extensions that do not build on CPU-only / MPS
setups, so we re-implement the handful of primitives we need with native
`torch` ops. Semantics match `torch_scatter` for `dim=0`.
"""
from __future__ import annotations

import torch


def _broadcast_index(index: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Expand a 1-D `index` (N,) to match the shape of `src` (N, ...)."""
    if index.dim() != 1:
        raise ValueError("index must be 1-D (one group id per row)")
    view = [index.shape[0]] + [1] * (src.dim() - 1)
    return index.view(view).expand_as(src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum rows of `src` sharing the same `index`. Output shape (dim_size, ...)."""
    out = src.new_zeros((dim_size,) + tuple(src.shape[1:]))
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Mean over rows sharing the same `index`."""
    summed = scatter_add(src, index, dim_size)
    count = src.new_zeros((dim_size,) + (1,) * (src.dim() - 1))
    ones = src.new_ones((src.shape[0],) + (1,) * (src.dim() - 1))
    count.index_add_(0, index, ones)
    return summed / count.clamp_min(1.0)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Per-group maximum. Empty groups map to the tensor min (a very negative value)."""
    out = src.new_full((dim_size,) + tuple(src.shape[1:]), float("-inf"))
    out.scatter_reduce_(0, _broadcast_index(index, src), src, reduce="amax", include_self=True)
    # groups with no members stay at -inf; replace with 0 for numeric safety.
    out[out == float("-inf")] = 0.0
    return out


def scatter_softmax(src: torch.Tensor, index: torch.Tensor, dim_size: int, eps: float = 1e-12) -> torch.Tensor:
    """Softmax computed *within each group* defined by `index`.

    Returns a tensor the same shape as `src`; entries belonging to the same
    group sum to 1 along the group. Numerically stabilised by subtracting the
    per-group max.
    """
    group_max = scatter_max(src.detach(), index, dim_size)
    max_per_row = group_max[index]
    exp = torch.exp(src - max_per_row)
    denom = scatter_add(exp, index, dim_size)[index] + eps
    return exp / denom

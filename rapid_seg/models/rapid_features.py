"""Range-Aware Pointwise Distance Distribution (RAPiD) features.

RAPiD encodes, for every point, the sorted vector of 4D distances to its `k`
nearest neighbours within a region of interest (RoI). The 4D distance combines
3D geometry and surface-material reflectivity (Eq. 2-4), the descriptor is
double-sorted (Eq. 1) so it is invariant to rigid transforms and neighbour
ordering, and `k` adapts to LiDAR density with range (`k_near/k_mid/k_far`).

  * 4D distance combining 3D geometry and reflectivity   -- Eq. (2)
  * reflectivity scaling function g(r)                    -- Eq. (3)-(4)
  * lexicographic / inner double-sort                     -- Eq. (1)
  * outlier clipping at threshold delta + normalisation
  * range-aware neighbour count (k_near/k_mid/k_far)
  * Intra-Ring RAPiD  (R-RAPiD): RoI = the LiDAR ring
  * Intra-Class RAPiD (C-RAPiD): RoI = the semantic class

Rows are max-padded to `k_max = max(k_near, k_mid, k_far)` so a frame yields a
fixed `(m, k_max)` feature matrix G.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RAPiDConfig:
    """RAPiD feature configuration (also the top-level ``rapid_seg`` config).

    ``k_near/k_mid/k_far`` are the range-aware neighbour counts (SemanticKITTI
    default 10/7/5; nuScenes 8/6/3). ``k_close`` is an alias of ``k_near``.
    ``device`` accepts ``"auto"`` (CUDA -> MPS -> CPU), ``"cuda"`` or ``"cpu"``.
    """

    # range-aware neighbour counts
    k_near: int = 10
    k_mid: int = 7
    k_far: int = 5
    # range bucket thresholds (metres): near < r_near <= mid < r_mid <= far
    r_near: float = 20.0
    r_mid: float = 35.0
    # number of LiDAR beams / rings for R-RAPiD
    num_beams: int = 64
    # outlier threshold in the normalised distribution
    delta: float = 0.9
    # small epsilon for numeric safety
    eps: float = 1e-9
    # runtime knobs
    device: str = "auto"
    batch_size: int = 32
    precision: str = "float32"

    @property
    def k_max(self) -> int:
        return max(self.k_near, self.k_mid, self.k_far)

    @property
    def k_close(self) -> int:
        """Alias for ``k_near`` (close-range neighbour count)."""
        return self.k_near

    def get_k_for_standard_rapid(self) -> int:
        """Neighbour count used by non range-aware (standard) RAPiD."""
        return self.k_mid

    def resolved_device(self) -> str:
        """Concrete device string, resolving ``"auto"``."""
        if self.device != "auto":
            return self.device
        from ..utils.device import get_device
        return str(get_device())


def _range_bucket_k(rng: torch.Tensor, cfg: RAPiDConfig) -> torch.Tensor:
    """Map each point's range to its neighbour count k (Sec. 3.1)."""
    k = torch.full_like(rng, cfg.k_far, dtype=torch.long)
    k = torch.where(rng < cfg.r_mid, torch.tensor(cfg.k_mid, device=rng.device), k)
    k = torch.where(rng < cfg.r_near, torch.tensor(cfg.k_near, device=rng.device), k)
    return k


def assign_rings(xyz: torch.Tensor, num_beams: int) -> torch.Tensor:
    """Assign every point to a LiDAR ring via elevation angle (Eq. 6-7).

    phi_i = floor( arcsin( z / ||p|| ) / d_phi ),  binned into `num_beams` rings.
    """
    rng = xyz.norm(dim=1).clamp_min(1e-6)
    elev = torch.asin((xyz[:, 2] / rng).clamp(-1.0, 1.0))  # vertical angle
    e_min, e_max = elev.min(), elev.max()
    d_phi = (e_max - e_min) / max(num_beams, 1)
    ring = torch.floor((elev - e_min) / (d_phi + 1e-9)).long()
    return ring.clamp(0, num_beams - 1)


def _pdd_within_group(
    xyz: torch.Tensor,
    reflect: torch.Tensor,
    k_per_point: torch.Tensor,
    k_max: int,
):
    """kNN 4D-distance search *within a single RoI group*.

    Returns:
        coord_d: (n, k_max) Euclidean coordinate distance to each neighbour
        refl_d:  (n, k_max) raw reflectivity difference to each neighbour
        valid:   (n, k_max) bool mask of real (non-padded) neighbours
    """
    n = xyz.shape[0]
    device = xyz.device
    coord_d = torch.zeros((n, k_max), device=device)
    refl_d = torch.zeros((n, k_max), device=device)
    valid = torch.zeros((n, k_max), dtype=torch.bool, device=device)
    if n <= 1:
        return coord_d, refl_d, valid

    # pairwise coordinate distances within the group
    dmat = torch.cdist(xyz, xyz)                       # (n, n)
    kk = min(k_max + 1, n)                             # +1 to drop self
    nn_d, nn_idx = torch.topk(dmat, kk, dim=1, largest=False)
    # drop the self column (distance 0)
    nn_d, nn_idx = nn_d[:, 1:], nn_idx[:, 1:]          # (n, kk-1)
    avail = nn_d.shape[1]

    coord_d[:, :avail] = nn_d
    refl_d[:, :avail] = reflect[nn_idx] - reflect[:, None]
    # mark validity: neighbour column j valid if j < this point's own k
    col = torch.arange(k_max, device=device)[None, :]
    valid_by_k = col < k_per_point[:, None]
    valid_by_avail = col < avail
    valid = valid_by_k & valid_by_avail
    return coord_d, refl_d, valid


def compute_rapid_frame(
    xyz: torch.Tensor,
    reflect: torch.Tensor,
    group_id: torch.Tensor,
    cfg: RAPiDConfig,
) -> torch.Tensor:
    """Compute the per-point RAPiD descriptor for one frame given RoI groups.

    Args:
        xyz:      (m, 3) point coordinates
        reflect:  (m,)   per-point reflectivity
        group_id: (m,)   RoI id per point (ring id for R-RAPiD, class id for C-RAPiD)
        cfg:      RAPiDConfig

    Returns:
        G: (m, k_max) normalised, inner-sorted RAPiD feature matrix.
    """
    m = xyz.shape[0]
    device = xyz.device
    k_max = cfg.k_max
    rng = xyz.norm(dim=1)
    k_per_point = _range_bucket_k(rng, cfg)

    coord_d = torch.zeros((m, k_max), device=device)
    refl_d = torch.zeros((m, k_max), device=device)
    valid = torch.zeros((m, k_max), dtype=torch.bool, device=device)

    # process each RoI group independently (Eq. 5 / Eq. 8 union over rings/classes)
    for g in torch.unique(group_id):
        sel = (group_id == g).nonzero(as_tuple=True)[0]
        cd, rd, vd = _pdd_within_group(xyz[sel], reflect[sel], k_per_point[sel], k_max)
        coord_d[sel], refl_d[sel], valid[sel] = cd, rd, vd

    # --- reflectivity scaling g(r), Eq. (3)-(4) ---
    # D_min / D_max: range of coordinate-difference norms over all considered pairs
    if valid.any():
        cd_valid = coord_d[valid]
        d_min = cd_valid.min()
        d_max = cd_valid.max().clamp_min(d_min + cfg.eps)
    else:
        d_min = torch.tensor(0.0, device=device)
        d_max = torch.tensor(1.0, device=device)
    r_min, r_max = reflect.min(), reflect.max()
    # g is affine, so g(r_j) - g(r_l) = (r_j - r_l) * (D_max-D_min)/(r_max-r_min)
    r_scale = (d_max - d_min) / (r_max - r_min).clamp_min(cfg.eps)
    refl_term = refl_d * r_scale

    # --- 4D distance, Eq. (2) ---
    rho = torch.sqrt(coord_d.pow(2) + refl_term.pow(2) + cfg.eps)
    # invalid / padded entries pushed to +inf so they sort last
    rho = torch.where(valid, rho, torch.full_like(rho, float("inf")))

    # --- inner sort ascending within each row, Eq. (1) ---
    rho, _ = torch.sort(rho, dim=1)

    # --- normalise to [0, 1] and clip outliers at delta (Sec. 3.1) ---
    finite = torch.isfinite(rho)
    scale = rho[finite].max() if finite.any() else torch.tensor(1.0, device=device)
    scale = scale.clamp_min(cfg.eps)
    norm = rho / scale                                   # padded entries -> +inf
    outlier = norm > cfg.delta                           # includes padded (+inf)
    # per-row max of the non-outlier (in-distribution) entries
    masked = torch.where(outlier, torch.full_like(norm, float("-inf")), norm)
    row_cap = masked.max(dim=1, keepdim=True).values
    row_cap = torch.where(torch.isfinite(row_cap), row_cap, torch.full_like(row_cap, cfg.delta))
    G = torch.where(outlier, row_cap.expand_as(norm), norm)
    return G.contiguous()


def compute_r_rapid(xyz: torch.Tensor, reflect: torch.Tensor, cfg: RAPiDConfig) -> torch.Tensor:
    """Intra-Ring RAPiD (Sec. 3.2): RoI = the LiDAR ring of each point."""
    ring = assign_rings(xyz, cfg.num_beams)
    return compute_rapid_frame(xyz, reflect, ring, cfg)


def compute_c_rapid(
    xyz: torch.Tensor,
    reflect: torch.Tensor,
    labels: torch.Tensor,
    cfg: RAPiDConfig,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Intra-Class RAPiD (Sec. 3.2): RoI = the semantic class of each point.

    `labels` are ground-truth at train time or pseudo-labels at test time
    (produced by a pre-trained R-RAPiD-Seg, per the C-RAPiD-Seg variant).
    """
    group = labels.clone().long()
    if ignore_index is not None:
        # keep ignored points in their own singleton group
        group = torch.where(group == ignore_index, torch.full_like(group, -1), group)
    return compute_rapid_frame(xyz, reflect, group, cfg)

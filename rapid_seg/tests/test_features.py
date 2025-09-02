"""Tests for RAPiD feature computation on the real SemanticKITTI subset."""
import torch

from rapid_seg import RAPiDCalculator, PointCloudLoader
from rapid_seg.config import create_config


def _real_cloud(bin_path, n=3000):
    coords, reflectivity = PointCloudLoader(max_points=n).load_and_preprocess(bin_path)
    return coords, reflectivity


def test_standard_rapid_shape_real(bin_path):
    coords, refl = _real_cloud(bin_path)
    feats = RAPiDCalculator(device="cpu").compute_rapid_features(coords, refl, k=10)
    assert feats.shape == (coords.shape[0], 10)
    assert torch.isfinite(feats).all()
    assert feats.min() >= 0.0 and feats.max() <= 1.0 + 1e-4     # normalised


def test_range_aware_shape_real(bin_path):
    coords, refl = _real_cloud(bin_path)
    feats = RAPiDCalculator(device="cpu").compute_range_aware_rapid(
        coords, refl, k_close=10, k_mid=7, k_far=5)
    assert feats.shape == (coords.shape[0], 10)                 # max(k)=10


def _zrot(theta):
    c, s = torch.cos(torch.tensor(theta)), torch.sin(torch.tensor(theta))
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _assert_invariant(f0, f1):
    """Invariance up to float32 precision on large-magnitude LiDAR coords.

    Rare neighbour-tie reorderings can differ on a handful of entries, so we
    assert a tiny mean difference and that ~all entries match closely, rather
    than exact equality.
    """
    diff = (f0 - f1).abs()
    assert diff.mean() < 1e-3
    assert (diff < 5e-3).float().mean() > 0.999


def test_standard_rapid_rigid_invariance_real(bin_path):
    """Standard RAPiD is invariant to a full rigid transform on real geometry."""
    coords, refl = _real_cloud(bin_path)
    calc = RAPiDCalculator(device="cpu")
    f0 = calc.compute_rapid_features(coords, refl, k=10)
    coords_t = coords @ _zrot(0.7).T + torch.tensor([3.0, -2.0, 1.0])
    f1 = calc.compute_rapid_features(coords_t, refl, k=10)
    _assert_invariant(f0, f1)


def test_r_rapid_rotation_invariance_real(bin_path):
    """R-RAPiD is invariant to rotation about the sensor axis on real geometry."""
    coords, refl = _real_cloud(bin_path)
    calc = RAPiDCalculator(device="cpu")
    f0 = calc.compute_r_rapid(coords, refl)
    f1 = calc.compute_r_rapid(coords @ _zrot(1.1).T, refl)
    _assert_invariant(f0, f1)


def test_reflectivity_changes_features_real(bin_path):
    """The 4D distance uses reflectivity: perturbing it changes the descriptor."""
    coords, refl = _real_cloud(bin_path)
    calc = RAPiDCalculator(device="cpu")
    f0 = calc.compute_rapid_features(coords, refl, k=10)
    f1 = calc.compute_rapid_features(coords, torch.rand_like(refl), k=10)
    assert not torch.allclose(f0, f1, atol=1e-2)


def test_config_profiles():
    for p in ("fast", "balanced", "accurate"):
        cfg = create_config(p)
        assert cfg.k_max == max(cfg.k_near, cfg.k_mid, cfg.k_far)
    assert create_config("balanced", k_mid=8).k_mid == 8

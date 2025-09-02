"""Getting started with RAPiD features.

    python examples/01_getting_started.py
"""
import torch

from rapid_seg import RAPiDCalculator
from rapid_seg.config import create_config


def main():
    # sample point cloud
    n_points = 1000
    coordinates = torch.randn(n_points, 3) * 10.0        # 3D coordinates
    reflectivity = torch.rand(n_points) * 0.8 + 0.1      # reflectivity values

    # configure + calculator
    config = create_config("balanced", k_mid=8)
    calculator = RAPiDCalculator(device=config.resolved_device())

    # standard RAPiD features
    k = config.get_k_for_standard_rapid()
    feats = calculator.compute_rapid_features(coordinates, reflectivity, k)
    print(f"RAPiD features shape: {tuple(feats.shape)}")
    print(f"Features range: [{feats.min():.3f}, {feats.max():.3f}]")

    # range-aware RAPiD features
    ra = calculator.compute_range_aware_rapid(
        coordinates, reflectivity, k_close=10, k_mid=7, k_far=5)
    print(f"Range-aware RAPiD shape: {tuple(ra.shape)}")


if __name__ == "__main__":
    main()

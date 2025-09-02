"""RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D LiDAR Segmentation.

Official implementation of

    Li Li, Hubert P. H. Shum, Toby P. Breckon.
    "RAPiD-Seg: Range-Aware Pointwise Distance Distribution Networks for 3D
    LiDAR Segmentation." ECCV 2024 (Oral). arXiv:2407.10159

Public API
----------
    from rapid_seg import RAPiDCalculator, PointCloudLoader, RapidSeg
    from rapid_seg.config import create_config

RAPiDCalculator computes RAPiD features (geometry + reflectivity, range-aware,
rigid-transform invariant); RapidSeg is the full segmentation network with the
double-nested VSA autoencoder, channel-wise fusion and the R-/C-RAPiD-Seg
variants; PointCloudLoader reads KITTI-format scans.
"""

from .core import RAPiDCalculator
from .data import PointCloudLoader
from .config import create_config, RAPiDConfig
from .models import RapidSeg

__version__ = "1.0.0"

__all__ = [
    "RAPiDCalculator",
    "PointCloudLoader",
    "RapidSeg",
    "create_config",
    "RAPiDConfig",
    "__version__",
]

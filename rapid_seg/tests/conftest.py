"""Shared fixtures — all tests run on the real SemanticKITTI seq-04 subset.

Data is discovered under ``<repo>/data``:
    * ``semantickitti_sp/04/*.pth``                     (preprocessed frames)
    * ``semantickitti_full/sequences/04/velodyne|labels`` (official .bin/.label)

Tests that need real data are skipped (not failed) when it is absent, so the
suite still runs in a data-less checkout.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
SP_DIR = DATA / "semantickitti_sp" / "04"
FULL_ROOT = DATA / "semantickitti_full"
FULL_VELO = FULL_ROOT / "sequences" / "04" / "velodyne"


def _sp_frames():
    return sorted(SP_DIR.glob("*.pth")) if SP_DIR.exists() else []


def _full_bins():
    return sorted(FULL_VELO.glob("*.bin")) if FULL_VELO.exists() else []


@pytest.fixture(scope="session")
def sp_pth_path():
    frames = _sp_frames()
    if not frames:
        pytest.skip("no real SemanticKITTI .pth subset under data/semantickitti_sp/04")
    return str(frames[0])


@pytest.fixture(scope="session")
def bin_path():
    bins = _full_bins()
    if not bins:
        pytest.skip("no real SemanticKITTI .bin under data/semantickitti_full")
    return str(bins[0])


@pytest.fixture(scope="session")
def full_root():
    if not _full_bins():
        pytest.skip("no real SemanticKITTI .bin/.label under data/semantickitti_full")
    return str(FULL_ROOT)


@pytest.fixture(scope="session")
def rapid_cfg():
    from rapid_seg.models.rapid_features import RAPiDConfig
    return RAPiDConfig(k_near=10, k_mid=7, k_far=5, num_beams=64)


@pytest.fixture(scope="session")
def real_frame(full_root, rapid_cfg):
    """One real frame via the official loader (subsampled for test speed)."""
    from rapid_seg.data import SemanticKITTIFull, FullKITTIConfig
    ds = SemanticKITTIFull(
        FullKITTIConfig(root=full_root, sequences=[4], max_points=4000, cache=False),
        rapid_cfg, with_crapid=True,
    )
    return ds[0]

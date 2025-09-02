"""Tests for the SemanticKITTI data loaders on the real seq-04 subset."""
import numpy as np
import torch

from rapid_seg.data import (
    PointCloudLoader, SemanticKITTIFull, FullKITTIConfig, SemanticKITTIDataset,
    KITTIConfig, load_kitti_pth, is_valid_frame, read_velodyne, read_label,
    remap_labels, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, SPLITS, LEARNING_MAP,
)
from rapid_seg.data.semantickitti_full import _build_map_lut


def test_read_velodyne_real(bin_path):
    scan = read_velodyne(bin_path)
    assert scan.ndim == 2 and scan.shape[1] == 4      # [x, y, z, remission]
    assert scan.shape[0] > 1000
    remission = scan[:, 3]
    assert np.isfinite(scan).all()
    assert 0.0 <= remission.min() and remission.max() <= 1.0


def test_read_label_learning_map_real(bin_path):
    label_path = bin_path.replace("velodyne", "labels").replace(".bin", ".label")
    lut = _build_map_lut(LEARNING_MAP)
    learning = read_label(label_path, lut)
    assert learning.min() >= 0 and learning.max() <= 19      # 0=ignore, 1..19 classes


def test_learning_map_matches_reference(sp_pth_path):
    """learning_map(raw ids) must reproduce the reference learning labels."""
    d = load_kitti_pth(sp_pth_path)
    raw = np.asarray(d["semantic_label_raw"])
    ref = np.asarray(d["semantic_label"])
    lut = _build_map_lut(LEARNING_MAP)
    mapped = lut[np.clip(raw, 0, len(lut) - 1)]
    assert (mapped == ref).mean() == 1.0


def test_full_dataset_frame_real(full_root, rapid_cfg):
    ds = SemanticKITTIFull(
        FullKITTIConfig(root=full_root, sequences=[4], max_points=3000, cache=False),
        rapid_cfg, with_crapid=True,
    )
    assert len(ds) > 0
    f = ds[0]
    m = f["xyz"].shape[0]
    assert f["xyz"].shape == (m, 3)
    assert f["intensity"].shape == (m, 2)
    assert f["rapid_r"].shape == (m, rapid_cfg.k_max)
    assert f["rapid_c"].shape == (m, rapid_cfg.k_max)
    labels = f["label"]
    valid = labels[labels != IGNORE_INDEX]
    assert valid.min() >= 0 and valid.max() < NUM_CLASSES
    assert torch.isfinite(f["rapid_r"]).all()


def test_pointcloud_loader_real(bin_path):
    loader = PointCloudLoader(max_points=2000)
    coords, reflectivity = loader.load_and_preprocess(bin_path)
    assert coords.shape[0] == 2000 and coords.shape[1] == 3
    assert reflectivity.shape == (2000,)


def test_sp_dataset_real(sp_pth_path, rapid_cfg):
    import os
    frame_files = [sp_pth_path]
    ds = SemanticKITTIDataset(
        KITTIConfig(root=os.path.dirname(os.path.dirname(sp_pth_path)),
                    max_points=3000, cache=False),
        rapid_cfg, frame_files=frame_files,
    )
    f = ds[0]
    assert f["rapid_r"].shape[1] == rapid_cfg.k_max
    assert is_valid_frame(sp_pth_path)


def test_remap_and_splits():
    raw = np.array([0, 1, 5, 19], dtype=np.int64)          # learning ids
    out = remap_labels(raw)
    assert out.tolist() == [IGNORE_INDEX, 0, 4, 18]         # 0->ignore, k->k-1
    assert set(SPLITS) == {"train", "val", "test"}
    assert 8 in SPLITS["val"]
    assert len(CLASS_NAMES) == NUM_CLASSES == 19

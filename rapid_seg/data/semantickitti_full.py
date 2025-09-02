"""Full official SemanticKITTI dataloader.

Reads the **raw, official** SemanticKITTI directory layout:

    <root>/
      sequences/
        00/ velodyne/000000.bin ...        # float32 [x, y, z, remission]
            labels/000000.label ...        # uint32: (sem & 0xFFFF) | (inst << 16)
            calib.txt  poses.txt  times.txt
        01/ ...
      semantic-kitti.yaml                   # optional; learning_map + splits

Raw semantic ids (0..259) are mapped to the 19 evaluated classes via the
official `learning_map` (0 = unlabeled/ignore, 1..19 = classes), then remapped
to contiguous `0..18` (ignore -> IGNORE_INDEX) for training — the exact same
label convention as `SemanticKITTIDataset` (the `.pth` subset), so the model,
losses and metrics are unchanged across the two loaders.

Sequence splits (official): train = 00-07,09,10 · val = 08 · test = 11-21.
Test frames have no `labels/` and are returned with all-ignore labels.
"""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ..models.rapid_features import RAPiDConfig, compute_r_rapid, compute_c_rapid
from .semantickitti import CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, remap_labels

# ---- official semantic-kitti.yaml learning_map (raw id -> 0..19) ----
LEARNING_MAP = {
    0: 0, 1: 0, 10: 1, 11: 2, 13: 5, 15: 3, 16: 5, 18: 4, 20: 5, 30: 6, 31: 7,
    32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 52: 0, 60: 9, 70: 15,
    71: 16, 72: 17, 80: 18, 81: 19, 99: 0, 252: 1, 253: 7, 254: 6, 255: 8,
    256: 5, 257: 5, 258: 4, 259: 5,
}

# official sequence splits
SPLITS = {
    "train": [0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
    "val": [8],
    "test": list(range(11, 22)),
}


def _build_map_lut(learning_map: dict) -> np.ndarray:
    lut = np.zeros(max(learning_map.keys()) + 1, dtype=np.int64)
    for k, v in learning_map.items():
        lut[k] = v
    return lut


def read_velodyne(path: str) -> np.ndarray:
    """Read a KITTI .bin scan -> (N, 4) float32 [x, y, z, remission]."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_label(path: str, lut: np.ndarray) -> np.ndarray:
    """Read a .label file -> learning semantic ids (0..19). Instance dropped."""
    raw = np.fromfile(path, dtype=np.uint32)
    sem = (raw & 0xFFFF).astype(np.int64)      # lower 16 bits = semantic
    sem = np.clip(sem, 0, len(lut) - 1)
    return lut[sem]                            # raw -> learning id (0..19)


def augment_xyz(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Standard LiDAR-seg augmentation (rotation is a no-op for RAPiD by design)."""
    theta = rng.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
    xyz = xyz @ rot.T
    if rng.random() < 0.5:
        xyz[:, 0] = -xyz[:, 0]
    if rng.random() < 0.5:
        xyz[:, 1] = -xyz[:, 1]
    xyz = xyz * rng.uniform(0.95, 1.05, size=3).astype(np.float32)
    xyz = xyz + rng.normal(0, 0.02, xyz.shape).astype(np.float32)
    return xyz


@dataclass
class FullKITTIConfig:
    root: str = "data/semantickitti_full"    # dir containing `sequences/`
    split: str = "train"                      # train|val|test, or a custom name
    sequences: Optional[Sequence[int]] = None  # override the split's sequences
    max_points: Optional[int] = None          # None = keep the full frame
    max_frames: Optional[int] = None          # cap frames (e.g. for a subset)
    augment: bool = False
    cache: bool = True
    seed: int = 0
    yaml_path: Optional[str] = None           # override learning_map / splits


class SemanticKITTIFull(Dataset):
    """Map-style dataset over the official SemanticKITTI format."""

    def __init__(self, cfg: FullKITTIConfig, rapid_cfg: RAPiDConfig, with_crapid: bool = False):
        self.cfg = cfg
        self.rapid_cfg = rapid_cfg
        self.with_crapid = with_crapid

        learning_map, splits = LEARNING_MAP, SPLITS
        if cfg.yaml_path and os.path.exists(cfg.yaml_path):
            import yaml
            with open(cfg.yaml_path) as f:
                y = yaml.safe_load(f)
            learning_map = {int(k): int(v) for k, v in y["learning_map"].items()}
            if "split" in y:
                splits = {k: list(v) for k, v in y["split"].items()}
        self.lut = _build_map_lut(learning_map)

        seqs = cfg.sequences if cfg.sequences is not None else splits[cfg.split]
        seq_root = os.path.join(cfg.root, "sequences")
        self.samples: List[tuple] = []          # (velodyne_path, label_path|None)
        for s in seqs:
            sd = os.path.join(seq_root, f"{int(s):02d}")
            bins = sorted(glob.glob(os.path.join(sd, "velodyne", "*.bin")))
            for b in bins:
                lp = b.replace("velodyne", "labels").replace(".bin", ".label")
                self.samples.append((b, lp if os.path.exists(lp) else None))
        if not self.samples:
            raise FileNotFoundError(
                f"no scans under {seq_root} for split={cfg.split} seqs={seqs}")
        if cfg.max_frames is not None:
            self.samples = self.samples[: cfg.max_frames]

        self.cache_dir = os.path.join(cfg.root, "_cache_full")
        if cfg.cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.samples)

    def _cache_key(self, bin_path: str) -> str:
        sig = f"{bin_path}|{self.cfg.max_points}|{self.cfg.seed}|{int(self.cfg.augment)}|" \
              f"{self.rapid_cfg.k_max}|{self.rapid_cfg.num_beams}|{int(self.with_crapid)}"
        return hashlib.md5(sig.encode()).hexdigest()[:16]

    def __getitem__(self, idx: int):
        bin_path, label_path = self.samples[idx]

        if self.cfg.cache and not self.cfg.augment:
            cf = os.path.join(self.cache_dir, self._cache_key(bin_path) + ".pt")
            if os.path.exists(cf):
                return torch.load(cf, weights_only=True)

        scan = read_velodyne(bin_path)
        xyz = scan[:, :3].astype(np.float32)
        refl = scan[:, 3].astype(np.float32)
        if label_path is not None:
            learning = read_label(label_path, self.lut)     # 0..19
            label = remap_labels(learning)                  # -1 / 0..18
        else:
            label = np.full(xyz.shape[0], IGNORE_INDEX, np.int64)

        rng = np.random.default_rng(
            self.cfg.seed + int(hashlib.md5(bin_path.encode()).hexdigest(), 16) % (10 ** 6))
        if self.cfg.max_points is not None and xyz.shape[0] > self.cfg.max_points:
            sel = rng.choice(xyz.shape[0], self.cfg.max_points, replace=False)
            xyz, refl, label = xyz[sel], refl[sel], label[sel]
        if self.cfg.augment:
            xyz = augment_xyz(xyz, rng)

        xyz_t = torch.from_numpy(np.ascontiguousarray(xyz))
        reflect = torch.from_numpy(np.ascontiguousarray(refl))
        label_t = torch.from_numpy(np.ascontiguousarray(label))
        intensity = torch.stack([reflect, reflect], dim=1)   # KITTI: single remission channel

        rapid_r = compute_r_rapid(xyz_t, reflect, self.rapid_cfg)
        frame = {"xyz": xyz_t, "intensity": intensity, "reflect": reflect,
                 "label": label_t, "rapid_r": rapid_r}
        if self.with_crapid:
            frame["rapid_c"] = compute_c_rapid(xyz_t, reflect, label_t, self.rapid_cfg, IGNORE_INDEX)

        if self.cfg.cache and not self.cfg.augment:
            torch.save(frame, cf)
        return frame


def collate_single(batch):
    """batch_size=1 collate: return the single frame dict unchanged.

    The RAPiD-Seg network voxelises one frame per step (standard for LiDAR
    segmentation). Multi-frame batching would require batch-aware voxelisation
    (a batch index into `voxelize` and the ME coordinate batch column); the
    single-frame path is what the training script uses.
    """
    assert len(batch) == 1, "use batch_size=1 (single-frame voxelisation)"
    return batch[0]


def make_dataloader(cfg: FullKITTIConfig, rapid_cfg: RAPiDConfig, with_crapid=False,
                    shuffle=None, num_workers=0) -> DataLoader:
    """Build a DataLoader over the official SemanticKITTI format (batch_size=1)."""
    ds = SemanticKITTIFull(cfg, rapid_cfg, with_crapid)
    if shuffle is None:
        shuffle = cfg.split == "train"
    return DataLoader(ds, batch_size=1, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_single)

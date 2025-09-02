"""Real SemanticKITTI subset dataset.

Loads preprocessed SemanticKITTI `.pth` frames (sequence 04, from the
`larryshaw0079/SemanticKITTI-SP` HuggingFace dataset) that contain real
coordinates, remission (reflectivity) and per-point semantic labels, and turns
each frame into the tensors RAPiD-Seg consumes.

Labels in the source are the standard SemanticKITTI *learning* labels:
    0        -> unlabeled / ignore
    1 .. 19  -> the 19 evaluated classes
We remap to contiguous 0..18 (ignore -> IGNORE_INDEX) for training.
"""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from ..models.rapid_features import RAPiDConfig, compute_r_rapid, compute_c_rapid
from .kitti_pth import load_kitti_pth

# 19 SemanticKITTI evaluation classes (learning ids 1..19 -> index 0..18)
CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
    "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
]
NUM_CLASSES = len(CLASS_NAMES)   # 19
IGNORE_INDEX = -1


def remap_labels(raw: np.ndarray) -> np.ndarray:
    """learning label (0 ignore, 1..19) -> contiguous (IGNORE, 0..18)."""
    out = raw.astype(np.int64) - 1
    out[raw == 0] = IGNORE_INDEX
    return out


@dataclass
class KITTIConfig:
    root: str = "data/semantickitti_sp"
    sequence: str = "04"
    max_points: int = 16000       # subsample per frame (CPU/MPS-friendly)
    num_frames: Optional[int] = None
    seed: int = 0
    cache: bool = True


class SemanticKITTIDataset(Dataset):
    """Real SemanticKITTI frames -> {xyz, intensity, reflect, label, rapid_r[, rapid_c]}."""

    def __init__(
        self,
        cfg: KITTIConfig,
        rapid_cfg: RAPiDConfig,
        frame_files: Optional[List[str]] = None,
        with_crapid: bool = False,
    ):
        self.cfg = cfg
        self.rapid_cfg = rapid_cfg
        self.with_crapid = with_crapid
        if frame_files is None:
            pat = os.path.join(cfg.root, cfg.sequence, "*.pth")
            frame_files = sorted(glob.glob(pat))
            if cfg.num_frames is not None:
                frame_files = frame_files[: cfg.num_frames]
        if not frame_files:
            raise FileNotFoundError(f"no .pth frames found under {cfg.root}/{cfg.sequence}")
        self.frame_files = frame_files
        self.cache_dir = os.path.join(cfg.root, "_cache")
        if cfg.cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.frames: List[Dict[str, torch.Tensor]] = [
            self._prepare(f) for f in self.frame_files
        ]

    # ------------------------------------------------------------------
    def _cache_key(self, path: str) -> str:
        sig = f"{path}|{self.cfg.max_points}|{self.cfg.seed}|{self.rapid_cfg.k_max}|" \
              f"{self.rapid_cfg.num_beams}|{int(self.with_crapid)}"
        return hashlib.md5(sig.encode()).hexdigest()[:16]

    def _prepare(self, path: str) -> Dict[str, torch.Tensor]:
        if self.cfg.cache:
            cf = os.path.join(self.cache_dir, self._cache_key(path) + ".pt")
            if os.path.exists(cf):
                return torch.load(cf, weights_only=True)

        d = load_kitti_pth(path)
        xyz = np.asarray(d["coord"], np.float32)
        refl = np.asarray(d["remission"], np.float32)
        label = remap_labels(np.asarray(d["semantic_label"]))

        # deterministic subsample for CPU/MPS feasibility
        n = xyz.shape[0]
        if n > self.cfg.max_points:
            rng = np.random.default_rng(
                self.cfg.seed + int(hashlib.md5(path.encode()).hexdigest(), 16) % (10**6)
            )
            sel = rng.choice(n, self.cfg.max_points, replace=False)
            xyz, refl, label = xyz[sel], refl[sel], label[sel]

        xyz = torch.from_numpy(xyz)
        reflect = torch.from_numpy(refl)
        label = torch.from_numpy(label)
        # F_I = (intensity, reflectivity); source gives one remission channel,
        # used for both (KITTI provides a single remission value per point).
        intensity = torch.stack([reflect, reflect], dim=1)

        rapid_r = compute_r_rapid(xyz, reflect, self.rapid_cfg)
        frame = {
            "xyz": xyz, "intensity": intensity, "reflect": reflect,
            "label": label, "rapid_r": rapid_r,
        }
        if self.with_crapid:
            frame["rapid_c"] = compute_c_rapid(xyz, reflect, label, self.rapid_cfg, IGNORE_INDEX)

        if self.cfg.cache:
            torch.save(frame, cf)
        return frame

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, i):
        return self.frames[i]


def split_frames(root: str, sequence: str, n_train: int, n_val: int, seed: int = 0):
    """Deterministic train/val split over the available real frames."""
    files = sorted(glob.glob(os.path.join(root, sequence, "*.pth")))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    train = [files[i] for i in idx[:n_train]]
    val = [files[i] for i in idx[n_train:n_train + n_val]]
    return train, val

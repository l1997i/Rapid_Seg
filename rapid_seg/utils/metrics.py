"""Segmentation metrics: per-class IoU and mIoU (paper Sec. 6 protocol)."""
from __future__ import annotations

import torch


class IoUMeter:
    """Accumulates TP/FP/FN per class and reports IoU_i and mIoU."""

    def __init__(self, num_classes: int, ignore_index: int = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fp = torch.zeros(self.num_classes, dtype=torch.long)
        self.fn = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor):
        pred = pred.detach().cpu()
        target = target.detach().cpu()
        valid = target != self.ignore_index
        pred, target = pred[valid], target[valid]
        for c in range(self.num_classes):
            pc, tc = pred == c, target == c
            self.tp[c] += int((pc & tc).sum())
            self.fp[c] += int((pc & ~tc).sum())
            self.fn[c] += int((~pc & tc).sum())

    def iou_per_class(self):
        denom = (self.tp + self.fp + self.fn).clamp_min(1)
        iou = self.tp.float() / denom.float()
        present = (self.tp + self.fn) > 0     # classes that appear in targets
        return iou, present

    def miou(self) -> float:
        iou, present = self.iou_per_class()
        if present.sum() == 0:
            return 0.0
        return float(iou[present].mean())

    def accuracy(self) -> float:
        total = int((self.tp + self.fn).sum())
        return float(self.tp.sum()) / max(total, 1)

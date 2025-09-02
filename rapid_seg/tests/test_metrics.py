"""Tests for the mIoU / accuracy meter, driven by real labels."""
import torch

from rapid_seg.utils.metrics import IoUMeter
from rapid_seg.data import NUM_CLASSES, IGNORE_INDEX


def test_perfect_prediction_real_labels(real_frame):
    labels = real_frame["label"]
    meter = IoUMeter(NUM_CLASSES, IGNORE_INDEX)
    meter.update(labels.clone(), labels)                     # perfect prediction
    assert abs(meter.miou() - 1.0) < 1e-6
    assert abs(meter.accuracy() - 1.0) < 1e-6


def test_metrics_bounds_real_labels(real_frame):
    labels = real_frame["label"]
    torch.manual_seed(0)
    pred = torch.randint(0, NUM_CLASSES, labels.shape)
    meter = IoUMeter(NUM_CLASSES, IGNORE_INDEX)
    meter.update(pred, labels)
    assert 0.0 <= meter.miou() <= 1.0
    assert 0.0 <= meter.accuracy() <= 1.0


def test_ignore_index_excluded():
    meter = IoUMeter(num_classes=3, ignore_index=IGNORE_INDEX)
    target = torch.tensor([0, 1, 2, IGNORE_INDEX])
    pred = torch.tensor([0, 1, 2, 0])                        # ignored position wrong
    meter.update(pred, target)
    assert abs(meter.miou() - 1.0) < 1e-6                    # ignored point not counted

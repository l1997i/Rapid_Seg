"""Tests for the RAPiD-Seg network on a real SemanticKITTI frame."""
import torch

from rapid_seg import RapidSeg
from rapid_seg.models import ae_total_loss, segmentation_loss
from rapid_seg.data import NUM_CLASSES, IGNORE_INDEX


def _net(variant, rapid_dim):
    return RapidSeg(NUM_CLASSES, rapid_dim=rapid_dim, variant=variant, voxel_size=0.5,
                    vsa_dim=16, ae_dim=32, ae_latent_dim=8, vsa_latents=4,
                    backbone_dim=48, backbone_blocks=2, backbone="voxel")


def test_forward_backward_r_variant_real(real_frame):
    xyz, inten, rr, lbl = (real_frame["xyz"], real_frame["intensity"],
                           real_frame["rapid_r"], real_frame["label"])
    net = _net("R", rr.shape[1])
    logits, aux = net(xyz, inten, rr)
    assert logits.shape == (xyz.shape[0], NUM_CLASSES)

    ae = aux["ae_r"]
    l_ae, _ = ae_total_loss(rr, ae["recon"], ae["h_point"], lbl, lam=0.1, ignore_index=IGNORE_INDEX)
    l_seg, _ = segmentation_loss(logits, lbl, ignore_index=IGNORE_INDEX)
    (l_seg + l_ae).backward()
    gnorm = sum(p.grad.norm().item() for p in net.parameters() if p.grad is not None)
    assert gnorm > 0


def test_forward_c_variant_real(real_frame):
    xyz, inten, rr = real_frame["xyz"], real_frame["intensity"], real_frame["rapid_r"]
    rc = real_frame["rapid_c"]
    net = _net("C", rr.shape[1])
    logits, _ = net(xyz, inten, rr, rapid_c=rc)
    assert logits.shape == (xyz.shape[0], NUM_CLASSES)


def test_training_step_decreases_loss_real(real_frame):
    """Overfitting a single real frame for a few steps must reduce the loss."""
    torch.manual_seed(0)
    xyz, inten, rr, lbl = (real_frame["xyz"], real_frame["intensity"],
                           real_frame["rapid_r"], real_frame["label"])
    net = _net("R", rr.shape[1])
    opt = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.9)

    losses = []
    for _ in range(15):
        logits, _ = net(xyz, inten, rr)
        loss, _ = segmentation_loss(logits, lbl, ignore_index=IGNORE_INDEX)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert min(losses[-3:]) < losses[0]          # loss decreased on real data

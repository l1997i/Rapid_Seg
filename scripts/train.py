"""Train RAPiD-Seg on SemanticKITTI.

Training recipe (Sec. 6, Implementation Details):
  * SGD optimiser, initial lr 1e-3
  * 2-epoch linear warmup + cosine schedule for the remainder
  * two stages: (1) RAPiD-AE pretraining (recon + class-aware contrastive),
    (2) full-network segmentation training initialised from the pretrained AE,
    with the AE objective kept as an auxiliary regulariser.
  * loss:  L_seg (CE + Lovasz)  +  gamma * (L_recon + lambda * L_contr)

Reads either the official `.bin`/`.label` layout (--dataset full) or the
preprocessed `.pth` frames (--dataset sp). On an NVIDIA GPU with MinkowskiEngine
the MinkUNet34 backbone is used automatically.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rapid_seg.utils import get_device
from rapid_seg.utils.metrics import IoUMeter
from rapid_seg.models import (
    RAPiDConfig, RapidSeg, voxelize, ae_total_loss, segmentation_loss,
)
from rapid_seg.data import (
    SemanticKITTIDataset, KITTIConfig, split_frames, NUM_CLASSES, IGNORE_INDEX,
    SemanticKITTIFull, FullKITTIConfig, make_dataloader,
)


def set_seed(s):
    random.seed(s); torch.manual_seed(s)
    try:
        import numpy as np; np.random.seed(s)
    except Exception:
        pass


def warmup_cosine(step_epoch, total, warmup=2):
    """LR multiplier: linear warmup then cosine decay to ~0."""
    if step_epoch < warmup:
        return (step_epoch + 1) / warmup
    p = (step_epoch - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def to_dev(frame, dev, keys):
    return {k: frame[k].to(dev) for k in keys if k in frame}


def source_len(source):
    return len(source)


def epoch_iter(source):
    """Yield frame dicts for one epoch.

    Accepts an in-memory list (shuffled per epoch) or a DataLoader (shuffling
    handled by the loader), so training scales from a small in-memory subset to
    the full SemanticKITTI split streamed from disk.
    """
    if isinstance(source, list):
        order = list(range(len(source)))
        random.shuffle(order)
        for i in order:
            yield source[i]
    else:
        yield from source


# --------------------------------------------------------------------------- #
def pretrain_ae(net, frames, dev, epochs, lr, lam, gamma_unused, log):
    """Stage 1: RAPiD-AE reconstruction + class-aware contrastive."""
    ae = net.rapid_ae
    opt = torch.optim.SGD(ae.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    print(f"\n[Stage 1] AE pretraining  ({epochs} epochs, {source_len(frames)} frames)")
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr * warmup_cosine(ep, epochs)
        tot = {"loss": 0.0, "recon": 0.0, "contr": 0.0}
        ae.train()
        for f in epoch_iter(frames):
            xyz = f["xyz"].to(dev); rr = f["rapid_r"].to(dev); lbl = f["label"].to(dev)
            vidx, _, nv = voxelize(xyz, net.voxel_size)
            out = ae(rr, vidx, nv)
            loss, stats = ae_total_loss(rr, out["recon"], out["h_point"], lbl,
                                        lam=lam, ignore_index=IGNORE_INDEX)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 5.0)
            opt.step()
            tot["loss"] += loss.item()
            tot["recon"] += float(stats["recon"]); tot["contr"] += float(stats["contr"])
        n = source_len(frames)
        rec = {k: v / n for k, v in tot.items()}
        rec["epoch"] = ep; rec["lr"] = opt.param_groups[0]["lr"]
        log.append({"stage": "ae", **rec})
        print(f"  ae ep {ep:02d}  L={rec['loss']:.4f}  recon={rec['recon']:.4f}  "
              f"contr={rec['contr']:.4f}  lr={rec['lr']:.2e}")


@torch.no_grad()
def evaluate(net, frames, dev, variant):
    net.eval()
    meter = IoUMeter(NUM_CLASSES, IGNORE_INDEX)
    tot = 0.0
    for f in frames:
        xyz, inten, rr, lbl = (f["xyz"].to(dev), f["intensity"].to(dev),
                               f["rapid_r"].to(dev), f["label"].to(dev))
        rc = f["rapid_c"].to(dev) if variant == "C" and "rapid_c" in f else None
        logits, _ = net(xyz, inten, rr, rapid_c=rc)
        l, _ = segmentation_loss(logits, lbl, ignore_index=IGNORE_INDEX)
        tot += l.item()
        meter.update(logits.argmax(1), lbl)
    return tot / max(source_len(frames), 1), meter.miou(), meter.accuracy()


def train_seg(net, train_f, val_f, dev, variant, epochs, lr, lam, gamma, log):
    """Stage 2: full-network segmentation training (seg + auxiliary AE loss)."""
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    print(f"\n[Stage 2] {variant}-RAPiD-Seg training  ({epochs} epochs, "
          f"{source_len(train_f)} train / {source_len(val_f)} val frames)")
    best = 0.0
    for ep in range(epochs):
        for g in opt.param_groups:
            g["lr"] = lr * warmup_cosine(ep, epochs)
        net.train()
        tot = {"loss": 0.0, "seg": 0.0, "ae": 0.0}
        for f in epoch_iter(train_f):
            xyz, inten, rr, lbl = (f["xyz"].to(dev), f["intensity"].to(dev),
                                   f["rapid_r"].to(dev), f["label"].to(dev))
            rc = f["rapid_c"].to(dev) if variant == "C" and "rapid_c" in f else None
            logits, aux = net(xyz, inten, rr, rapid_c=rc)
            l_seg, _ = segmentation_loss(logits, lbl, ignore_index=IGNORE_INDEX)
            ae = aux["ae_r"]
            l_ae, _ = ae_total_loss(rr, ae["recon"], ae["h_point"], lbl,
                                    lam=lam, ignore_index=IGNORE_INDEX)
            loss = l_seg + gamma * l_ae
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            tot["loss"] += loss.item(); tot["seg"] += l_seg.item(); tot["ae"] += l_ae.item()
        n = source_len(train_f)
        tr = {k: v / n for k, v in tot.items()}
        val_loss, miou, acc = evaluate(net, val_f, dev, variant)
        best = max(best, miou)
        rec = {"stage": "seg", "epoch": ep, "lr": opt.param_groups[0]["lr"],
               "train_loss": tr["loss"], "train_seg": tr["seg"], "train_ae": tr["ae"],
               "val_loss": val_loss, "val_miou": miou, "val_acc": acc}
        log.append(rec)
        print(f"  seg ep {ep:02d}  train L={tr['loss']:.4f} (seg={tr['seg']:.3f} "
              f"ae={tr['ae']:.3f})  val L={val_loss:.4f}  mIoU={miou*100:.2f}  "
              f"acc={acc*100:.2f}  lr={rec['lr']:.2e}")
    return best


def build_sources(args, rapid_cfg, with_c):
    """Return (train_source, val_source), each an in-memory list or a DataLoader.

    --dataset sp   : preprocessed `.pth` subset (in-memory lists)
    --dataset full : official `.bin`/`.label` layout; streamed via DataLoader
                     when using whole splits, or materialised when capped by
                     --n_train / --n_val (handy for quick runs on a few frames).
    """
    if args.dataset == "sp":
        n_tr = args.n_train if args.n_train is not None else 60
        n_va = args.n_val if args.n_val is not None else 12
        train_files, val_files = split_frames(args.data_root, "04", n_tr, n_va, args.seed)
        kc = KITTIConfig(root=args.data_root, max_points=args.max_points or 16000, seed=args.seed)
        tr = SemanticKITTIDataset(kc, rapid_cfg, frame_files=train_files, with_crapid=with_c)
        va = SemanticKITTIDataset(kc, rapid_cfg, frame_files=val_files, with_crapid=with_c)
        return tr.frames, va.frames

    # --- official SemanticKITTI format ---
    if args.n_train is not None:
        # capped subset -> materialise a few frames in memory (quick runs)
        ds = SemanticKITTIFull(FullKITTIConfig(root=args.data_root, sequences=args.sequences,
                                               max_points=args.max_points, seed=args.seed),
                               rapid_cfg, with_crapid=with_c)
        n = min(len(ds), args.n_train + (args.n_val or 0))
        idx = list(range(n)); random.Random(args.seed).shuffle(idx)
        tr_idx, va_idx = idx[:args.n_train], idx[args.n_train:n]
        print(f"[data] materialising {len(tr_idx)}+{len(va_idx)} frames from official "
              f".bin/.label under {args.data_root} ...")
        return [ds[i] for i in tr_idx], [ds[i] for i in va_idx]

    # whole official splits -> stream from disk (per-frame RAPiD cache on first epoch)
    train_seqs = args.sequences if args.sequences else None
    train_loader = make_dataloader(
        FullKITTIConfig(root=args.data_root, split="train", sequences=train_seqs,
                        max_points=args.max_points, augment=args.augment, seed=args.seed),
        rapid_cfg, with_crapid=with_c, shuffle=True, num_workers=args.num_workers)
    val_loader = make_dataloader(
        FullKITTIConfig(root=args.data_root, split="val",
                        max_points=args.max_points, seed=args.seed),
        rapid_cfg, with_crapid=with_c, shuffle=False, num_workers=args.num_workers)
    print(f"[data] streaming official splits from {args.data_root} "
          f"(train={len(train_loader)} · val={len(val_loader)} frames)")
    return train_loader, val_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sp", "full"], default="sp",
                    help="sp=preprocessed .pth subset; full=official .bin/.label layout")
    ap.add_argument("--data_root", default=None,
                    help="dataset root (default: data/semantickitti_sp or data/semantickitti_full)")
    ap.add_argument("--sequences", type=int, nargs="*", default=None,
                    help="restrict --dataset full to these sequences (default: official split)")
    ap.add_argument("--variant", choices=["R", "C"], default="R")
    ap.add_argument("--backbone", choices=["auto", "minkunet", "voxel"], default="auto",
                    help="auto=MinkowskiUNet34 on NVIDIA GPU, else light voxel backbone")
    ap.add_argument("--n_train", type=int, default=None,
                    help="cap #train frames (default: whole split; streamed for --dataset full)")
    ap.add_argument("--n_val", type=int, default=None, help="cap #val frames")
    ap.add_argument("--max_points", type=int, default=None,
                    help="subsample points per frame (default: keep all)")
    ap.add_argument("--augment", action="store_true", help="enable train-time augmentation")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--ae_epochs", type=int, default=100)
    ap.add_argument("--seg_epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=0.1, help="lambda for contrastive")
    ap.add_argument("--gamma", type=float, default=0.5, help="aux AE loss weight in seg stage")
    ap.add_argument("--voxel_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/run")
    ap.add_argument("--device", default="cuda", help="cuda(default)/cpu")
    args = ap.parse_args()
    if args.data_root is None:
        args.data_root = "data/semantickitti_sp" if args.dataset == "sp" else "data/semantickitti_full"

    set_seed(args.seed)
    dev = get_device(prefer=args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"[device] {dev}  |  variant={args.variant}")

    rapid_cfg = RAPiDConfig(k_near=10, k_mid=7, k_far=5, num_beams=64)
    with_c = args.variant == "C"
    t0 = time.time()
    train_src, val_src = build_sources(args, rapid_cfg, with_c)
    print(f"[data] {args.dataset}: {source_len(train_src)} train + {source_len(val_src)} val "
          f"frames ready in {time.time()-t0:.1f}s")

    net = RapidSeg(NUM_CLASSES, rapid_dim=rapid_cfg.k_max, variant=args.variant,
                   voxel_size=args.voxel_size, vsa_dim=32, ae_dim=64, ae_latent_dim=16,
                   vsa_latents=8, backbone_dim=96, backbone_blocks=4,
                   backbone=args.backbone).to(dev)
    print(f"[model] {args.variant}-RAPiD-Seg params="
          f"{sum(p.numel() for p in net.parameters())/1e6:.3f}M")

    log = []
    t0 = time.time()
    pretrain_ae(net, train_src, dev, args.ae_epochs, args.lr, args.lam, args.gamma, log)
    best = train_seg(net, train_src, val_src, dev, args.variant,
                     args.seg_epochs, args.lr, args.lam, args.gamma, log)
    dur = time.time() - t0

    with open(os.path.join(args.out, "metrics.json"), "w") as fp:
        json.dump({"args": vars(args), "log": log, "best_val_miou": best,
                   "train_time_s": dur, "device": str(dev)}, fp, indent=2)
    torch.save(net.state_dict(), os.path.join(args.out, "model.pt"))
    print(f"\n[done] best val mIoU={best*100:.2f}  |  {dur:.1f}s  |  saved -> {args.out}")


if __name__ == "__main__":
    main()

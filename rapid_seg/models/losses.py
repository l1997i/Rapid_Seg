"""Loss functions.

  * class_aware_contrastive_loss  -- Eq. (9)   L_contr
  * reconstruction_loss (MSE)     -- Eq. (10)  L_recon
      L_total = L_recon + lambda * L_contr        (Sec. 4)
  * lovasz_softmax                -- standard LiDAR-seg auxiliary loss

The contrastive loss pulls together embeddings of the same semantic class and
pushes apart different classes, using per-point nearest positive/negative
neighbours (P(i)/N(i)) and margins alpha_p / alpha_n.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(g: torch.Tensor, g_hat: torch.Tensor) -> torch.Tensor:
    """MSE reconstruction loss, Eq. (10)."""
    return F.mse_loss(g_hat, g)


def class_aware_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    alpha_p: float = 0.9,
    alpha_n: float = 0.5,
    ignore_index: int = -1,
    max_points: int = 4096,
) -> torch.Tensor:
    """Class-aware contrastive loss (Eq. 9).

    For each anchor i, find its nearest same-class point P(i) and nearest
    different-class point N(i) (in embedding space), then:

        L = mean_i [ ReLU(alpha_p - sim(H_i, H_{P(i)}))
                     + ReLU(sim(H_i, H_{N(i)}) - alpha_n) ]

    sim(.) is cosine similarity. To bound the O(n^2) similarity matrix we
    subsample up to `max_points` valid anchors per call.
    """
    device = embeddings.device
    valid = labels != ignore_index
    idx = valid.nonzero(as_tuple=True)[0]
    if idx.numel() < 2:
        return embeddings.new_zeros(())
    if idx.numel() > max_points:
        perm = torch.randperm(idx.numel(), device=device)[:max_points]
        idx = idx[perm]

    emb = F.normalize(embeddings[idx], dim=1)              # (n, d')
    lab = labels[idx]                                      # (n,)
    sim = emb @ emb.t()                                    # (n, n) cosine sim
    n = sim.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=device)

    same = (lab[:, None] == lab[None, :]) & ~eye           # positive mask
    diff = lab[:, None] != lab[None, :]                    # negative mask

    # nearest positive = highest similarity among same-class (else skip anchor)
    pos_sim = sim.masked_fill(~same, float("-inf")).max(dim=1).values
    has_pos = torch.isfinite(pos_sim)
    # nearest negative = highest similarity among different-class (hardest neg)
    neg_sim = sim.masked_fill(~diff, float("-inf")).max(dim=1).values
    has_neg = torch.isfinite(neg_sim)

    loss = embeddings.new_zeros(())
    denom = 0
    if has_pos.any():
        loss = loss + F.relu(alpha_p - pos_sim[has_pos]).sum()
        denom += int(has_pos.sum())
    if has_neg.any():
        loss = loss + F.relu(neg_sim[has_neg] - alpha_n).sum()
        denom += int(has_neg.sum())
    return loss / max(denom, 1)


def ae_total_loss(g, g_hat, embeddings, labels, lam=0.1, alpha_p=0.9, alpha_n=0.5, ignore_index=-1):
    """L_total = L_recon + lambda * L_contr (Sec. 4)."""
    l_recon = reconstruction_loss(g, g_hat)
    l_contr = class_aware_contrastive_loss(
        embeddings, labels, alpha_p, alpha_n, ignore_index
    )
    return l_recon + lam * l_contr, {"recon": l_recon.detach(), "contr": l_contr.detach()}


# ---------------------------------------------------------------------------
# Lovasz-softmax (Berman et al. 2018) -- standard auxiliary loss for LiDAR seg
# ---------------------------------------------------------------------------
def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax(probas: torch.Tensor, labels: torch.Tensor, ignore_index: int = -1) -> torch.Tensor:
    """Lovasz-softmax over flat (N, C) probabilities and (N,) labels."""
    if probas.numel() == 0:
        return probas.new_zeros(())
    C = probas.shape[1]
    losses = []
    valid = labels != ignore_index
    probas, labels = probas[valid], labels[valid]
    if probas.numel() == 0:
        return probas.new_zeros(())
    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        grad = _lovasz_grad(fg[perm])
        losses.append(torch.dot(errors_sorted, grad))
    if not losses:
        return probas.new_zeros(())
    return torch.stack(losses).mean()


def segmentation_loss(logits, labels, class_weights=None, ignore_index=-1, lovasz_weight=1.0):
    """Weighted CE + Lovasz-softmax, the common LiDAR-seg objective."""
    ce = F.cross_entropy(logits, labels, weight=class_weights, ignore_index=ignore_index)
    lov = lovasz_softmax(F.softmax(logits, dim=1), labels, ignore_index)
    return ce + lovasz_weight * lov, {"ce": ce.detach(), "lovasz": lov.detach()}

"""Losses for optional contrastive graph representation experiments."""

import torch
import torch.nn.functional as F


def info_nce_loss(q, k, queue, temperature: float):
    """Compute one-positive, queue-negative InfoNCE cross entropy."""
    q = F.normalize(q.float(), dim=1)
    k = F.normalize(k.float(), dim=1)
    l_pos = torch.einsum("bc,bc->b", q, k).unsqueeze(-1)
    l_neg = torch.einsum("bc,ck->bk", q, queue)
    logits = torch.cat([l_pos, l_neg], dim=1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)

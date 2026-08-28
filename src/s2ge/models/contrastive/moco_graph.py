"""Optional momentum contrast for single-token graph embeddings."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from s2ge.models.contrastive.losses import info_nce_loss


class MoCoGraph(nn.Module):
    """Maintain momentum graph encoders and a FIFO negative queue."""
    def __init__(self, encoder, projector, embed_dim: int, queue_size: int = 65536, momentum: float = 0.999, temperature: float = 0.07):
        super().__init__()
        self.momentum_encoder = copy.deepcopy(encoder)
        self.momentum_projector = copy.deepcopy(projector)
        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        for p in self.momentum_encoder.parameters():
            p.requires_grad = False
        for p in self.momentum_projector.parameters():
            p.requires_grad = False
        self.register_buffer("queue", F.normalize(torch.randn(embed_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def update_momentum(self, encoder, projector):
        for param_q, param_k in zip(encoder.parameters(), self.momentum_encoder.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1.0 - self.momentum)
        for param_q, param_k in zip(projector.parameters(), self.momentum_projector.parameters()):
            param_k.data.mul_(self.momentum).add_(param_q.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        if batch_size == 0:
            return
        if batch_size > self.queue_size:
            keys = keys[-self.queue_size :]
            batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        end_ptr = ptr + batch_size
        if end_ptr <= self.queue_size:
            self.queue[:, ptr:end_ptr] = keys.transpose(0, 1)
        else:
            first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:first].transpose(0, 1)
            self.queue[:, : end_ptr - self.queue_size] = keys[first:].transpose(0, 1)
        self.queue_ptr[0] = end_ptr % self.queue_size

    def forward(self, q_embeds, k_embeds):
        loss = info_nce_loss(q_embeds, k_embeds, self.queue.clone().detach(), self.temperature)
        self.dequeue_and_enqueue(F.normalize(k_embeds.float(), dim=1))
        return loss

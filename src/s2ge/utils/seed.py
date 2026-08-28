"""Shared pseudo-random seed initialization."""

import random

import numpy as np
import torch


def seed_everything(seed: int = 0):
    """Seed Python, NumPy, and all visible PyTorch CUDA generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

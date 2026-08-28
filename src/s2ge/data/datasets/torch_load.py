"""Trusted local PyG graph loading across PyTorch serialization defaults."""

from __future__ import annotations

from pathlib import Path

import torch


def load_graph_file(path: str | Path):
    """Load a trusted local PyG graph file across PyTorch 2.6+ defaults."""

    # Processed graph files are created locally by the preprocessing pipeline.
    # Full object loading is required for PyG Data objects.
    return torch.load(path, weights_only=False)

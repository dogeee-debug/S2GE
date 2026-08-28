"""LLaGA-compatible projector construction."""

import re

import torch.nn as nn


def build_llaga_graph_projector(mm_hidden_size, hidden_size, projector_type="linear"):
    """Build a linear or MLP projector from a released LLaGA type name."""
    if projector_type == "linear":
        return nn.Linear(mm_hidden_size, hidden_size)

    mlp_match = re.match(r"^(\d+)-layer-mlp$", projector_type)
    if mlp_match:
        depth = int(mlp_match.group(1))
        layers = [nn.Linear(mm_hidden_size, hidden_size)]
        for _ in range(1, depth):
            layers.extend([nn.GELU(), nn.Linear(hidden_size, hidden_size)])
        return nn.Sequential(*layers)

    raise ValueError(f"Unknown LLaGA projector type: {projector_type}")


__all__ = ["build_llaga_graph_projector"]

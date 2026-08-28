"""Factory for graph-to-decoder interface projectors."""

import torch.nn as nn

from s2ge.models.projectors.llaga_projector import build_llaga_graph_projector


def build_projector(args, llm_embed_dim: int):
    """Build the configured projector from GNN width to decoder width."""
    if getattr(args, 'graph_projector_mode', 'legacy') == 'llaga':
        return build_llaga_graph_projector(
            mm_hidden_size=args.gnn_hidden_dim,
            hidden_size=llm_embed_dim,
            projector_type=getattr(args, 'llaga_projector_type', '2-layer-mlp'),
        )
    return nn.Sequential(
        nn.Linear(args.gnn_hidden_dim, 2048),
        nn.Sigmoid(),
        nn.Linear(2048, llm_embed_dim),
    )

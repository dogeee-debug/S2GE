"""Optional graph augmentations for the disabled-by-default MoCo path."""

import torch
from torch_geometric.utils import dropout_edge


def augment_dropedge(graphs, drop_prob: float):
    """Return a cloned graph batch with randomly removed edges."""
    if drop_prob <= 0 or graphs.edge_index.numel() == 0:
        return graphs
    aug_graphs = graphs.clone()
    edge_index, edge_mask = dropout_edge(aug_graphs.edge_index, p=drop_prob, training=True)
    aug_graphs.edge_index = edge_index
    if getattr(aug_graphs, 'edge_attr', None) is not None:
        aug_graphs.edge_attr = aug_graphs.edge_attr[edge_mask]
    return aug_graphs


def augment_feature_mask(graphs, mask_prob: float):
    """Return a cloned graph batch with randomly masked feature entries."""
    if mask_prob <= 0:
        return graphs
    aug_graphs = graphs.clone()
    if getattr(aug_graphs, 'x', None) is None:
        return aug_graphs
    keep_mask = torch.rand_like(aug_graphs.x, dtype=torch.float32) > mask_prob
    aug_graphs.x = aug_graphs.x * keep_mask.to(dtype=aug_graphs.x.dtype)
    return aug_graphs


def build_views(graphs, training: bool, enabled: bool, query_dropedge_rate: float, key_view_mode: str, key_dropedge_rate: float, key_feature_mask_rate: float):
    """Build query/key views only when contrastive training is enabled."""
    if not training or not enabled:
        return graphs, graphs
    query_graphs = augment_dropedge(graphs, query_dropedge_rate)
    if key_view_mode == 'dropedge':
        key_graphs = augment_dropedge(graphs, key_dropedge_rate)
    else:
        key_graphs = augment_feature_mask(graphs, key_feature_mask_rate)
    return query_graphs, key_graphs

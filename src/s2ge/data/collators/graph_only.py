"""Minimal PyG batching for graph-only baselines and diagnostics."""

from torch_geometric.data import Batch


def build_graph_only_collate_fn():
    """Return a collator that batches graphs without text processing."""
    def _collate(original_batch):
        batch = {k: [d[k] for d in original_batch] for k in original_batch[0].keys()}
        if 'graph' in batch:
            batch['graph'] = Batch.from_data_list(batch['graph'])
        return batch

    return _collate

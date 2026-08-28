"""Auxiliary objectives for graph-token training."""

import torch
import torch.nn.functional as F


def _build_subgraph_adj_from_batch(self, graphs, graph_token_node_ids, graph_token_mask):
    """Build symmetrically normalized adjacency over exposed graph tokens.

    Self-loops are added before normalization. Padding rows and columns remain
    zero and are excluded from the loss by ``graph_token_mask``.
    """
    if graph_token_node_ids is None or graph_token_mask.numel() == 0:
        return None
    batch_size, seq_len = graph_token_mask.shape
    adj = torch.zeros((batch_size, seq_len, seq_len), dtype=torch.float32, device=graph_token_mask.device)
    if seq_len == 0:
        return adj
    batch_index = graphs.batch
    edge_index = graphs.edge_index.long()
    if batch_index.numel() == 0 or edge_index.numel() == 0:
        eye = torch.eye(seq_len, dtype=torch.float32, device=graph_token_mask.device).unsqueeze(0)
        return eye * graph_token_mask.unsqueeze(1).to(torch.float32) * graph_token_mask.unsqueeze(2).to(torch.float32)
    edge_graph_ids = batch_index[edge_index[0]]
    for graph_idx in range(batch_size):
        valid_mask = graph_token_mask[graph_idx]
        node_ids = graph_token_node_ids[graph_idx][valid_mask]
        valid_count = int(node_ids.numel())
        if valid_count == 0:
            continue
        mapping = torch.full((batch_index.size(0),), -1, dtype=torch.long, device=graph_token_mask.device)
        mapping[node_ids] = torch.arange(valid_count, device=graph_token_mask.device)
        graph_edge_mask = edge_graph_ids == graph_idx
        graph_edges = edge_index[:, graph_edge_mask]
        if graph_edges.numel() > 0:
            src = mapping[graph_edges[0]]
            dst = mapping[graph_edges[1]]
            keep = (src >= 0) & (dst >= 0)
            src = src[keep]
            dst = dst[keep]
            if src.numel() > 0:
                adj[graph_idx, src, dst] = 1.0
                adj[graph_idx, dst, src] = 1.0
        adj[graph_idx, torch.arange(valid_count), torch.arange(valid_count)] = 1.0
        degree_vec = adj[graph_idx, :valid_count, :valid_count].sum(dim=1).clamp(min=1.0)
        inv_sqrt_degree = degree_vec.rsqrt()
        adj[graph_idx, :valid_count, :valid_count] = adj[graph_idx, :valid_count, :valid_count] * inv_sqrt_degree.unsqueeze(1) * inv_sqrt_degree.unsqueeze(0)
    return adj


def _compute_align_loss(self, graph_token_embeds, graph_token_mask, graph_token_node_ids, graphs):
    """Match projected-token cosine similarity to normalized adjacency.

    This is the adjacency-based alignment term: mean squared error is computed
    only over valid token pairs. ``align_lambda`` is applied by ``GraphLLM``
    when the auxiliary term is added to the language-model loss.
    """
    if not (self.align_lambda > 0 and self.graph_token_mode == "nodes" and self.align_on_projected_tokens):
        return None
    adj = self._build_subgraph_adj_from_batch(graphs, graph_token_node_ids, graph_token_mask)
    if adj is None:
        return None
    normed = F.normalize(graph_token_embeds.float(), dim=-1)
    sim = torch.bmm(normed, normed.transpose(1, 2))
    mask2d = graph_token_mask.unsqueeze(1).to(torch.float32) * graph_token_mask.unsqueeze(2).to(torch.float32)
    align_diff = (sim - adj) * mask2d
    align_loss = align_diff.pow(2).sum() / mask2d.sum().clamp(min=1.0)
    self._latest_aux_metrics["align_loss"] = float(align_loss.detach().item())
    return align_loss


def _compute_moco_loss(self, query_graphs, key_graphs, query_projected):
    """Compute the optional MoCo objective for single-token graph mode."""
    if not self.moco_enabled or self.moco is None or self.graph_token_mode != "single":
        return None
    with torch.no_grad():
        self.moco.update_momentum(self.graph_encoder, self.projector)
        self.moco.momentum_encoder.eval()
        self.moco.momentum_projector.eval()
        key_projected = self._encode_graph_features(key_graphs, encoder=self.moco.momentum_encoder, projector=self.moco.momentum_projector)
        key_projected = F.normalize(key_projected.float(), dim=1)
    query_projected = F.normalize(query_projected.float(), dim=1)
    return self.moco(query_projected, key_projected)

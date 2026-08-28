"""Graph encoding, role perception, and graph-token intervention helpers.

These functions are attached to :class:`GraphLLM` to keep the assembly class
readable. Masks and node IDs are passed explicitly because padding positions,
local PyG IDs, and original graph IDs are not interchangeable.
"""

from pathlib import Path

import numpy as np
import torch
from torch_geometric.utils import degree


def _pool_graph_embeds(self, graphs, node_embeds):
    return self.readout.pool_graph_embeds(node_embeds, graphs)


def _encode_node_embeddings(self, graphs, encoder):
    """Resolve graph features and run the GNN in float32 for stability."""
    graphs = graphs.to(self.llm_input_device)
    x = self._resolve_graph_node_features(graphs)
    x = self._inject_infection_features(x, graphs)
    edge_attr = self._resolve_graph_edge_attr(graphs)
    for module in encoder.modules():
        if hasattr(module, "weight") and module.weight.dtype == torch.bfloat16:
            module.weight.data = module.weight.data.to(torch.float32)
            if hasattr(module, "bias") and module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float32)
    with torch.cuda.amp.autocast(enabled=False):
        node_embeds, _ = encoder(x, graphs.edge_index.long(), edge_attr)
    self._update_node_score_stats(graphs, encoder)
    return graphs, node_embeds.float()


def _resolve_graph_node_features(self, graphs):
    """Load sampled rows from the optional split-memmap feature backend."""
    backend = getattr(graphs, "graph_storage_backend", "")
    if isinstance(backend, (list, tuple)):
        backend = backend[0] if backend else ""
    x = graphs.x.float()
    if backend != "split_mmap":
        return x

    store_path = getattr(graphs, "x_store_path", "")
    store_shape = getattr(graphs, "x_store_shape", None)
    store_dtype = getattr(graphs, "x_store_dtype", "float16")
    node_ids = getattr(graphs, "n_id", None)
    if not store_path or store_shape is None:
        raise ValueError("split_mmap graph is missing x_store metadata")
    if node_ids is None:
        raise RuntimeError("split_mmap graph features require HSGS/NeighborLoader sampling so n_id is available")

    path = str(store_path[0] if isinstance(store_path, (list, tuple)) else store_path)
    if isinstance(store_shape, torch.Tensor):
        shape_values = store_shape[0].tolist() if store_shape.dim() > 1 else store_shape.tolist()
    elif isinstance(store_shape, np.ndarray):
        shape_values = store_shape[0].tolist() if store_shape.ndim > 1 else store_shape.tolist()
    elif isinstance(store_shape, list) and store_shape and isinstance(store_shape[0], (list, tuple)):
        shape_values = store_shape[0]
    else:
        shape_values = store_shape
    shape = tuple(int(v) for v in shape_values)
    if isinstance(store_dtype, (list, tuple)):
        store_dtype = store_dtype[0] if store_dtype else "float16"

    cache_key = (path, shape, str(store_dtype))
    memmap = self._feature_store_cache.get(cache_key)
    if memmap is None:
        memmap = np.memmap(Path(path), dtype=np.dtype(str(store_dtype)), mode="r", shape=shape)
        self._feature_store_cache[cache_key] = memmap
    node_ids_np = node_ids.detach().cpu().numpy().astype(np.int64, copy=False)
    sampled = np.asarray(memmap[node_ids_np], dtype=np.float32)
    return torch.from_numpy(sampled).to(self.llm_input_device)


def _resolve_graph_edge_attr(self, graphs):
    """Resolve materialized edge features or lazily indexed type embeddings."""
    if graphs.edge_attr is not None:
        return graphs.edge_attr.float()
    edge_type = getattr(graphs, "edge_type", None)
    edge_embed_path = getattr(graphs, "edge_type_embed_path", "")
    if edge_type is None or not edge_embed_path:
        return None

    path = str(edge_embed_path[0] if isinstance(edge_embed_path, (list, tuple)) else edge_embed_path)
    edge_type_embeds = self._edge_type_embed_cache.get(path)
    if edge_type_embeds is None:
        edge_type_embeds = torch.load(path, map_location=self.llm_input_device, weights_only=False).float()
        self._edge_type_embed_cache[path] = edge_type_embeds
    return edge_type_embeds[edge_type.long().to(edge_type_embeds.device)]


def _inject_infection_features(self, x, graphs):
    """Add the optional experimental anchor-proximity feature projection."""
    if self.infection_projector is None or not hasattr(graphs, "infection_proximity"):
        return x
    infection = getattr(graphs, "infection_proximity")
    if infection is None:
        return x
    if infection.dim() != 2:
        raise ValueError(f"infection_proximity must be rank-2, got shape {tuple(infection.shape)}")
    if infection.size(1) != self.infection_k:
        raise ValueError(
            f"infection_proximity width mismatch: expected {self.infection_k}, got {infection.size(1)}"
        )
    scale = float(self.infection_clip_max + 1)
    infection = infection.to(device=x.device, dtype=torch.float32) / scale
    infection = infection.to(dtype=self.infection_projector.weight.dtype)
    return x + self.infection_projector(infection)


def _ensure_node_score_storage(self, size: int):
    if self._node_score_sum is None:
        self._node_score_sum = torch.zeros(size, dtype=torch.float32)
        self._node_score_count = torch.zeros(size, dtype=torch.float32)
        return
    current = int(self._node_score_sum.numel())
    if size <= current:
        return
    grow = size - current
    self._node_score_sum = torch.cat([self._node_score_sum, torch.zeros(grow, dtype=torch.float32)], dim=0)
    self._node_score_count = torch.cat([self._node_score_count, torch.zeros(grow, dtype=torch.float32)], dim=0)


def _update_node_score_stats(self, graphs, encoder):
    if not self.collect_hsgs_node_scores or not self.training:
        return
    node_scores = getattr(encoder, "last_node_importance", None)
    global_node_ids = getattr(graphs, "n_id", None)
    if node_scores is None or global_node_ids is None:
        return
    node_scores = node_scores.detach().float().view(-1).cpu()
    global_node_ids = global_node_ids.detach().long().view(-1).cpu()
    if node_scores.numel() != global_node_ids.numel():
        return

    total_nodes_attr = getattr(graphs, "source_num_nodes", 0)
    if isinstance(total_nodes_attr, torch.Tensor):
        total_nodes = int(total_nodes_attr.view(-1)[0].item()) if total_nodes_attr.numel() > 0 else 0
    elif isinstance(total_nodes_attr, (list, tuple)):
        total_nodes = int(total_nodes_attr[0]) if total_nodes_attr else 0
    else:
        total_nodes = int(total_nodes_attr)
    if total_nodes <= 0:
        total_nodes = int(global_node_ids.max().item()) + 1 if global_node_ids.numel() > 0 else 0

    self._ensure_node_score_storage(total_nodes)
    ones = torch.ones_like(node_scores, dtype=torch.float32)
    self._node_score_sum.index_add_(0, global_node_ids, node_scores)
    self._node_score_count.index_add_(0, global_node_ids, ones)


def export_hsgs_node_scores(self, path):
    if self._node_score_sum is None or self._node_score_count is None:
        raise RuntimeError("No HSGS node scores have been collected")
    return self.export_hsgs_node_scores_from_tensors(path, self._node_score_sum, self._node_score_count)


def get_hsgs_node_score_tensors(self):
    if self._node_score_sum is None or self._node_score_count is None:
        return None, None
    return self._node_score_sum.clone(), self._node_score_count.clone()


def export_hsgs_node_scores_from_tensors(self, path, score_sum, score_count):
    path.parent.mkdir(parents=True, exist_ok=True)
    score_sum = score_sum.detach().cpu().float()
    score_count = score_count.detach().cpu().float()
    mean_score = score_sum / score_count.clamp(min=1.0)
    adjusted_score = mean_score * (1.0 - torch.exp(-score_count / max(self.hsgs_score_tau, 1e-6)))
    payload = {
        "score_sum": score_sum,
        "count": score_count,
        "mean_score": mean_score,
        "adjusted_score": adjusted_score,
        "tau": float(self.hsgs_score_tau),
    }
    torch.save(payload, path)
    return path


def _encode_graph_features(self, graphs, encoder, projector=None):
    graphs, node_embeds = self._encode_node_embeddings(graphs, encoder)
    graph_embeds = self._pool_graph_embeds(graphs, node_embeds)
    if projector is not None:
        projector_dtype = next(projector.parameters()).dtype
        graph_embeds = graph_embeds.to(dtype=projector_dtype)
        graph_embeds = projector(graph_embeds)
    return graph_embeds


def _build_degree_token_bias(self, graphs, graph_token_node_ids, graph_token_mask, hidden_dim):
    if self.degree_projector is None or graph_token_node_ids is None or not graph_token_mask.any():
        return None
    num_nodes = int(getattr(graphs, "num_nodes", 0) or 0)
    if num_nodes <= 0:
        return None
    device = self.degree_projector.weight.device
    deg = degree(graphs.edge_index[0], num_nodes=num_nodes, dtype=torch.float32)
    deg = deg.to(device)
    deg_norm = (deg / deg.max().clamp(min=1.0)).unsqueeze(-1)
    deg_norm = deg_norm.to(dtype=self.degree_projector.weight.dtype)
    deg_bias = self.degree_projector(deg_norm)
    safe_ids = graph_token_node_ids.clamp(min=0)
    token_bias = deg_bias[safe_ids]
    return token_bias * graph_token_mask.unsqueeze(-1).to(dtype=deg_bias.dtype)


def _add_node_identity_encoding(self, graph_token_embeds, graph_token_mask):
    if self.node_id_embed is None or graph_token_embeds.size(1) == 0:
        return graph_token_embeds
    seq_len = graph_token_embeds.size(1)
    local_ids = torch.arange(seq_len, device=graph_token_embeds.device)
    pos_embed = self.node_id_embed(local_ids).unsqueeze(0)
    return graph_token_embeds + pos_embed * graph_token_mask.unsqueeze(-1).to(dtype=graph_token_embeds.dtype)


def _bucketize_query_distance(self, distance_tensor):
    if distance_tensor.numel() == 0:
        return distance_tensor
    disconnected_bucket = self.graph_token_budget + 1
    bucketed = distance_tensor.clone()
    bucketed = bucketed.masked_fill(bucketed < 0, disconnected_bucket)
    return bucketed.clamp(min=0, max=disconnected_bucket)


def _coarse_bucketize_query_distance(self, distance_tensor, num_buckets):
    # Bins: 0, 1, 2, 3, >=4, and disconnected (<0).
    # num_buckets controls total slots; last bucket is always disconnected
    if distance_tensor.numel() == 0:
        return distance_tensor
    disconnected_bucket = num_buckets - 1
    max_explicit = num_buckets - 2  # e.g. 4 for 6-bucket scheme
    bucketed = distance_tensor.clone().clamp(min=0, max=max_explicit)
    bucketed = bucketed.masked_fill(distance_tensor < 0, disconnected_bucket)
    return bucketed.long().long()


def _build_query_syntax_bias(self, syntax_metadata, graph_token_mask):
    """Construct pre-projector role, distance, and corridor token biases."""
    if syntax_metadata is None or not self.graph_query_syntax_enabled or graph_token_mask.numel() == 0:
        return None
    active = syntax_metadata.get("active")
    if active is None or active.numel() == 0 or not active.any():
        return None
    active_mask = active.to(device=self.llm_input_device).unsqueeze(-1).unsqueeze(-1).to(dtype=torch.float32)

    token_bias = None
    if self.query_role_embed is not None and self.graph_role_encoding_enabled:
        role_ids = syntax_metadata["role_ids"].to(device=self.llm_input_device)
        role_bias = self.query_role_embed(role_ids)
        token_bias = role_bias if token_bias is None else token_bias + role_bias

    if self.graph_distance_encoding_enabled:
        if self.distance_src_embed is not None and self.distance_dst_embed is not None:
            d_src_buckets = self._bucketize_query_distance(syntax_metadata["d_src"].to(device=self.llm_input_device))
            d_dst_buckets = self._bucketize_query_distance(syntax_metadata["d_dst"].to(device=self.llm_input_device))
            distance_bias = self.distance_src_embed(d_src_buckets) + self.distance_dst_embed(d_dst_buckets)
            token_bias = distance_bias if token_bias is None else token_bias + distance_bias
        if self.corridor_projector is not None:
            delta = syntax_metadata["delta"].to(device=self.llm_input_device, dtype=torch.float32)
            corridor = torch.exp(-self.graph_corridor_alpha * delta.clamp(min=0.0))
            corridor = corridor * (delta >= 0).to(dtype=corridor.dtype)
            corridor = corridor.to(dtype=self.corridor_projector.weight.dtype)
            corridor_bias = self.corridor_projector(corridor.unsqueeze(-1))
            token_bias = corridor_bias if token_bias is None else token_bias + corridor_bias

    if token_bias is None:
        return None
    token_bias = token_bias * active_mask.to(dtype=token_bias.dtype, device=token_bias.device)
    return token_bias * graph_token_mask.unsqueeze(-1).to(dtype=token_bias.dtype, device=token_bias.device)


def _build_query_role_residual(self, syntax_metadata, graph_token_mask, base_graph_token_embeds=None):
    """Construct the optional post-projector role residual branch."""
    if self.query_role_residual_embed is None or syntax_metadata is None or graph_token_mask.numel() == 0:
        return None
    active = syntax_metadata.get("active")
    if active is None or active.numel() == 0 or not active.any():
        return None
    role_ids = syntax_metadata["role_ids"].to(device=self.llm_input_device)
    residual = self.query_role_residual_embed(role_ids)
    if self.query_role_residual_norm is not None:
        residual = self.query_role_residual_norm(residual)
    if self.query_role_residual_alpha is not None:
        residual = residual * self.query_role_residual_alpha.to(device=residual.device, dtype=residual.dtype)
    if (
        base_graph_token_embeds is not None
        and self.graph_role_residual_clip_ratio > 0
    ):
        base_norm = base_graph_token_embeds.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        residual_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        max_norm = base_norm * self.graph_role_residual_clip_ratio
        scale = torch.clamp(max_norm / residual_norm, max=1.0)
        residual = residual * scale
    active_mask = active.to(device=self.llm_input_device).unsqueeze(-1).unsqueeze(-1).to(dtype=residual.dtype)
    residual = residual * active_mask
    return residual * graph_token_mask.unsqueeze(-1).to(dtype=residual.dtype, device=residual.device)


def _build_query_distance_residual(self, syntax_metadata, graph_token_mask, base_graph_token_embeds=None):
    """Construct the optional coarse-distance residual in the LLM space."""
    if (
        self.distance_src_residual_embed is None
        or self.distance_dst_residual_embed is None
        or syntax_metadata is None
        or graph_token_mask.numel() == 0
    ):
        return None
    active = syntax_metadata.get("active")
    if active is None or active.numel() == 0 or not active.any():
        return None
    role_ids = syntax_metadata["role_ids"].to(device=self.llm_input_device)
    num_roles = int(getattr(self, "query_role_vocab_size", 4))
    num_embeddings = self.distance_src_residual_embed.num_embeddings
    num_buckets = num_embeddings // num_roles if self.graph_distance_residual_role_conditioned_enabled else num_embeddings
    d_src_buckets = self._coarse_bucketize_query_distance(syntax_metadata["d_src"].to(device=self.llm_input_device), num_buckets)
    d_dst_buckets = self._coarse_bucketize_query_distance(syntax_metadata["d_dst"].to(device=self.llm_input_device), num_buckets)
    if self.graph_distance_residual_role_conditioned_enabled:
        src_indices = role_ids * num_buckets + d_src_buckets
        dst_indices = role_ids * num_buckets + d_dst_buckets
    else:
        src_indices = d_src_buckets
        dst_indices = d_dst_buckets
    residual = self.distance_src_residual_embed(src_indices) + self.distance_dst_residual_embed(dst_indices)
    # Alpha gate: init=0 means zero contribution at the start of training.
    if self.distance_residual_alpha is not None:
        residual = residual * self.distance_residual_alpha.to(dtype=residual.dtype)
    # clip residual norm relative to base token norm
    if base_graph_token_embeds is not None and self.graph_distance_residual_clip_ratio > 0:
        base_norm = base_graph_token_embeds.detach().norm(dim=-1, keepdim=True).clamp(min=1e-6)
        residual_norm = residual.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = torch.clamp(base_norm * self.graph_distance_residual_clip_ratio / residual_norm, max=1.0)
        residual = residual * scale
    active_mask = active.to(device=self.llm_input_device).unsqueeze(-1).unsqueeze(-1).to(dtype=residual.dtype)
    residual = residual * active_mask
    return residual * graph_token_mask.unsqueeze(-1).to(dtype=residual.dtype, device=residual.device)


def _shuffle_graph_token_sequence(self, graph_token_embeds, graph_token_mask):
    """Apply the explicit random-order intervention to valid graph tokens."""
    if not self.graph_token_shuffle_enabled or graph_token_embeds.size(1) <= 1:
        return graph_token_embeds, graph_token_mask
    shuffled_embeds = graph_token_embeds.clone()
    shuffled_mask = graph_token_mask.clone()
    for batch_idx in range(graph_token_embeds.size(0)):
        token_count = int(graph_token_mask[batch_idx].sum().item())
        if token_count <= 1:
            continue
        perm = torch.randperm(token_count, device=graph_token_embeds.device)
        shuffled_embeds[batch_idx, :token_count] = graph_token_embeds[batch_idx, :token_count][perm]
        shuffled_mask[batch_idx, :token_count] = graph_token_mask[batch_idx, :token_count][perm]
    return shuffled_embeds, shuffled_mask


def _corrupt_graph_token_sequence(self, graph_token_embeds, graph_token_mask):
    """Replace valid graph tokens with norm-matched noise for diagnostics."""
    if not self.graph_token_corrupt_enabled or graph_token_embeds.numel() == 0:
        return graph_token_embeds
    corrupted = graph_token_embeds.clone()
    active = graph_token_mask.unsqueeze(-1)
    if not active.any():
        return corrupted
    noise = torch.randn_like(corrupted) * self.graph_token_corrupt_std
    base_norm = graph_token_embeds.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    noise_norm = noise.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    noise = noise * (base_norm / noise_norm)
    corrupted = torch.where(active, noise, corrupted)
    return corrupted


def _encode_graph_token_batch(self, graphs, encoder, projector=None):
    """Encode, order, annotate, project, and optionally intervene on tokens.

    Returns projected graph tokens, their validity mask, a pooled graph vector,
    and local node IDs used by adjacency-based alignment.
    """
    graphs, node_embeds = self._encode_node_embeddings(graphs, encoder)
    pooled_graph_embeds = self._pool_graph_embeds(graphs, node_embeds)
    graph_token_node_ids = None
    syntax_metadata = None

    if self.graph_tokens_disabled:
        graph_token_embeds = pooled_graph_embeds.new_zeros((pooled_graph_embeds.size(0), 0, pooled_graph_embeds.size(-1)))
        graph_token_mask = torch.zeros((pooled_graph_embeds.size(0), 0), dtype=torch.bool, device=pooled_graph_embeds.device)
    elif self.graph_token_mode == "nodes":
        graph_token_node_ids, graph_token_mask, _, syntax_metadata = self.readout.build_node_id_sequences(
            graphs,
            max_nodes=self.graph_token_budget,
            return_syntax_metadata=True,
        )
        if graph_token_node_ids.numel() == 0:
            graph_token_embeds = node_embeds.new_zeros((0, 0, node_embeds.size(-1)))
        else:
            safe_ids = graph_token_node_ids.clamp(min=0)
            graph_token_embeds = node_embeds[safe_ids]
            graph_token_embeds = graph_token_embeds * graph_token_mask.unsqueeze(-1).to(dtype=node_embeds.dtype, device=node_embeds.device)
        degree_bias = self._build_degree_token_bias(
            graphs,
            graph_token_node_ids,
            graph_token_mask,
            node_embeds.size(-1),
        )
        if degree_bias is not None:
            graph_token_embeds = graph_token_embeds + degree_bias
        syntax_bias = self._build_query_syntax_bias(syntax_metadata, graph_token_mask)
        if syntax_bias is not None:
            graph_token_embeds = graph_token_embeds + syntax_bias
        graph_token_embeds = self._add_node_identity_encoding(graph_token_embeds, graph_token_mask)
    else:
        graph_token_embeds = pooled_graph_embeds.unsqueeze(1)
        graph_token_mask = torch.ones(
            (pooled_graph_embeds.size(0), 1),
            dtype=torch.bool,
            device=pooled_graph_embeds.device,
        )

    if projector is not None:
        projector_dtype = next(projector.parameters()).dtype
        graph_token_embeds = graph_token_embeds.to(dtype=projector_dtype)
        graph_token_embeds = projector(graph_token_embeds)
        projected_graph_token_embeds = graph_token_embeds
        role_residual = self._build_query_role_residual(syntax_metadata, graph_token_mask, base_graph_token_embeds=projected_graph_token_embeds)
        if role_residual is not None:
            graph_token_embeds = graph_token_embeds + role_residual
            self._latest_aux_metrics["role_residual_alpha"] = float(self.query_role_residual_alpha.detach().item()) if self.query_role_residual_alpha is not None else 1.0
        if self.distance_residual_alpha is not None:
            self._latest_aux_metrics["distance_residual_alpha"] = float(self.distance_residual_alpha.detach().item())
        distance_residual = self._build_query_distance_residual(
            syntax_metadata,
            graph_token_mask,
            base_graph_token_embeds=projected_graph_token_embeds,
        )
        if distance_residual is not None:
            graph_token_embeds = graph_token_embeds + distance_residual
    graph_token_embeds, graph_token_mask = self._shuffle_graph_token_sequence(graph_token_embeds, graph_token_mask)
    graph_token_embeds = self._corrupt_graph_token_sequence(graph_token_embeds, graph_token_mask)
    self._latest_aux_metrics["graph_token_shuffle_enabled"] = float(self.graph_token_shuffle_enabled)
    self._latest_aux_metrics["graph_token_corrupt_enabled"] = float(self.graph_token_corrupt_enabled)
    self._latest_aux_metrics["syntax_disconnect_rate"] = 0.0
    if syntax_metadata is not None and syntax_metadata["disconnect"].numel() > 0:
        active = syntax_metadata.get("active")
        if active is not None and active.any():
            disconnect = syntax_metadata["disconnect"][active]
            self._latest_aux_metrics["syntax_disconnect_rate"] = float(disconnect.float().mean().item())
    return graph_token_embeds, graph_token_mask, pooled_graph_embeds, graph_token_node_ids

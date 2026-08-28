"""Graph pooling and role-aware node serialization for decoder input."""

from collections import deque

import torch
from torch_scatter import scatter


class GraphReadout:
    """Convert batched node embeddings into pooled or ordered graph tokens.

    Query-syntax ordering places endpoints first and ranks remaining nodes by
    shortest-path corridor membership and endpoint proximity. The returned IDs
    always refer to the current batched PyG graph, not the original global ID
    space; ``graph_batch.n_id`` is the explicit bridge when global IDs matter.
    """
    def __init__(self, bfs_order_enabled: bool = False, query_syntax_enabled: bool = False):
        self.bfs_order_enabled = bool(bfs_order_enabled)
        self.query_syntax_enabled = bool(query_syntax_enabled)

    def pool_graph_embeds(self, node_embeds, graph_batch):
        """Mean-pool target nodes when available, otherwise all graph nodes."""
        if hasattr(graph_batch, "target_node_mask") and graph_batch.target_node_mask is not None:
            target_mask = graph_batch.target_node_mask.bool()
            if target_mask.numel() == node_embeds.size(0) and target_mask.any():
                return scatter(node_embeds[target_mask], graph_batch.batch[target_mask], dim=0, reduce="mean")
        return scatter(node_embeds, graph_batch.batch, dim=0, reduce="mean")

    @staticmethod
    def _target_first_order(node_ids, target_mask):
        if target_mask is None:
            return node_ids
        graph_target_mask = target_mask[node_ids].bool()
        return torch.cat([node_ids[graph_target_mask], node_ids[~graph_target_mask]], dim=0)

    @staticmethod
    def _build_local_adj(graph_batch, ordered_node_ids):
        edge_index = getattr(graph_batch, "edge_index", None)
        if edge_index is None or edge_index.numel() == 0:
            return {int(node_id): [] for node_id in ordered_node_ids.tolist()}

        node_set = set(int(node_id) for node_id in ordered_node_ids.tolist())
        local_adj = {int(node_id): [] for node_id in ordered_node_ids.tolist()}
        src_nodes = edge_index[0].tolist()
        dst_nodes = edge_index[1].tolist()
        for src, dst in zip(src_nodes, dst_nodes):
            src = int(src)
            dst = int(dst)
            if src not in node_set or dst not in node_set:
                continue
            local_adj[src].append(dst)
            local_adj[dst].append(src)
        return local_adj

    def _bfs_order_node_ids(self, graph_batch, node_ids, target_mask, force_bfs: bool = False):
        ordered = self._target_first_order(node_ids, target_mask)
        if ordered.numel() <= 1 or (not self.bfs_order_enabled and not force_bfs) or not hasattr(graph_batch, "edge_index"):
            return ordered

        if target_mask is None:
            return ordered
        graph_target_mask = target_mask[ordered].bool()
        if not graph_target_mask.any():
            return ordered

        local_adj = self._build_local_adj(graph_batch, ordered)
        seeds = ordered[graph_target_mask].tolist()
        queue = [int(seed) for seed in seeds]
        visited = set(queue)
        bfs_order = []
        cursor = 0
        while cursor < len(queue):
            node_id = queue[cursor]
            cursor += 1
            bfs_order.append(node_id)
            for neigh in local_adj.get(node_id, []):
                if neigh not in visited:
                    visited.add(neigh)
                    queue.append(neigh)

        for node_id in ordered.tolist():
            node_id = int(node_id)
            if node_id not in visited:
                bfs_order.append(node_id)
                visited.add(node_id)
        return torch.tensor(bfs_order, dtype=torch.long, device=node_ids.device)

    @staticmethod
    def _bfs_distances(local_adj, start_node):
        distances = {int(start_node): 0}
        queue = deque([int(start_node)])
        while queue:
            node_id = queue.popleft()
            for neigh in local_adj.get(node_id, []):
                if neigh in distances:
                    continue
                distances[neigh] = distances[node_id] + 1
                queue.append(neigh)
        return distances

    @staticmethod
    def _extract_single_node_id(mask, node_ids):
        if mask is None or mask.numel() == 0:
            return None
        graph_mask = mask[node_ids].bool()
        matches = node_ids[graph_mask]
        if matches.numel() == 0:
            return None
        return int(matches[0].item())

    def _query_syntax_order_node_ids(self, graph_batch, node_ids, target_mask):
        """Order endpoints and path-corridor nodes while recording role data."""
        base_order = self._bfs_order_node_ids(graph_batch, node_ids, target_mask, force_bfs=True)
        if base_order.numel() <= 1 or not self.query_syntax_enabled:
            return base_order, None

        query_src_mask = getattr(graph_batch, "query_src_mask", None)
        query_dst_mask = getattr(graph_batch, "query_dst_mask", None)
        if query_src_mask is None and query_dst_mask is None:
            return base_order, None

        src_node = self._extract_single_node_id(query_src_mask, base_order)
        dst_node = self._extract_single_node_id(query_dst_mask, base_order)
        if src_node is None and dst_node is None:
            return base_order, None

        local_adj = self._build_local_adj(graph_batch, base_order)
        bfs_rank = {int(node_id): rank for rank, node_id in enumerate(base_order.tolist())}
        unreachable = len(local_adj) + 1

        d_src = self._bfs_distances(local_adj, src_node) if src_node is not None else {}
        d_dst = self._bfs_distances(local_adj, dst_node) if dst_node is not None else {}
        d_sd = d_src.get(dst_node, -1) if src_node is not None and dst_node is not None else -1
        disconnected = d_sd < 0

        def _distance_value(distance_map, node_id):
            return int(distance_map.get(int(node_id), unreachable))

        role_by_node = {}
        d_src_by_node = {}
        d_dst_by_node = {}
        delta_by_node = {}
        corridor_by_node = {}
        for node_id in base_order.tolist():
            node_id = int(node_id)
            node_d_src = _distance_value(d_src, node_id)
            node_d_dst = _distance_value(d_dst, node_id)
            d_src_by_node[node_id] = node_d_src
            d_dst_by_node[node_id] = node_d_dst
            if not disconnected and node_d_src < unreachable and node_d_dst < unreachable:
                delta = max(0, node_d_src + node_d_dst - d_sd)
            else:
                delta = -1
            delta_by_node[node_id] = int(delta)
            corridor_by_node[node_id] = 0 if disconnected or delta != 0 else 1
            if node_id == src_node:
                role_by_node[node_id] = 0
            elif node_id == dst_node:
                role_by_node[node_id] = 1
            elif corridor_by_node[node_id] == 1:
                role_by_node[node_id] = 2
            else:
                role_by_node[node_id] = 3

        ordered_nodes = []
        seen = set()
        for node_id in (src_node, dst_node):
            if node_id is None or node_id in seen:
                continue
            ordered_nodes.append(int(node_id))
            seen.add(int(node_id))

        remaining = [int(node_id) for node_id in base_order.tolist() if int(node_id) not in seen]
        if disconnected:
            remaining.sort(
                key=lambda node_id: (
                    d_src_by_node[node_id] + d_dst_by_node[node_id],
                    bfs_rank[node_id],
                )
            )
        else:
            remaining.sort(
                key=lambda node_id: (
                    delta_by_node[node_id],
                    d_src_by_node[node_id] + d_dst_by_node[node_id],
                    abs(d_src_by_node[node_id] - d_dst_by_node[node_id]),
                    bfs_rank[node_id],
                )
            )
        ordered_nodes.extend(remaining)

        syntax_metadata = {
            "role_by_node": role_by_node,
            "d_src_by_node": d_src_by_node,
            "d_dst_by_node": d_dst_by_node,
            "delta_by_node": delta_by_node,
            "corridor_by_node": corridor_by_node,
            "disconnected": disconnected,
        }
        ordered = torch.tensor(ordered_nodes, dtype=torch.long, device=node_ids.device)
        return ordered, syntax_metadata

    def build_node_id_sequences(self, graph_batch, max_nodes=None, return_syntax_metadata=False):
        """Build padded per-graph node-ID sequences and validity masks.

        When requested, syntax metadata uses four roles: source, destination,
        shortest-path corridor, and other. Distances of ``-1`` denote an
        unavailable or disconnected endpoint relation.
        """
        batch_index = graph_batch.batch
        if batch_index.numel() == 0:
            empty_ids = torch.zeros((0, 0), dtype=torch.long, device=batch_index.device)
            empty_mask = torch.zeros((0, 0), dtype=torch.bool, device=batch_index.device)
            empty_lengths = torch.zeros(0, dtype=torch.long, device=batch_index.device)
            if return_syntax_metadata:
                empty_meta = {
                    "role_ids": torch.zeros((0, 0), dtype=torch.long, device=batch_index.device),
                    "d_src": torch.zeros((0, 0), dtype=torch.long, device=batch_index.device),
                    "d_dst": torch.zeros((0, 0), dtype=torch.long, device=batch_index.device),
                    "delta": torch.zeros((0, 0), dtype=torch.long, device=batch_index.device),
                    "corridor": torch.zeros((0, 0), dtype=torch.float32, device=batch_index.device),
                    "disconnect": torch.zeros(0, dtype=torch.bool, device=batch_index.device),
                    "active": torch.zeros(0, dtype=torch.bool, device=batch_index.device),
                }
                return empty_ids, empty_mask, empty_lengths, empty_meta
            return empty_ids, empty_mask, empty_lengths

        num_graphs = int(getattr(graph_batch, "num_graphs", int(batch_index.max().item()) + 1))
        target_mask = getattr(graph_batch, "target_node_mask", None)
        if target_mask is not None and target_mask.numel() != batch_index.size(0):
            target_mask = None

        ordered_node_ids = []
        lengths = []
        syntax_rows = []
        syntax_disconnect = []
        syntax_active = []
        for graph_idx in range(num_graphs):
            node_ids = (batch_index == graph_idx).nonzero(as_tuple=False).view(-1)
            node_ids, syntax_meta = self._query_syntax_order_node_ids(graph_batch, node_ids, target_mask)
            if max_nodes is not None:
                node_ids = node_ids[:max_nodes]
            ordered_node_ids.append(node_ids)
            lengths.append(int(node_ids.numel()))

            if return_syntax_metadata:
                role_ids = []
                d_src_values = []
                d_dst_values = []
                delta_values = []
                corridor_values = []
                for node_id in node_ids.tolist():
                    node_id = int(node_id)
                    if syntax_meta is None:
                        role_ids.append(3)
                        d_src_values.append(-1)
                        d_dst_values.append(-1)
                        delta_values.append(-1)
                        corridor_values.append(0.0)
                    else:
                        role_ids.append(int(syntax_meta["role_by_node"][node_id]))
                        d_src_values.append(int(syntax_meta["d_src_by_node"][node_id]))
                        d_dst_values.append(int(syntax_meta["d_dst_by_node"][node_id]))
                        delta_values.append(int(syntax_meta["delta_by_node"][node_id]))
                        corridor_values.append(float(syntax_meta["corridor_by_node"][node_id]))
                syntax_rows.append(
                    {
                        "role_ids": role_ids,
                        "d_src": d_src_values,
                        "d_dst": d_dst_values,
                        "delta": delta_values,
                        "corridor": corridor_values,
                    }
                )
                syntax_disconnect.append(bool(syntax_meta["disconnected"]) if syntax_meta is not None else False)
                syntax_active.append(bool(syntax_meta is not None))

        max_length = max(lengths) if lengths else 0
        padded_ids = torch.full((num_graphs, max_length), -1, dtype=torch.long, device=batch_index.device)
        mask = torch.zeros((num_graphs, max_length), dtype=torch.bool, device=batch_index.device)
        for graph_idx, node_ids in enumerate(ordered_node_ids):
            if node_ids.numel() == 0:
                continue
            seq_len = node_ids.size(0)
            padded_ids[graph_idx, :seq_len] = node_ids
            mask[graph_idx, :seq_len] = True

        lengths_tensor = torch.tensor(lengths, dtype=torch.long, device=batch_index.device)
        if not return_syntax_metadata:
            return padded_ids, mask, lengths_tensor

        role_ids = torch.full((num_graphs, max_length), 3, dtype=torch.long, device=batch_index.device)
        d_src_tensor = torch.full((num_graphs, max_length), -1, dtype=torch.long, device=batch_index.device)
        d_dst_tensor = torch.full((num_graphs, max_length), -1, dtype=torch.long, device=batch_index.device)
        delta_tensor = torch.full((num_graphs, max_length), -1, dtype=torch.long, device=batch_index.device)
        corridor_tensor = torch.zeros((num_graphs, max_length), dtype=torch.float32, device=batch_index.device)
        for graph_idx, syntax_row in enumerate(syntax_rows):
            seq_len = len(syntax_row["role_ids"])
            if seq_len == 0:
                continue
            role_ids[graph_idx, :seq_len] = torch.tensor(syntax_row["role_ids"], dtype=torch.long, device=batch_index.device)
            d_src_tensor[graph_idx, :seq_len] = torch.tensor(syntax_row["d_src"], dtype=torch.long, device=batch_index.device)
            d_dst_tensor[graph_idx, :seq_len] = torch.tensor(syntax_row["d_dst"], dtype=torch.long, device=batch_index.device)
            delta_tensor[graph_idx, :seq_len] = torch.tensor(syntax_row["delta"], dtype=torch.long, device=batch_index.device)
            corridor_tensor[graph_idx, :seq_len] = torch.tensor(syntax_row["corridor"], dtype=torch.float32, device=batch_index.device)

        syntax_metadata = {
            "role_ids": role_ids,
            "d_src": d_src_tensor,
            "d_dst": d_dst_tensor,
            "delta": delta_tensor,
            "corridor": corridor_tensor,
            "disconnect": torch.tensor(syntax_disconnect, dtype=torch.bool, device=batch_index.device),
            "active": torch.tensor(syntax_active, dtype=torch.bool, device=batch_index.device),
        }
        return padded_ids, mask, lengths_tensor, syntax_metadata

    def build_node_sequences(self, node_embeds, graph_batch, max_nodes=None):
        """Gather padded node-embedding sequences in readout order."""
        node_id_sequences, mask, lengths = self.build_node_id_sequences(graph_batch, max_nodes=max_nodes)
        if node_id_sequences.numel() == 0:
            empty = node_embeds.new_zeros((0, 0, node_embeds.size(-1)))
            return empty, mask.to(device=node_embeds.device), lengths.to(device=node_embeds.device)

        safe_ids = node_id_sequences.clamp(min=0)
        padded = node_embeds[safe_ids]
        padded = padded * mask.unsqueeze(-1).to(dtype=node_embeds.dtype, device=node_embeds.device)
        return padded, mask.to(device=node_embeds.device), lengths.to(device=node_embeds.device)

    def __call__(self, node_embeds, graph_batch):
        """Alias pooled graph readout for legacy callers."""
        return self.pool_graph_embeds(node_embeds, graph_batch)

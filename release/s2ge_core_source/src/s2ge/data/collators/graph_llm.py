from torch_geometric.data import Batch

from s2ge.data.samplers.hsgs import hsgs_fanouts
from s2ge.data.samplers.pyg_neighbor import _build_query_role_masks
from s2ge.data.samplers.pyg_neighbor import sample_graph_with_hsgs


def _serialize_query_adjacency_desc(sampled_graph, query_nodes, max_nodes: int):
    node_ids = getattr(sampled_graph, "n_id", None)
    edge_index = getattr(sampled_graph, "edge_index", None)
    if node_ids is None or edge_index is None or edge_index.numel() == 0:
        return ""

    global_node_ids = [int(v) for v in node_ids.detach().cpu().tolist()]
    local_by_global = {global_id: local_idx for local_idx, global_id in enumerate(global_node_ids)}

    ordered_local = []
    seen_local = set()
    normalized_query_nodes = []
    for node_id in query_nodes or []:
        try:
            global_id = int(node_id)
        except (TypeError, ValueError):
            continue
        local_idx = local_by_global.get(global_id)
        if local_idx is None or local_idx in seen_local:
            continue
        seen_local.add(local_idx)
        ordered_local.append(local_idx)
        normalized_query_nodes.append(global_id)

    for local_idx, global_id in enumerate(global_node_ids):
        if local_idx in seen_local:
            continue
        seen_local.add(local_idx)
        ordered_local.append(local_idx)
        if len(ordered_local) >= max_nodes:
            break

    if not ordered_local:
        return ""

    ordered_global = [global_node_ids[local_idx] for local_idx in ordered_local]
    allowed_local = set(ordered_local)
    src_nodes = edge_index[0].detach().cpu().tolist()
    dst_nodes = edge_index[1].detach().cpu().tolist()

    edge_lines = []
    edge_budget = max(4, max_nodes * 4)
    for src_local, dst_local in zip(src_nodes, dst_nodes):
        src_local = int(src_local)
        dst_local = int(dst_local)
        if src_local not in allowed_local or dst_local not in allowed_local:
            continue
        edge_lines.append(f"{global_node_ids[src_local]} -> {global_node_ids[dst_local]}")
        if len(edge_lines) >= edge_budget:
            break

    if not edge_lines:
        return ""

    query_lines = []
    for pos, global_id in enumerate(normalized_query_nodes):
        role = "src" if pos == 0 else "dst" if pos == 1 else f"query_{pos}"
        query_lines.append(f"{role}: node_idx={global_id}")

    node_lines = [f"node_idx={global_id}" for global_id in ordered_global]
    parts = [
        "Serialized query-centered sampled subgraph.",
        "Query nodes:",
        "\n".join(query_lines) if query_lines else "unknown",
        "Node table:",
        "\n".join(node_lines),
        "Adjacency list:",
        "\n".join(edge_lines),
    ]
    return "\n".join(part for part in parts if part)


def build_graph_llm_collate_fn(args):
    hsgs_enabled = getattr(args, 'hsgs_enabled', False)
    hsgs_num_hops = max(1, int(getattr(args, 'hsgs_num_hops', 0) or args.gnn_num_layers))
    fanouts = hsgs_fanouts(hsgs_num_hops, args.hsgs_k_init, args.hsgs_gamma)
    max_seed_nodes = getattr(args, 'hsgs_seed_budget', max(1, fanouts[-1]))
    desc_mode = getattr(args, 'grbench_desc_mode', 'summary')
    desc_max_nodes = max(4, int(getattr(args, 'grbench_desc_max_nodes', 32)))

    def _collate(original_batch):
        batch = {k: [d[k] for d in original_batch] for k in original_batch[0].keys()}
        if 'graph' in batch:
            if hsgs_enabled:
                forced_seed_nodes = batch.get('query_nodes', [None] * len(batch['graph']))
                sampled_graphs = [
                    sample_graph_with_hsgs(
                        graph,
                        fanouts=fanouts,
                        max_seed_nodes=max_seed_nodes,
                        forced_seed_nodes=query_nodes,
                    )
                    for graph, query_nodes in zip(batch['graph'], forced_seed_nodes)
                ]
                if desc_mode == 'query_adjacency':
                    batch['desc'] = [
                        _serialize_query_adjacency_desc(graph, query_nodes, max_nodes=desc_max_nodes)
                        for graph, query_nodes in zip(sampled_graphs, forced_seed_nodes)
                    ]
                pyg_batch = Batch.from_data_list(sampled_graphs)
                batch['graph'] = pyg_batch
                batch['target_node_mask'] = getattr(pyg_batch, 'target_node_mask', None)
                batch['hsgs_fanouts'] = fanouts
            else:
                full_graphs = []
                forced_seed_nodes = batch.get('query_nodes', [None] * len(batch['graph']))
                for graph, query_nodes in zip(batch['graph'], forced_seed_nodes):
                    graph_copy = graph.clone()
                    query_src_mask, query_dst_mask = _build_query_role_masks(int(graph_copy.num_nodes), query_nodes)
                    graph_copy.query_src_mask = query_src_mask
                    graph_copy.query_dst_mask = query_dst_mask
                    graph_copy.query_node_mask = query_src_mask | query_dst_mask
                    full_graphs.append(graph_copy)
                pyg_batch = Batch.from_data_list(full_graphs)
                batch['graph'] = pyg_batch
                batch['target_node_mask'] = getattr(pyg_batch, 'target_node_mask', None)
        return batch

    return _collate

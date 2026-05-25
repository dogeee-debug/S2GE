import torch
from torch_geometric.loader import NeighborLoader

from s2ge.data.samplers.seed_select import select_seed_nodes


def _build_query_role_masks(num_nodes: int, query_nodes, local_node_ids=None):
    src_mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    dst_mask = torch.zeros(int(num_nodes), dtype=torch.bool)
    if query_nodes is None:
        return src_mask, dst_mask

    query_nodes = torch.as_tensor(query_nodes, dtype=torch.long).view(-1)
    if query_nodes.numel() == 0:
        return src_mask, dst_mask

    def _resolve_local_index(global_node_id: int):
        if local_node_ids is None:
            return global_node_id if 0 <= global_node_id < int(num_nodes) else None
        matches = (local_node_ids == int(global_node_id)).nonzero(as_tuple=False).view(-1)
        if matches.numel() == 0:
            return None
        return int(matches[0].item())

    src_index = _resolve_local_index(int(query_nodes[0].item()))
    if src_index is not None:
        src_mask[src_index] = True
    if query_nodes.numel() > 1:
        dst_index = _resolve_local_index(int(query_nodes[1].item()))
        if dst_index is not None:
            dst_mask[dst_index] = True
    return src_mask, dst_mask


def sample_graph_with_hsgs(graph, fanouts, max_seed_nodes: int, forced_seed_nodes=None):
    graph_cpu = graph.cpu()
    source_num_nodes = int(getattr(graph_cpu, "source_num_nodes", int(graph_cpu.num_nodes)))
    seed_nodes = select_seed_nodes(graph_cpu, max_seed_nodes=max_seed_nodes, forced_seed_nodes=forced_seed_nodes)
    if seed_nodes.numel() == 0:
        seed_nodes = torch.arange(min(int(graph_cpu.num_nodes), max_seed_nodes), dtype=torch.long)

    if graph_cpu.edge_index.numel() == 0 or int(graph_cpu.num_nodes) <= seed_nodes.numel():
        sampled = graph_cpu.clone()
        target_node_mask = torch.zeros(int(sampled.num_nodes), dtype=torch.bool)
        target_node_mask[seed_nodes.clamp(max=int(sampled.num_nodes) - 1)] = True
        sampled.target_node_mask = target_node_mask
        query_src_mask, query_dst_mask = _build_query_role_masks(int(sampled.num_nodes), forced_seed_nodes)
        sampled.query_src_mask = query_src_mask
        sampled.query_dst_mask = query_dst_mask
        sampled.query_node_mask = query_src_mask | query_dst_mask
        sampled.seed_node_ids = seed_nodes
        sampled.source_num_nodes = source_num_nodes
        for attr_name in ("graph_storage_backend", "x_store_path", "x_store_shape", "x_store_dtype", "edge_type_embed_path"):
            if hasattr(graph_cpu, attr_name):
                setattr(sampled, attr_name, getattr(graph_cpu, attr_name))
        return sampled

    loader = NeighborLoader(
        graph_cpu,
        num_neighbors=fanouts,
        input_nodes=seed_nodes,
        batch_size=int(seed_nodes.numel()),
        shuffle=False,
    )
    sampled = next(iter(loader))
    target_node_mask = torch.zeros(int(sampled.num_nodes), dtype=torch.bool)
    target_node_mask[: int(sampled.batch_size)] = True
    sampled.target_node_mask = target_node_mask
    query_src_mask, query_dst_mask = _build_query_role_masks(int(sampled.num_nodes), forced_seed_nodes, local_node_ids=sampled.n_id)
    sampled.query_src_mask = query_src_mask
    sampled.query_dst_mask = query_dst_mask
    sampled.query_node_mask = query_src_mask | query_dst_mask
    sampled.seed_node_ids = sampled.n_id[: int(sampled.batch_size)].clone()
    sampled.source_num_nodes = source_num_nodes
    for attr_name in ("graph_storage_backend", "x_store_path", "x_store_shape", "x_store_dtype", "edge_type_embed_path"):
        if hasattr(graph_cpu, attr_name):
            setattr(sampled, attr_name, getattr(graph_cpu, attr_name))
    return sampled

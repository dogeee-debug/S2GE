import torch


def _truncate_unique_seeds(seeds: torch.Tensor, max_seed_nodes: int):
    ordered = list(dict.fromkeys(int(seed) for seed in seeds.tolist()))
    return torch.tensor(ordered[:max_seed_nodes], dtype=torch.long)


def _append_unique_seeds(ordered: list[int], seen: set[int], seeds: torch.Tensor, num_nodes: int):
    for seed in seeds.tolist():
        seed = int(seed)
        if 0 <= seed < num_nodes and seed not in seen:
            ordered.append(seed)
            seen.add(seed)


def _collect_seed_attrs(graph, attr_names):
    collected = []
    for attr_name in attr_names:
        if hasattr(graph, attr_name):
            seeds = getattr(graph, attr_name)
            if seeds is not None:
                seeds = torch.as_tensor(seeds, dtype=torch.long).view(-1)
                if seeds.numel() > 0:
                    collected.append(seeds)
    return collected


def _collect_seed_masks(graph, attr_names):
    collected = []
    for attr_name in attr_names:
        if hasattr(graph, attr_name):
            mask = getattr(graph, attr_name)
            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.bool).view(-1)
                if mask.numel() == int(graph.num_nodes) and mask.any():
                    collected.append(mask.nonzero(as_tuple=False).view(-1))
    return collected


def _degree_ranked_seeds(graph, max_seed_nodes: int):
    num_nodes = int(graph.num_nodes)
    if num_nodes <= max_seed_nodes:
        return torch.arange(num_nodes, dtype=torch.long)
    row, col = graph.edge_index
    deg = torch.bincount(col, minlength=num_nodes) + torch.bincount(row, minlength=num_nodes)
    topk = min(max_seed_nodes, num_nodes)
    return torch.topk(deg, k=topk, largest=True).indices


def select_seed_nodes(graph, max_seed_nodes: int, forced_seed_nodes=None):
    num_nodes = int(graph.num_nodes)
    ordered = []
    seen = set()

    if forced_seed_nodes is not None:
        forced = torch.as_tensor(forced_seed_nodes, dtype=torch.long).view(-1)
        if forced.numel() > 0:
            _append_unique_seeds(ordered, seen, forced, num_nodes)
            if len(ordered) >= max_seed_nodes:
                return torch.tensor(ordered[:max_seed_nodes], dtype=torch.long)

    primary_sources = []
    primary_sources.extend(_collect_seed_attrs(graph, ['query_nodes', 'question_nodes', 'root_nodes', 'input_nodes']))
    primary_sources.extend(_collect_seed_masks(graph, ['query_node_mask', 'question_node_mask', 'root_node_mask', 'input_node_mask']))
    for seeds in primary_sources:
        _append_unique_seeds(ordered, seen, seeds, num_nodes)
        if len(ordered) >= max_seed_nodes:
            return torch.tensor(ordered[:max_seed_nodes], dtype=torch.long)

    secondary_sources = []
    secondary_sources.extend(_collect_seed_attrs(graph, ['seed_nodes', 'target_nodes']))
    secondary_sources.extend(_collect_seed_masks(graph, ['seed_node_mask', 'target_node_mask']))
    for seeds in secondary_sources:
        _append_unique_seeds(ordered, seen, seeds, num_nodes)
        if len(ordered) >= max_seed_nodes:
            return torch.tensor(ordered[:max_seed_nodes], dtype=torch.long)

    _append_unique_seeds(ordered, seen, _degree_ranked_seeds(graph, max_seed_nodes=max_seed_nodes), num_nodes)
    return torch.tensor(ordered[:max_seed_nodes], dtype=torch.long)

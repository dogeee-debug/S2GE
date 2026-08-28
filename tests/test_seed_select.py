"""Regression tests for deterministic query-aware seed selection."""

import torch
from torch_geometric.data import Data

from s2ge.data.samplers.seed_select import select_seed_nodes


def test_degree_ties_are_broken_by_node_id():
    """Equal-degree fallback seeds must not depend on backend sort behavior."""
    # Undirected chain 0--1--2--3 gives nodes 1 and 2 the same highest degree.
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    graph = Data(edge_index=edge_index, num_nodes=4)

    seeds = select_seed_nodes(graph, max_seed_nodes=3)

    assert seeds.tolist() == [1, 2, 0]

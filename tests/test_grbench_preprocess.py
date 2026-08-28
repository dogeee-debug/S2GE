import json

import torch
from torch_geometric.data import Data

from s2ge.data.preprocess._builders.implementation import (
    _write_grbench_graph_tables_streaming,
    build_grbench,
)


def test_grbench_streaming_tables_and_light_graph(tmp_path):
    root = tmp_path / "GRBENCH" / "processed_data" / "dblp"
    root.mkdir(parents=True)
    (root / "data.json").write_text(
        json.dumps({"qid": 0, "question": "q", "answer": "a"}) + "\n",
        encoding="utf-8",
    )
    graph = {
        "paper_nodes": {
            "p1": {
                "features": {"title": "Paper One"},
                "neighbors": {"author": ["a1"], "reference": ["p2"]},
            },
            "p2": {"features": {"title": "Paper Two"}, "neighbors": {}},
        },
        "author_nodes": {
            "a1": {"features": {"name": "Alice"}, "neighbors": {}},
        },
    }
    graph_path = root / "graph.json"
    graph_path.write_text(json.dumps(graph), encoding="utf-8")

    info = _write_grbench_graph_tables_streaming(
        graph_path,
        root / "nodes.csv",
        root / "edges.csv",
        include_node_attr=True,
    )
    assert info["node_count"] == 3
    assert info["edge_count"] == 2

    build_grbench(
        tmp_path / "GRBENCH",
        domain="dblp",
        feature_mode="light",
        embed_dim=16,
    )
    data = torch.load(root / "graph.pt", weights_only=False)

    assert isinstance(data, Data)
    assert data.x.shape == (3, 16)
    assert data.edge_index.shape[1] == 2

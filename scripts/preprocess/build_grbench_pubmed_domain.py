#!/usr/bin/env python3
"""Convert PyG Planetoid/PubMed into a GRBench-style source domain."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PyG Planetoid/PubMed into a GRBench graph.json source domain."
    )
    parser.add_argument("--pyg-root", type=Path, default=Path("datasets_raw/pyg_pubmed"))
    parser.add_argument("--grbench-root", type=Path, default=Path("GRBENCH"))
    parser.add_argument("--domain", type=str, default="pubmed")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def _class_label(label_idx: int) -> str:
    return f"pubmed_class_{int(label_idx)}"


def build_pubmed_domain(
    pyg_root: Path,
    grbench_root: Path,
    domain: str = "pubmed",
    force: bool = False,
) -> Path:
    output_root = grbench_root / "processed_data" / domain
    output_root.mkdir(parents=True, exist_ok=True)
    graph_json_path = output_root / "graph.json"
    manifest_path = output_root / "pubmed_source_manifest.json"

    if graph_json_path.exists() and manifest_path.exists() and not force:
        print(json.dumps({"status": "reused", "graph_json": str(graph_json_path)}, ensure_ascii=False))
        return output_root

    dataset = Planetoid(root=str(pyg_root), name="PubMed")
    data = dataset[0]
    labels = data.y.detach().cpu()
    edge_index = data.edge_index.detach().cpu()

    adjacency: dict[int, set[int]] = defaultdict(set)
    for src, dst in edge_index.t().tolist():
        adjacency[int(src)].add(int(dst))

    graph_payload = {"paper_nodes": {}}
    for node_idx in range(int(data.num_nodes)):
        node_id = f"pubmed:{node_idx}"
        graph_payload["paper_nodes"][node_id] = {
            "features": {
                "name": node_id,
                "label": _class_label(int(labels[node_idx])),
            },
            "neighbors": {
                "cites": sorted(f"pubmed:{dst}" for dst in adjacency.get(node_idx, ())),
            },
        }

    graph_json_path.write_text(
        json.dumps(graph_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    def _mask_count(name: str) -> int:
        mask = getattr(data, name, None)
        return int(torch.count_nonzero(mask).item()) if mask is not None else 0

    manifest = {
        "status": "built",
        "pyg_root": str(pyg_root),
        "grbench_root": str(grbench_root),
        "domain": domain,
        "dataset_name": "Planetoid/PubMed",
        "node_count": int(data.num_nodes),
        "edge_count_directed": int(edge_index.size(1)),
        "feature_dim": int(dataset.num_node_features),
        "class_count": int(dataset.num_classes),
        "split_counts": {
            "train": _mask_count("train_mask"),
            "val": _mask_count("val_mask"),
            "test": _mask_count("test_mask"),
        },
        "graph_json": str(graph_json_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return output_root


def main() -> None:
    args = parse_args()
    build_pubmed_domain(args.pyg_root, args.grbench_root, args.domain, args.force)


if __name__ == "__main__":
    main()

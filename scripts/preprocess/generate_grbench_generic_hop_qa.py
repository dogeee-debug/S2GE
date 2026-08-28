#!/usr/bin/env python3
"""Generate balanced shortest-hop QA records for any processed GRBench domain."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[2]
for candidate in (ROOT, ROOT / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from s2ge.data.preprocess.builders import _stream_grbench_nodes
from s2ge.data.preprocess.splits import write_split_files


QUESTION_TEMPLATES = (
    "What is the minimum number of graph hops needed to connect {src} to {dst}?",
    "What is the shortest hop count from {src} to {dst} in the graph?",
    "How many graph hops at minimum separate {src} from {dst}?",
)
DISPLAY_FEATURE_KEYS = ("name", "title", "label", "term", "display_name")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HopQA for an arbitrary GRBench graph domain.")
    parser.add_argument("--grbench-root", type=Path, default=Path("GRBENCH"))
    parser.add_argument("--source-domain", type=str, required=True)
    parser.add_argument("--target-domain", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=3200)
    parser.add_argument("--reservoir-size", type=int, default=50000)
    parser.add_argument("--min-hop", type=int, default=1)
    parser.add_argument("--max-hop", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=300000)
    parser.add_argument("--max-visited", type=int, default=250000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _link_or_copy(source: Path, target: Path) -> None:
    """Reuse graph artifacts while remaining portable on systems without symlink rights."""
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source.resolve())
        return
    except OSError:
        pass
    try:
        os.link(source, target)
        return
    except OSError:
        shutil.copy2(source, target)


def _sample_source_nodes(nodes_csv: Path, reservoir_size: int, rng: random.Random) -> list[int]:
    reservoir: list[int] = []
    seen = 0
    for chunk in pd.read_csv(nodes_csv, usecols=["node_idx"], chunksize=200000):
        for node_idx in chunk["node_idx"].astype(np.int64).tolist():
            seen += 1
            if len(reservoir) < reservoir_size:
                reservoir.append(int(node_idx))
                continue
            slot = rng.randrange(seen)
            if slot < reservoir_size:
                reservoir[slot] = int(node_idx)
    if not reservoir:
        raise ValueError(f"No nodes found in {nodes_csv}")
    return reservoir


def _count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        return sum(1 for line in handle if line.strip())


def _build_undirected_csr(edges_csv: Path, num_nodes: int):
    edge_count = _count_csv_rows(edges_csv)
    if edge_count <= 0:
        raise ValueError(f"No edges found in {edges_csv}; regenerate the source preprocessing first.")
    src = np.zeros(edge_count * 2, dtype=np.int64)
    dst = np.zeros(edge_count * 2, dtype=np.int64)
    cursor = 0
    try:
        chunks = pd.read_csv(edges_csv, usecols=["src_idx", "dst_idx"], chunksize=200000)
    except EmptyDataError as exc:
        raise ValueError(f"Cannot build HopQA from the empty edge table {edges_csv}") from exc
    for chunk in chunks:
        src_chunk = chunk["src_idx"].to_numpy(dtype=np.int64, copy=False)
        dst_chunk = chunk["dst_idx"].to_numpy(dtype=np.int64, copy=False)
        size = len(chunk)
        next_cursor = cursor + size
        src[cursor:next_cursor] = src_chunk
        dst[cursor:next_cursor] = dst_chunk
        reverse_cursor = next_cursor + size
        src[next_cursor:reverse_cursor] = dst_chunk
        dst[next_cursor:reverse_cursor] = src_chunk
        cursor = reverse_cursor
    src = src[:cursor]
    dst = dst[:cursor]
    order = np.argsort(src, kind="mergesort")
    src = src[order]
    dst = dst[order]
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=num_nodes), out=row_ptr[1:])
    return row_ptr, dst


def _degree(row_ptr: np.ndarray, node_idx: int) -> int:
    return int(row_ptr[node_idx + 1] - row_ptr[node_idx])


def _reconstruct_path(parent: dict[int, int], dst: int) -> list[int]:
    path = [dst]
    current = dst
    while parent[current] != -1:
        current = parent[current]
        path.append(current)
    path.reverse()
    return path


def _find_path_at_exact_hop(
    src: int,
    target_hop: int,
    row_ptr: np.ndarray,
    col_idx: np.ndarray,
    max_visited: int,
    rng: random.Random,
):
    parent = {src: -1}
    depth = {src: 0}
    queue = deque([src])
    candidates: list[int] = []
    while queue:
        node = queue.popleft()
        node_depth = depth[node]
        if node_depth >= target_hop:
            continue
        start, end = int(row_ptr[node]), int(row_ptr[node + 1])
        for neighbor_value in col_idx[start:end]:
            neighbor = int(neighbor_value)
            if neighbor in parent:
                continue
            next_depth = node_depth + 1
            parent[neighbor] = node
            depth[neighbor] = next_depth
            if next_depth == target_hop:
                candidates.append(neighbor)
            else:
                queue.append(neighbor)
            if len(parent) >= max_visited:
                return None
    if not candidates:
        return None
    return _reconstruct_path(parent, int(rng.choice(candidates)))


def _load_node_table(nodes_csv: Path):
    nodes_df = pd.read_csv(nodes_csv, usecols=["node_idx", "node_id", "node_type"])
    return {
        int(row.node_idx): {"node_id": str(row.node_id), "node_type": str(row.node_type)}
        for row in nodes_df.itertuples(index=False)
    }


def _display_name(node_type: str, node_id: str, node_info) -> str:
    features = node_info.get("features", {}) if isinstance(node_info, dict) else {}
    label = node_id
    for key in DISPLAY_FEATURE_KEYS:
        value = features.get(key)
        if isinstance(value, str) and value.strip():
            label = value.strip()
            break
    return f"{node_type}:{node_id}" if label == node_id else f"{label} ({node_type})"


def _collect_display_metadata(graph_json: Path, needed_node_ids: set[str]):
    metadata = {}
    for node_type, node_id, node_info in _stream_grbench_nodes(graph_json):
        if node_id in needed_node_ids:
            metadata[node_id] = _display_name(str(node_type), str(node_id), node_info)
            if len(metadata) == len(needed_node_ids):
                break
    for node_id in needed_node_ids - metadata.keys():
        metadata[node_id] = f"node:{node_id}"
    return metadata


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0 or args.min_hop < 1 or args.max_hop < args.min_hop:
        raise ValueError("Require num_samples > 0 and 1 <= min_hop <= max_hop")
    rng = random.Random(args.seed)
    source_root = args.grbench_root / "processed_data" / args.source_domain
    target_root = args.grbench_root / "processed_data" / args.target_domain
    target_root.mkdir(parents=True, exist_ok=True)

    graph_json = source_root / "graph.json"
    nodes_csv = source_root / "nodes.csv"
    edges_csv = source_root / "edges.csv"
    for required in (graph_json, nodes_csv, edges_csv):
        if not required.exists():
            raise FileNotFoundError(f"Source GRBench artifact missing: {required}")

    idx_to_meta = _load_node_table(nodes_csv)
    source_nodes = _sample_source_nodes(nodes_csv, args.reservoir_size, rng)
    row_ptr, col_idx = _build_undirected_csr(edges_csv, num_nodes=len(idx_to_meta))
    source_nodes = [idx for idx in source_nodes if _degree(row_ptr, idx) > 0]
    if not source_nodes:
        raise ValueError("No sampled source nodes have graph neighbors")

    records = []
    seen_pairs = set()
    needed_node_ids: set[str] = set()
    target_counts = {hop: 0 for hop in range(args.min_hop, args.max_hop + 1)}
    base_target, remainder = divmod(args.num_samples, len(target_counts))
    desired_counts = {
        hop: base_target + (1 if offset < remainder else 0)
        for offset, hop in enumerate(target_counts)
    }

    attempts = 0
    while len(records) < args.num_samples and attempts < args.max_attempts:
        attempts += 1
        remaining = [hop for hop in target_counts if target_counts[hop] < desired_counts[hop]]
        target_hop = rng.choice(remaining or list(target_counts))
        src = int(rng.choice(source_nodes))
        path = _find_path_at_exact_hop(src, target_hop, row_ptr, col_idx, args.max_visited, rng)
        if not path:
            continue
        dst = int(path[-1])
        if src == dst or (src, dst) in seen_pairs:
            continue
        seen_pairs.add((src, dst))
        target_counts[target_hop] += 1
        path_ids = [idx_to_meta[idx]["node_id"] for idx in path]
        needed_node_ids.update(path_ids)
        records.append({"src_idx": src, "dst_idx": dst, "hop_count": target_hop, "path_ids": path_ids})

    if len(records) != args.num_samples:
        raise RuntimeError(
            f"Collected only {len(records)}/{args.num_samples} samples after {attempts} attempts; "
            "increase --max-attempts or relax the hop range."
        )

    display_meta = _collect_display_metadata(graph_json, needed_node_ids)
    data_path = target_root / "data.json"
    with data_path.open("w", encoding="utf-8") as handle:
        for qid, record in enumerate(records):
            src_meta = idx_to_meta[record["src_idx"]]
            dst_meta = idx_to_meta[record["dst_idx"]]
            sample = {
                "qid": str(qid),
                "question": rng.choice(QUESTION_TEMPLATES).format(
                    src=display_meta[src_meta["node_id"]],
                    dst=display_meta[dst_meta["node_id"]],
                ),
                "answer": str(record["hop_count"]),
                "path": record["path_ids"],
                "meta": {
                    "src_node_id": src_meta["node_id"],
                    "dst_node_id": dst_meta["node_id"],
                    "src_idx": record["src_idx"],
                    "dst_idx": record["dst_idx"],
                    "source_domain": args.source_domain,
                },
            }
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    reusable = (
        "graph.pt",
        "nodes.csv",
        "edges.csv",
        "graph.json",
        "node_text_embeds.pt",
        "node_text_embeds.mmap",
        "edge_text_embeds.pt",
        "edge_type_embeds.pt",
        "preprocess_meta.json",
    )
    for name in reusable:
        source = source_root / name
        if source.exists():
            _link_or_copy(source, target_root / name)

    indices = list(range(len(records)))
    train_end = int(len(indices) * 0.4)
    val_end = train_end + int(len(indices) * 0.1)
    write_split_files(target_root / "split", indices[:train_end], indices[train_end:val_end], indices[val_end:])

    manifest = {
        "source_domain": args.source_domain,
        "target_domain": args.target_domain,
        "num_samples": len(records),
        "min_hop": args.min_hop,
        "max_hop": args.max_hop,
        "attempts": attempts,
        "per_hop_counts": target_counts,
        "seed": args.seed,
    }
    (target_root / "hop_generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

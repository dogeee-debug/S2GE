"""GRBench preprocessing implementation for dense and streaming backends.

The builder converts raw domain JSON into stable node/edge tables, graph
metadata, and either lightweight structural features or text embeddings. Large
text domains can keep features in split memory-mapped stores so training only
materializes rows selected by neighborhood sampling.
"""

import csv
import os
import json
import re
from collections import deque
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from s2ge.data.preprocess.embeddings import load_embedding_bundle


GRBENCH_TEXT_MODE_MAX_NODES = 1_000_000
GRBENCH_PERSISTED_SEED_NODES = 4096
GRBENCH_TEXT_SANITIZER_VERSION = 1


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def _offline_mode_enabled():
    return any(
        os.getenv(name, "0") == "1"
        for name in ("S2GE_OFFLINE_MODE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _write_preprocess_meta(root: Path, dataset_name: str, feature_mode: str, embed_dim: int, extra: dict | None = None):
    meta = {
        "dataset": dataset_name,
        "feature_mode": str(feature_mode),
        "embed_dim": int(embed_dim),
    }
    if extra:
        meta.update(extra)
    (root / "preprocess_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_preprocess_meta(root: Path):
    meta_path = root / "preprocess_meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_tensor_file(tensor: torch.Tensor, output_path: Path):
    torch.save(tensor, output_path)


def _stream_grbench_nodes(path: Path, chunk_size: int = 8 * 1024 * 1024):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buffer = ""
        eof = False

        def _fill():
            nonlocal buffer, eof
            if eof:
                return
            chunk = f.read(chunk_size)
            if chunk == "":
                eof = True
                return
            buffer += chunk

        def _skip_ws(pos: int):
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) or eof:
                    return pos
                _fill()

        def _decode_from_buffer(pos: int):
            while True:
                try:
                    value, consumed = decoder.raw_decode(buffer[pos:])
                    return value, pos + consumed
                except json.JSONDecodeError:
                    if eof:
                        raise
                    _fill()

        def _maybe_compact(pos: int):
            nonlocal buffer
            if pos > chunk_size:
                buffer = buffer[pos:]
                return 0
            return pos

        _fill()
        pos = _skip_ws(0)
        while pos >= len(buffer) and not eof:
            _fill()
            pos = _skip_ws(pos)
        if pos >= len(buffer) or buffer[pos] != "{":
            raise ValueError(f"Expected a top-level JSON object in {path}")
        pos += 1

        while True:
            pos = _maybe_compact(pos)
            pos = _skip_ws(pos)
            if pos >= len(buffer):
                if eof:
                    return
                _fill()
                continue
            if buffer[pos] == "}":
                return

            node_type, pos = _decode_from_buffer(pos)
            pos = _skip_ws(pos)
            while pos >= len(buffer):
                if eof:
                    raise ValueError(f"Malformed GRBENCH JSON object in {path}")
                _fill()
                pos = _skip_ws(pos)
            if buffer[pos] != ":":
                raise ValueError(f"Expected ':' after GRBENCH node type in {path}")
            pos += 1
            pos = _skip_ws(pos)
            while pos >= len(buffer):
                if eof:
                    raise ValueError(f"Malformed GRBENCH JSON object in {path}")
                _fill()
                pos = _skip_ws(pos)
            if buffer[pos] != "{":
                raise ValueError(f"Expected '{{' starting GRBENCH node bucket in {path}")
            pos += 1

            while True:
                pos = _maybe_compact(pos)
                pos = _skip_ws(pos)
                if pos >= len(buffer):
                    if eof:
                        raise ValueError(f"Unexpected EOF while reading {node_type} bucket in {path}")
                    _fill()
                    continue
                if buffer[pos] == "}":
                    pos += 1
                    break

                node_id, pos = _decode_from_buffer(pos)
                pos = _skip_ws(pos)
                while pos >= len(buffer):
                    if eof:
                        raise ValueError(f"Malformed GRBENCH node entry in {path}")
                    _fill()
                    pos = _skip_ws(pos)
                if buffer[pos] != ":":
                    raise ValueError(f"Expected ':' after GRBENCH node id in {path}")
                pos += 1
                pos = _skip_ws(pos)
                node_info, pos = _decode_from_buffer(pos)
                yield str(node_type), str(node_id), node_info

                pos = _skip_ws(pos)
                while pos >= len(buffer):
                    if eof:
                        raise ValueError(f"Unexpected EOF while reading {node_type} bucket in {path}")
                    _fill()
                    pos = _skip_ws(pos)
                if buffer[pos] == ",":
                    pos += 1
                    continue
                if buffer[pos] == "}":
                    pos += 1
                    break
                raise ValueError(f"Expected ',' or '}}' inside {node_type} bucket in {path}")

            pos = _skip_ws(pos)
            while pos >= len(buffer):
                if eof:
                    return
                _fill()
                pos = _skip_ws(pos)
            if buffer[pos] == ",":
                pos += 1
                continue
            if buffer[pos] == "}":
                return
            raise ValueError(f"Expected ',' or '}}' after GRBENCH bucket in {path}")


def _flatten_value(value):
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_flatten_value(v)}" for k, v in value.items())
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def _trim_numeric_noise_text(text, max_tokens=48):
    tokens = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?", str(text))
    kept = []
    for token in tokens:
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        kept.append(token)
        if len(kept) >= max_tokens:
            break
    return " ".join(kept).strip()


def _sanitize_grbench_features(features, domain, node_type):
    domain = str(domain).lower()
    node_type = str(node_type)
    if not isinstance(features, dict):
        return features
    if domain not in {"goodreads", "biomedical"}:
        return features

    sanitized = {}
    if domain == "goodreads":
        drop_keys = {"popular_shelves", "num_pages", "publication_year", "isbn", "isbn13", "publication_month", "publication_day"}
        for key, value in features.items():
            key_str = str(key)
            if key_str in drop_keys:
                continue
            if key_str in {"description", "title"}:
                cleaned = _trim_numeric_noise_text(value, max_tokens=64 if key_str == "title" else 96)
                if cleaned:
                    sanitized[key_str] = cleaned
                continue
            if key_str == "genres" and isinstance(value, list):
                genres = [str(item).strip() for item in value[:8] if str(item).strip()]
                if genres:
                    sanitized[key_str] = genres
                continue
            if key_str in {"language_code", "country_code", "format", "is_ebook"}:
                text = str(value).strip()
                if text:
                    sanitized[key_str] = text
                continue
            text = _trim_numeric_noise_text(value, max_tokens=32)
            if text:
                sanitized[key_str] = text
        return sanitized

    drop_keys = {
        "id",
        "identifier",
        "mesh_id",
        "omim_id",
        "doid",
        "umls",
        "entrez",
        "tax_id",
        "drugbank_id",
        "inchi",
        "inchikey",
        "pubchem",
        "pubmed",
        "pmid",
    }
    for key, value in features.items():
        key_str = str(key)
        if key_str in drop_keys or key_str.endswith("_id"):
            continue
        if key_str in {"name", "title", "description", "definition"}:
            cleaned = _trim_numeric_noise_text(value, max_tokens=64)
            if cleaned:
                sanitized[key_str] = cleaned
            continue
        text = _trim_numeric_noise_text(value, max_tokens=24)
        if text:
            sanitized[key_str] = text
    if not sanitized and "name" in features:
        fallback = _trim_numeric_noise_text(features.get("name", ""), max_tokens=32)
        if fallback:
            sanitized["name"] = fallback
    return sanitized


def _normalize_grbench_neighbor_type(rel_name: str, available_node_types):
    rel_name = str(rel_name)
    candidates = [
        rel_name,
        f"{rel_name}_nodes",
        f"{rel_name}s",
        f"{rel_name}s_nodes",
    ]
    if rel_name in {"reference", "cited_by", "paper", "papers"}:
        candidates.extend(["paper_nodes", "paper"])
    if rel_name in {"author", "authors"}:
        candidates.extend(["author_nodes"])
    if rel_name in {"venue", "venues"}:
        candidates.extend(["venue_nodes"])
    if rel_name in {"field", "fields"}:
        candidates.extend(["field_nodes"])

    for candidate in candidates:
        if candidate in available_node_types:
            return candidate
    return None


def _resolve_grbench_neighbor_type(source_node_type: str, rel_name: str, available_node_types):
    dst_node_type = _normalize_grbench_neighbor_type(rel_name, available_node_types)
    if dst_node_type is not None:
        return dst_node_type
    if len(tuple(available_node_types)) == 1:
        # Homogeneous graph domains may expose one entity bucket while
        # preserving original relation labels on edges. In that case, unmatched
        # relations should resolve back to the single available node type.
        return tuple(available_node_types)[0]

    # Biomedical GRBENCH uses relation names like
    # "Disease-localizes-Anatomy" instead of DBLP-style "author".
    # Resolve the target side relative to the source type so these edges are
    # not silently dropped during table materialization.
    rel_parts = [part.strip() for part in str(rel_name).split("-") if part.strip()]
    source_base = str(source_node_type).removesuffix("_nodes")
    if len(rel_parts) >= 3:
        endpoints = (rel_parts[0], rel_parts[-1])
        target_base = endpoints[1] if endpoints[0] == source_base else endpoints[0]
        return _normalize_grbench_neighbor_type(target_base, available_node_types)
    return None


def _parse_grbench_graph_path(graph_json_path: Path, include_node_attr: bool = True, domain: str = ""):
    nodes, edge_specs, edges, id_to_index = [], [], [], {}
    for node_type, node_id, node_info in _stream_grbench_nodes(graph_json_path):
        if node_type not in id_to_index:
            id_to_index[node_type] = {}
        idx = len(nodes)
        id_to_index[node_type][node_id] = idx
        node_record = {"node_idx": idx, "node_id": node_id, "node_type": node_type}
        if include_node_attr:
            features = _sanitize_grbench_features(node_info.get("features", {}), domain, node_type)
            node_record["node_attr"] = f"type={node_type}; {_flatten_value(features)}"
        nodes.append(node_record)
        neighbors = node_info.get("neighbors", {})
        for neigh_type, neigh_ids in neighbors.items():
            if isinstance(neigh_ids, dict):
                neigh_ids = list(neigh_ids.keys())
            for neigh_id in neigh_ids:
                edge_specs.append((idx, node_type, node_id, str(neigh_type), str(neigh_id)))

    available_node_types = tuple(id_to_index.keys())
    for src_idx, src_node_type, src_id, neigh_type, neigh_id_str in edge_specs:
        dst_node_type = _resolve_grbench_neighbor_type(src_node_type, neigh_type, available_node_types)
        if dst_node_type is None:
            continue
        if neigh_id_str not in id_to_index[dst_node_type]:
            continue
        dst_idx = id_to_index[dst_node_type][neigh_id_str]
        edges.append(
            {
                "src_idx": src_idx,
                "dst_idx": dst_idx,
                "src_id": src_id,
                "dst_id": neigh_id_str,
                "edge_attr": neigh_type,
            }
        )
    return pd.DataFrame(nodes), pd.DataFrame(edges)


def _write_grbench_graph_tables_streaming(
    graph_json_path: Path,
    nodes_csv_path: Path,
    edges_csv_path: Path,
    include_node_attr: bool = True,
    domain: str = "",
):
    node_count = 0
    edge_count = 0
    node_types = []
    type_seen = set()
    node_lookup = defaultdict(dict)
    with nodes_csv_path.open("w", encoding="utf-8", newline="") as node_fp:
        node_writer = csv.writer(node_fp)
        header = ["node_idx", "node_id", "node_type"] + (["node_attr"] if include_node_attr else [])
        node_writer.writerow(header)
        for node_type, node_id, node_info in _stream_grbench_nodes(graph_json_path):
            node_type = str(node_type)
            node_id = str(node_id)
            node_idx = node_count
            row = [node_idx, node_id, node_type]
            if include_node_attr:
                features = _sanitize_grbench_features(node_info.get("features", {}), domain, node_type)
                row.append(f"type={node_type}; {_flatten_value(features)}")
            node_writer.writerow(row)
            node_lookup[node_type][node_id] = int(node_idx)
            if node_type not in type_seen:
                type_seen.add(node_type)
                node_types.append(node_type)
            node_count += 1

    with edges_csv_path.open("w", encoding="utf-8", newline="") as edge_fp:
        edge_writer = csv.writer(edge_fp)
        edge_writer.writerow(["src_idx", "dst_idx", "src_id", "dst_id", "edge_attr"])
        available_node_types = tuple(node_types)
        for node_type, node_id, node_info in _stream_grbench_nodes(graph_json_path):
            node_type = str(node_type)
            node_id = str(node_id)
            src_idx = node_lookup.get(node_type, {}).get(node_id)
            if src_idx is None:
                continue
            neighbors = node_info.get("neighbors", {})
            for neigh_type, neigh_ids in neighbors.items():
                neigh_type = str(neigh_type)
                dst_node_type = _resolve_grbench_neighbor_type(node_type, neigh_type, available_node_types)
                if dst_node_type is None:
                    continue
                if isinstance(neigh_ids, dict):
                    neigh_ids = list(neigh_ids.keys())
                dst_lookup = node_lookup.get(dst_node_type, {})
                for neigh_id in neigh_ids:
                    neigh_id = str(neigh_id)
                    dst_idx = dst_lookup.get(neigh_id)
                    if dst_idx is None:
                        continue
                    edge_writer.writerow([src_idx, int(dst_idx), node_id, neigh_id, neigh_type])
                    edge_count += 1
    return {"node_count": int(node_count), "edge_count": int(edge_count), "node_type_names": list(node_types)}


def _grbench_light_node_features(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, embed_dim: int = 1024):
    if embed_dim < 8:
        raise ValueError(f"grbench light feature dim must be >= 8, got {embed_dim}")

    node_type_codes, unique_type_index = pd.factorize(nodes_df["node_type"].astype(str), sort=True)
    unique_types = unique_type_index.astype(str).tolist()
    if len(unique_types) > embed_dim:
        raise ValueError(
            "grbench light feature dim is too small for one-hot node types: "
            f"{len(unique_types)} unique node types exceed embed_dim={embed_dim}. "
            "Increase --grbench-embed-dim or use --grbench-feature-mode=text."
        )

    num_nodes = len(nodes_df)
    x = torch.zeros((num_nodes, embed_dim), dtype=torch.float32)
    type_indices = torch.from_numpy(node_type_codes.astype(np.int64, copy=False))
    x[torch.arange(num_nodes), type_indices] = 1.0

    if len(edges_df) > 0:
        src = torch.tensor(edges_df["src_idx"].to_numpy(), dtype=torch.long)
        dst = torch.tensor(edges_df["dst_idx"].to_numpy(), dtype=torch.long)
        out_deg = torch.bincount(src, minlength=num_nodes).float()
        in_deg = torch.bincount(dst, minlength=num_nodes).float()
    else:
        out_deg = torch.zeros(num_nodes, dtype=torch.float32)
        in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    total_deg = in_deg + out_deg

    deg_features = torch.stack(
        [
            torch.log1p(in_deg),
            torch.log1p(out_deg),
            torch.log1p(total_deg),
            (in_deg > 0).float(),
            (out_deg > 0).float(),
        ],
        dim=1,
    )

    start = len(unique_types)
    stop = min(embed_dim, start + deg_features.size(1))
    x[:, start:stop] = deg_features[:, : stop - start]
    return x, unique_types


def _grbench_light_node_features_from_arrays(
    node_type_codes,
    unique_types,
    src_idx,
    dst_idx,
    num_nodes: int,
    embed_dim: int = 1024,
):
    if embed_dim < 8:
        raise ValueError(f"grbench light feature dim must be >= 8, got {embed_dim}")
    if len(unique_types) > embed_dim:
        raise ValueError(
            "grbench light feature dim is too small for one-hot node types: "
            f"{len(unique_types)} unique node types exceed embed_dim={embed_dim}. "
            "Increase --grbench-embed-dim or use --grbench-feature-mode=text."
        )

    x = torch.zeros((num_nodes, embed_dim), dtype=torch.float32)
    type_indices = torch.from_numpy(np.asarray(node_type_codes, dtype=np.int64))
    x[torch.arange(num_nodes), type_indices] = 1.0

    if src_idx is not None and dst_idx is not None and len(src_idx) > 0:
        src = torch.from_numpy(np.asarray(src_idx, dtype=np.int64))
        dst = torch.from_numpy(np.asarray(dst_idx, dtype=np.int64))
        out_deg = torch.bincount(src, minlength=num_nodes).float()
        in_deg = torch.bincount(dst, minlength=num_nodes).float()
    else:
        out_deg = torch.zeros(num_nodes, dtype=torch.float32)
        in_deg = torch.zeros(num_nodes, dtype=torch.float32)
    total_deg = in_deg + out_deg

    deg_features = torch.stack(
        [
            torch.log1p(in_deg),
            torch.log1p(out_deg),
            torch.log1p(total_deg),
            (in_deg > 0).float(),
            (out_deg > 0).float(),
        ],
        dim=1,
    )

    start = len(unique_types)
    stop = min(embed_dim, start + deg_features.size(1))
    x[:, start:stop] = deg_features[:, : stop - start]
    return x, list(unique_types)


def _load_cached_grbench_tables(nodes_csv_path: Path, edges_csv_path: Path, feature_mode: str):
    try:
        if feature_mode == "light":
            nodes_df = pd.read_csv(nodes_csv_path, usecols=["node_idx", "node_id", "node_type"])
            edges_df = pd.read_csv(edges_csv_path, usecols=["src_idx", "dst_idx", "edge_attr"])
        else:
            nodes_df = pd.read_csv(nodes_csv_path)
            edges_df = pd.read_csv(edges_csv_path)
            if "node_attr" not in nodes_df.columns:
                raise ValueError("nodes.csv missing node_attr required for text mode")
        if len(nodes_df) == 0:
            raise ValueError("nodes.csv is empty")
        return nodes_df, edges_df
    except Exception:
        return None, None


def _count_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as fp:
        next(fp, None)
        return sum(1 for _ in fp)


def _grbench_tables_cache_valid(nodes_csv_path: Path, edges_csv_path: Path, expected_node_count: int = -1):
    if not nodes_csv_path.exists() or not edges_csv_path.exists():
        return False
    try:
        node_rows = _count_csv_rows(nodes_csv_path)
        edge_rows = _count_csv_rows(edges_csv_path)
    except OSError:
        return False
    if node_rows <= 0:
        return False
    if expected_node_count >= 0 and node_rows != int(expected_node_count):
        return False
    if node_rows > 1 and edge_rows <= 0:
        return False
    return True


def _grbench_text_cache_version_matches(meta: dict, domain: str, feature_mode: str):
    if str(feature_mode).lower() != "text":
        return True
    if str(domain).lower() not in {"goodreads", "biomedical"}:
        return True
    return int(meta.get("text_sanitizer_version", 0)) == GRBENCH_TEXT_SANITIZER_VERSION


def _load_grbench_stream_info_from_cache(nodes_csv_path: Path, edges_csv_path: Path, meta: dict, chunksize: int = 200_000):
    node_count = int(meta.get("node_count", -1))
    edge_count = int(meta.get("edge_count", -1))
    node_type_names = meta.get("node_type_names", [])
    if (
        node_count >= 0
        and edge_count >= 0
        and isinstance(node_type_names, list)
        and all(isinstance(value, str) for value in node_type_names)
    ):
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "node_type_names": list(node_type_names),
        }

    scanned_node_count, scanned_type_names = _scan_nodes_csv_metadata(nodes_csv_path, chunksize=chunksize)
    return {
        "node_count": int(scanned_node_count),
        "edge_count": int(edge_count if edge_count >= 0 else _count_csv_rows(edges_csv_path)),
        "node_type_names": list(scanned_type_names),
    }


def _scan_nodes_csv_metadata(nodes_csv_path: Path, chunksize: int = 200_000):
    node_count = 0
    node_type_names = []
    node_type_seen = set()
    for chunk in pd.read_csv(nodes_csv_path, usecols=["node_idx", "node_type"], chunksize=chunksize):
        node_count += int(len(chunk))
        for node_type in chunk["node_type"].astype(str).unique().tolist():
            if node_type not in node_type_seen:
                node_type_seen.add(node_type)
                node_type_names.append(node_type)
    return node_count, sorted(node_type_names)


def _load_node_type_codes(nodes_csv_path: Path, num_nodes: int, chunksize: int = 200_000):
    type_to_code = {}
    codes = np.zeros(num_nodes, dtype=np.int32)
    cursor = 0
    for chunk in pd.read_csv(nodes_csv_path, usecols=["node_type"], chunksize=chunksize):
        values = chunk["node_type"].astype(str).tolist()
        for value in values:
            if value not in type_to_code:
                type_to_code[value] = len(type_to_code)
            codes[cursor] = type_to_code[value]
            cursor += 1
    if cursor != num_nodes:
        raise RuntimeError(f"node_type row mismatch: expected {num_nodes}, got {cursor}")
    type_names = [name for name, _ in sorted(type_to_code.items(), key=lambda item: item[1])]
    return codes, type_names


def _load_edge_arrays_from_csv(edges_csv_path: Path, num_nodes: int, edge_count: int | None = None, chunksize: int = 200_000):
    if edge_count is None:
        edge_count = _count_csv_rows(edges_csv_path)
    edge_count = int(edge_count)
    src = np.zeros(edge_count, dtype=np.int64)
    dst = np.zeros(edge_count, dtype=np.int64)
    degrees = np.zeros(num_nodes, dtype=np.int64)
    edge_type_to_code = {}
    edge_type_codes = np.zeros(edge_count, dtype=np.int64)
    cursor = 0
    for chunk in pd.read_csv(edges_csv_path, usecols=["src_idx", "dst_idx", "edge_attr"], chunksize=chunksize):
        size = len(chunk)
        next_cursor = cursor + size
        src_chunk = chunk["src_idx"].to_numpy(dtype=np.int64, copy=False)
        dst_chunk = chunk["dst_idx"].to_numpy(dtype=np.int64, copy=False)
        src[cursor:next_cursor] = src_chunk
        dst[cursor:next_cursor] = dst_chunk
        degrees += np.bincount(src_chunk, minlength=num_nodes)
        degrees += np.bincount(dst_chunk, minlength=num_nodes)
        codes_chunk = np.zeros(size, dtype=np.int64)
        for idx, value in enumerate(chunk["edge_attr"].fillna("").astype(str).tolist()):
            if value not in edge_type_to_code:
                edge_type_to_code[value] = len(edge_type_to_code)
            codes_chunk[idx] = edge_type_to_code[value]
        edge_type_codes[cursor:next_cursor] = codes_chunk
        cursor = next_cursor
    if cursor != edge_count:
        raise RuntimeError(f"edge row mismatch: expected {edge_count}, got {cursor}")
    edge_type_names = [name for name, _ in sorted(edge_type_to_code.items(), key=lambda item: item[1])]
    return {
        "edge_count": edge_count,
        "src": src,
        "dst": dst,
        "degree": degrees,
        "edge_type_codes": edge_type_codes,
        "edge_type_names": edge_type_names,
    }


def _build_persisted_seed_nodes(num_nodes: int, edges_df: pd.DataFrame, max_seed_nodes: int = GRBENCH_PERSISTED_SEED_NODES):
    topk = min(int(max_seed_nodes), int(num_nodes))
    if topk <= 0:
        return torch.zeros(0, dtype=torch.long)
    if num_nodes <= topk or len(edges_df) == 0:
        return torch.arange(topk, dtype=torch.long)

    src_idx = edges_df["src_idx"].to_numpy(dtype=np.int64, copy=False)
    dst_idx = edges_df["dst_idx"].to_numpy(dtype=np.int64, copy=False)
    degree = np.bincount(src_idx, minlength=num_nodes) + np.bincount(dst_idx, minlength=num_nodes)
    return _select_topk_from_degree(degree, topk)


def _select_topk_from_degree(degree: np.ndarray, budget: int):
    """Rank by degree descending and node ID ascending for stable ties."""
    budget = min(max(int(budget), 0), int(degree.shape[0]))
    if budget <= 0:
        return torch.zeros(0, dtype=torch.long)
    node_ids = np.arange(degree.shape[0], dtype=np.int64)
    ordered = node_ids[np.lexsort((node_ids, -degree))][:budget]
    return torch.from_numpy(ordered.astype(np.int64, copy=False))


def _build_persisted_seed_nodes_from_degree(degree: np.ndarray, max_seed_nodes: int = GRBENCH_PERSISTED_SEED_NODES):
    if degree.size == 0:
        return torch.zeros(0, dtype=torch.long)
    topk = min(int(max_seed_nodes), int(degree.shape[0]))
    if topk <= 0:
        return torch.zeros(0, dtype=torch.long)
    if degree.shape[0] <= topk:
        return torch.arange(topk, dtype=torch.long)
    return _select_topk_from_degree(degree, topk)


def _build_undirected_grbench_graph(num_nodes: int, edges_df: pd.DataFrame):
    if isinstance(edges_df, dict):
        src = np.asarray(edges_df.get("src", np.zeros(0, dtype=np.int64)), dtype=np.int64)
        dst = np.asarray(edges_df.get("dst", np.zeros(0, dtype=np.int64)), dtype=np.int64)
    else:
        if num_nodes <= 0 or len(edges_df) == 0:
            return None, None
        src = edges_df["src_idx"].to_numpy(dtype=np.int64, copy=False)
        dst = edges_df["dst_idx"].to_numpy(dtype=np.int64, copy=False)
    if num_nodes <= 0 or src.size == 0:
        return None, None
    rows = np.concatenate([src, dst]).astype(np.int64, copy=False)
    cols = np.concatenate([dst, src]).astype(np.int32, copy=False)
    order = np.argsort(rows, kind="mergesort")
    rows = rows[order]
    cols = cols[order]
    counts = np.bincount(rows, minlength=num_nodes)
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    return row_ptr, cols


def _select_infection_anchor_nodes(num_nodes: int, edges_df: pd.DataFrame, infection_k: int, node_type_codes: np.ndarray | None = None):
    infection_k = min(max(int(infection_k), 0), int(num_nodes))
    if infection_k <= 0:
        return torch.zeros(0, dtype=torch.long)
    if len(edges_df) == 0:
        return torch.arange(infection_k, dtype=torch.long)
    src_idx = edges_df["src_idx"].to_numpy(dtype=np.int64, copy=False)
    dst_idx = edges_df["dst_idx"].to_numpy(dtype=np.int64, copy=False)
    degree = np.bincount(src_idx, minlength=num_nodes) + np.bincount(dst_idx, minlength=num_nodes)
    return _select_infection_anchor_nodes_from_degree(degree, infection_k, node_type_codes=node_type_codes)


def _select_infection_anchor_nodes_from_degree(degree: np.ndarray, infection_k: int, node_type_codes: np.ndarray | None = None):
    infection_k = min(max(int(infection_k), 0), int(degree.shape[0]))
    if infection_k <= 0:
        return torch.zeros(0, dtype=torch.long)
    if node_type_codes is None or node_type_codes.size == 0:
        return _select_topk_from_degree(degree, infection_k)

    selected = []
    selected_set = set()
    unique_types = sorted(int(v) for v in np.unique(node_type_codes).tolist())
    type_rankings = {}
    for type_code in unique_types:
        candidates = np.flatnonzero(node_type_codes == type_code)
        if candidates.size == 0:
            type_rankings[type_code] = []
            continue
        order = candidates[np.lexsort((candidates, -degree[candidates]))]
        type_rankings[type_code] = order.tolist()

    while len(selected) < infection_k:
        added = False
        for type_code in unique_types:
            ranking = type_rankings[type_code]
            while ranking and ranking[0] in selected_set:
                ranking.pop(0)
            if not ranking:
                continue
            node_id = int(ranking.pop(0))
            selected.append(node_id)
            selected_set.add(node_id)
            added = True
            if len(selected) >= infection_k:
                break
        if not added:
            break

    if len(selected) < infection_k:
        for node_id in _select_topk_from_degree(degree, infection_k).tolist():
            if node_id not in selected_set:
                selected.append(int(node_id))
                selected_set.add(int(node_id))
            if len(selected) >= infection_k:
                break
    return torch.tensor(selected[:infection_k], dtype=torch.long)


def _compute_infection_proximity(num_nodes: int, edges_df: pd.DataFrame, anchor_ids: torch.Tensor, clip_max: int):
    if anchor_ids.numel() == 0:
        return torch.zeros((num_nodes, 0), dtype=torch.uint16)
    if clip_max < 1:
        raise ValueError(f"infection_clip_max must be >= 1, got {clip_max}")

    row_ptr, col_idx = _build_undirected_grbench_graph(num_nodes, edges_df)
    if row_ptr is None or col_idx is None:
        proximity = np.zeros((num_nodes, int(anchor_ids.numel())), dtype=np.uint16)
        for col, anchor in enumerate(anchor_ids.tolist()):
            if 0 <= int(anchor) < num_nodes:
                proximity[int(anchor), col] = np.uint16(clip_max + 1)
        return torch.from_numpy(proximity)

    proximity = np.zeros((num_nodes, int(anchor_ids.numel())), dtype=np.uint16)
    max_value = int(clip_max) + 1
    for col, anchor in enumerate(anchor_ids.tolist()):
        anchor = int(anchor)
        if not (0 <= anchor < num_nodes):
            continue
        visited = np.zeros(num_nodes, dtype=np.bool_)
        distance = np.full(num_nodes, -1, dtype=np.int32)
        frontier = deque([anchor])
        visited[anchor] = True
        distance[anchor] = 0
        while frontier:
            node = frontier.popleft()
            node_dist = int(distance[node])
            start = int(row_ptr[node])
            end = int(row_ptr[node + 1])
            if start == end:
                continue
            next_dist = node_dist + 1
            if node_dist >= clip_max:
                continue
            for neigh in col_idx[start:end]:
                neigh = int(neigh)
                if visited[neigh]:
                    continue
                visited[neigh] = True
                distance[neigh] = next_dist
                frontier.append(neigh)
        reachable = distance >= 0
        if reachable.any():
            clipped = np.minimum(distance[reachable], clip_max).astype(np.uint16, copy=False)
            proximity[reachable, col] = np.uint16(max_value) - clipped
        proximity[anchor, col] = np.uint16(max_value)
    return torch.from_numpy(proximity)


def _stream_encode_text_column(csv_path: Path, column_name: str, bundle, total_rows: int, out_path: Path, dtype=np.float16, chunksize: int = 4096):
    if total_rows <= 0:
        np.memmap(out_path, dtype=dtype, mode="w+", shape=(0, int(bundle.embedding_dim))).flush()
        return
    storage = np.memmap(out_path, dtype=dtype, mode="w+", shape=(total_rows, int(bundle.embedding_dim)))
    row_cursor = 0
    for chunk in pd.read_csv(csv_path, usecols=[column_name], chunksize=chunksize):
        texts = chunk[column_name].fillna("").astype(str).tolist()
        embeds = bundle.text2embedding(bundle.model, bundle.tokenizer, bundle.device, texts)
        embeds_np = embeds.detach().cpu().numpy().astype(dtype, copy=False)
        next_cursor = row_cursor + embeds_np.shape[0]
        storage[row_cursor:next_cursor] = embeds_np
        row_cursor = next_cursor
    storage.flush()
    if row_cursor != total_rows:
        raise RuntimeError(f"streamed row count mismatch for {csv_path}: expected {total_rows}, wrote {row_cursor}")


def _materialize_grbench_text_backend(
    root: Path,
    nodes_csv_path: Path,
    edges_df: pd.DataFrame,
    bundle,
    model_name: str,
    cache_embeddings: bool,
    large_graph_backend: bool,
    node_count: int,
    edge_count: int,
):
    expected_dim = int(bundle.embedding_dim)
    node_embed_cache_path = root / "node_text_embeds.pt"
    edge_embed_cache_path = root / "edge_text_embeds.pt"
    node_embed_stream_path = root / "node_text_embeds.mmap"
    edge_embed_stream_path = root / "edge_text_embeds.mmap"
    edge_type_embed_store_path = root / "edge_type_embeds.pt"
    meta = _read_preprocess_meta(root)

    if large_graph_backend:
        expected_bytes = int(node_count) * int(expected_dim) * np.dtype(np.float16).itemsize
        has_matching_mmap = (
            node_embed_stream_path.exists()
            and node_embed_stream_path.stat().st_size == expected_bytes
        )
        can_reuse = (
            cache_embeddings
            and has_matching_mmap
            and (
                (
                    meta.get("dataset") == "grbench"
                    and meta.get("feature_mode") == "text"
                    and meta.get("embedding_model") == str(model_name)
                    and int(meta.get("embed_dim", -1)) == expected_dim
                    and int(meta.get("node_count", -1)) == node_count
                    and int(meta.get("edge_count", -1)) == edge_count
                    and meta.get("storage_backend") == "split_mmap"
                )
                or not meta
                or meta.get("feature_mode") != "text"
            )
        )
        if not can_reuse:
            _stream_encode_text_column(nodes_csv_path, "node_attr", bundle, node_count, node_embed_stream_path)
        return {
            "x": torch.zeros((node_count, 1), dtype=torch.float32),
            "edge_attr": None,
            "storage_backend": "split_mmap",
            "x_store_path": str(node_embed_stream_path),
            "x_store_shape": (node_count, expected_dim),
            "x_store_dtype": "float16",
            "feature_dim": expected_dim,
            "expected_dim": expected_dim,
        }

    can_reuse = (
        cache_embeddings
        and node_embed_cache_path.exists()
        and edge_embed_cache_path.exists()
        and meta.get("dataset") == "grbench"
        and meta.get("feature_mode") == "text"
        and int(meta.get("embed_dim", -1)) == expected_dim
        and int(meta.get("node_count", -1)) == node_count
        and int(meta.get("edge_count", -1)) == edge_count
        and meta.get("storage_backend", "dense") == "dense"
    )
    if can_reuse:
        x = torch.load(node_embed_cache_path, weights_only=False)
        edge_attr = torch.load(edge_embed_cache_path, weights_only=False)
    else:
        node_attrs = pd.read_csv(nodes_csv_path, usecols=["node_attr"])["node_attr"].fillna("").astype(str).tolist()
        edge_texts = edges_df.edge_attr.fillna("").astype(str).tolist() if edge_count > 0 else []
        x = bundle.text2embedding(
            bundle.model,
            bundle.tokenizer,
            bundle.device,
            node_attrs,
            out_path=node_embed_stream_path if cache_embeddings else None,
        )
        edge_attr = bundle.text2embedding(
            bundle.model,
            bundle.tokenizer,
            bundle.device,
            edge_texts,
            out_path=edge_embed_stream_path if cache_embeddings and edge_count > 0 else None,
        )
        if edge_count == 0:
            edge_attr = torch.zeros((0, expected_dim), dtype=torch.float32)
            if cache_embeddings:
                _save_tensor_file(edge_attr, edge_embed_cache_path)
        if cache_embeddings:
            _save_tensor_file(x, node_embed_cache_path)
            if edge_count > 0:
                _save_tensor_file(edge_attr, edge_embed_cache_path)
            Path(node_embed_stream_path).unlink(missing_ok=True)
    return {
        "x": x,
        "edge_attr": edge_attr,
        "storage_backend": "dense",
        "expected_dim": expected_dim,
    }


def refresh_grbench_seed_nodes(
    grbench_root,
    domain="dblp",
    score_path=None,
    budget=None,
    score_weight: float = 0.7,
    prior_weight: float = 0.3,
    exploration_ratio: float = 0.25,
):
    """Update persisted HSGS seeds from collected scores and degree priors.

    This is an optional experimental maintenance operation. It mutates only
    ``graph.pt`` seed metadata; graph topology and feature stores are unchanged.
    """
    if not score_path:
        raise ValueError("score_path is required to refresh GRBench seed nodes")
    root = Path(grbench_root) / "processed_data" / domain
    graph_pt_path = root / "graph.pt"
    payload = torch.load(score_path, map_location="cpu", weights_only=False)
    graph = torch.load(graph_pt_path, weights_only=False)
    num_nodes = int(graph.num_nodes)

    adjusted_score = payload.get("adjusted_score")
    count = payload.get("count")
    if adjusted_score is None:
        score_sum = payload.get("score_sum")
        tau = float(payload.get("tau", 128.0))
        if score_sum is None or count is None:
            raise KeyError("score payload must contain adjusted_score or score_sum/count")
        adjusted_score = score_sum.float() / count.float().clamp(min=1.0)
        adjusted_score = adjusted_score * (1.0 - torch.exp(-count.float() / max(tau, 1e-6)))
    adjusted_score = adjusted_score.view(-1).float()
    if adjusted_score.numel() != num_nodes:
        raise ValueError(f"score length mismatch: expected {num_nodes}, got {adjusted_score.numel()}")
    if count is not None:
        count = count.view(-1).float()
        if count.numel() != num_nodes:
            raise ValueError(f"count length mismatch: expected {num_nodes}, got {count.numel()}")

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        degree = torch.bincount(edge_index[0].cpu(), minlength=num_nodes) + torch.bincount(edge_index[1].cpu(), minlength=num_nodes)
        prior = degree.float()
    else:
        prior = torch.ones(num_nodes, dtype=torch.float32)

    def _normalize(values: torch.Tensor):
        values = values.clone()
        max_value = float(values.max().item()) if values.numel() > 0 else 0.0
        if max_value <= 0.0:
            return torch.zeros_like(values, dtype=torch.float32)
        return values.float() / max_value

    combined = score_weight * _normalize(adjusted_score) + prior_weight * _normalize(prior)
    budget = int(budget if budget is not None else getattr(graph, "seed_node_budget", GRBENCH_PERSISTED_SEED_NODES))
    budget = max(1, min(budget, num_nodes))
    exploration_budget = min(budget, max(0, int(round(budget * float(exploration_ratio)))))
    exploit_budget = max(0, budget - exploration_budget)

    selected = []
    if exploit_budget > 0:
        selected.extend(torch.topk(combined, k=exploit_budget, largest=True).indices.tolist())
    if exploration_budget > 0:
        if count is not None:
            exploration_score = _normalize(prior) * (1.0 / (count + 1.0))
        else:
            exploration_score = _normalize(prior)
        prior_order = torch.topk(exploration_score, k=min(num_nodes, budget * 4), largest=True).indices.tolist()
        for node_id in prior_order:
            if node_id not in selected:
                selected.append(int(node_id))
            if len(selected) >= budget:
                break
    if len(selected) < budget:
        fallback = torch.topk(combined, k=budget, largest=True).indices.tolist()
        for node_id in fallback:
            if node_id not in selected:
                selected.append(int(node_id))
            if len(selected) >= budget:
                break

    graph.seed_nodes = torch.tensor(selected[:budget], dtype=torch.long)
    graph.seed_node_budget = int(graph.seed_nodes.numel())
    graph.seed_refresh_method = "score_prior_mix"
    torch.save(graph, graph_pt_path)
    return graph.seed_nodes


def build_grbench(
    grbench_root,
    domain="dblp",
    model_name="sbert",
    feature_mode: str = "light",
    embed_dim: int = 1024,
    cache_embeddings: bool = False,
    infection_enabled: bool = False,
    infection_k: int = 0,
    infection_clip_max: int = 255,
):
    """Build one processed GRBench domain from its raw graph and QA files.

    ``feature_mode='light'`` is dependency-light and deterministic. Text modes
    require the preprocessing extras and may emit memory-mapped feature stores
    for domains that exceed ``GRBENCH_TEXT_MODE_MAX_NODES``.
    """
    root = Path(grbench_root) / "processed_data" / domain
    graph_json_path = root / "graph.json"
    nodes_csv_path = root / "nodes.csv"
    edges_csv_path = root / "edges.csv"
    graph_pt_path = root / "graph.pt"
    node_embed_cache_path = root / "node_text_embeds.pt"
    edge_embed_cache_path = root / "edge_text_embeds.pt"
    node_embed_stream_path = root / "node_text_embeds.mmap"
    edge_embed_stream_path = root / "edge_text_embeds.mmap"
    edge_type_embed_path = root / "edge_type_embeds.pt"
    feature_mode = str(feature_mode).lower()
    meta = _read_preprocess_meta(root)
    known_node_count = int(meta.get("node_count", -1))
    known_edge_count = int(meta.get("edge_count", -1))
    if feature_mode == "text" and known_node_count < 0 and nodes_csv_path.exists():
        known_node_count = _count_csv_rows(nodes_csv_path)
    graph_json_size = graph_json_path.stat().st_size if graph_json_path.exists() else 0
    prefer_streaming_tables = (
        feature_mode == "text" and known_node_count > GRBENCH_TEXT_MODE_MAX_NODES
    ) or (
        feature_mode == "light" and graph_json_size >= 1_000_000_000
    )

    nodes_df, edges_df = None, None
    stream_info = None
    edge_arrays = None
    node_count = None
    edge_count = None
    tables_exist = _grbench_tables_cache_valid(
        nodes_csv_path,
        edges_csv_path,
        expected_node_count=known_node_count,
    ) and _grbench_text_cache_version_matches(meta, domain, feature_mode)
    if tables_exist:
        if prefer_streaming_tables:
            stream_info = _load_grbench_stream_info_from_cache(nodes_csv_path, edges_csv_path, meta)
        else:
            nodes_df, edges_df = _load_cached_grbench_tables(nodes_csv_path, edges_csv_path, feature_mode=feature_mode)

    if (nodes_df is None or edges_df is None) and stream_info is None:
        if not graph_json_path.exists():
            raise FileNotFoundError(
                "GRBENCH graph.json not found. Please stage the graph environment under "
                f"{graph_json_path} before running preprocessing."
            )
        include_node_attr = feature_mode == "text"
        if feature_mode == "text" or prefer_streaming_tables:
            stream_info = _write_grbench_graph_tables_streaming(
                graph_json_path,
                nodes_csv_path,
                edges_csv_path,
                include_node_attr=include_node_attr,
                domain=domain,
            )
            if feature_mode == "text" and stream_info["node_count"] <= GRBENCH_TEXT_MODE_MAX_NODES:
                nodes_df, edges_df = _load_cached_grbench_tables(nodes_csv_path, edges_csv_path, feature_mode=feature_mode)
        else:
            nodes_df, edges_df = _parse_grbench_graph_path(
                graph_json_path,
                include_node_attr=include_node_attr,
                domain=domain,
            )
            node_columns = ["node_idx", "node_id", "node_type"] + (["node_attr"] if "node_attr" in nodes_df.columns else [])
            nodes_df.to_csv(nodes_csv_path, index=False, columns=node_columns)
            if len(edges_df) > 0:
                edges_df.to_csv(edges_csv_path, index=False, columns=["src_idx", "dst_idx", "src_id", "dst_id", "edge_attr"])
            else:
                edges_df.to_csv(edges_csv_path, index=False)

    if feature_mode == "light":
        if nodes_df is not None and edges_df is not None:
            node_count = len(nodes_df)
            edge_count = len(edges_df)
            x, node_type_names = _grbench_light_node_features(nodes_df, edges_df, embed_dim=embed_dim)
        else:
            if stream_info is None:
                stream_info = _load_grbench_stream_info_from_cache(nodes_csv_path, edges_csv_path, meta)
            node_count = int(stream_info["node_count"])
            edge_count = int(stream_info["edge_count"] if known_edge_count < 0 else known_edge_count)
            edge_arrays = _load_edge_arrays_from_csv(
                edges_csv_path,
                num_nodes=node_count,
                edge_count=edge_count,
            )
            node_type_codes, node_type_names = _load_node_type_codes(nodes_csv_path, node_count)
            x, node_type_names = _grbench_light_node_features_from_arrays(
                node_type_codes,
                node_type_names,
                edge_arrays["src"],
                edge_arrays["dst"],
                num_nodes=node_count,
                embed_dim=embed_dim,
            )
        edge_attr = None
        storage_backend = "dense"
        x_store_path = ""
        x_store_shape = None
        x_store_dtype = ""
        edge_type_embed_path = ""
        feature_dim = int(embed_dim)
    elif feature_mode == "text":
        if nodes_df is not None and edges_df is not None:
            node_count = len(nodes_df)
            edge_count = len(edges_df)
        else:
            if stream_info is None:
                stream_info = _load_grbench_stream_info_from_cache(nodes_csv_path, edges_csv_path, meta)
            node_count = int(stream_info["node_count"])
            edge_count = int(stream_info["edge_count"])
        bundle = load_embedding_bundle(model_name=model_name)
        text_backend = _materialize_grbench_text_backend(
            root,
            nodes_csv_path,
            edges_df if edges_df is not None else pd.DataFrame(columns=["edge_attr"]),
            bundle,
            model_name=str(model_name),
            cache_embeddings=cache_embeddings,
            large_graph_backend=node_count > GRBENCH_TEXT_MODE_MAX_NODES,
            node_count=node_count,
            edge_count=edge_count,
        )
        x = text_backend["x"]
        edge_attr = text_backend["edge_attr"]
        storage_backend = text_backend["storage_backend"]
        x_store_path = text_backend.get("x_store_path", "")
        x_store_shape = text_backend.get("x_store_shape")
        x_store_dtype = text_backend.get("x_store_dtype", "")
        feature_dim = int(text_backend["feature_dim"] if "feature_dim" in text_backend else text_backend["expected_dim"])
        if nodes_df is not None:
            node_type_names = sorted(nodes_df["node_type"].astype(str).unique().tolist())
        else:
            node_type_names = list(stream_info["node_type_names"])
    else:
        raise ValueError(f"Unknown GRBENCH feature_mode: {feature_mode}")

    if edges_df is None and edge_arrays is None:
        if node_count is None:
            if stream_info is None:
                stream_info = _load_grbench_stream_info_from_cache(nodes_csv_path, edges_csv_path, meta)
            node_count = int(stream_info["node_count"])
        if edge_count is None:
            edge_count = int(stream_info["edge_count"] if stream_info is not None else known_edge_count)
        edge_arrays = _load_edge_arrays_from_csv(
            edges_csv_path,
            num_nodes=int(node_count),
            edge_count=int(edge_count if known_edge_count < 0 else known_edge_count),
        )
    effective_num_nodes = int(node_count if nodes_df is None else len(nodes_df))
    effective_edge_count = int(edge_count if edges_df is None else len(edges_df))

    if effective_num_nodes > 0 and effective_edge_count <= 0:
        raise ValueError(
            "GRBENCH preprocessing produced zero edges. This usually means neighbor-type "
            "resolution failed for the current domain. Check graph.json relation naming "
            f"and regenerate {edges_csv_path}."
        )

    if edge_arrays is not None and edge_arrays["edge_count"] > 0:
        src_idx = torch.from_numpy(edge_arrays["src"])
        dst_idx = torch.from_numpy(edge_arrays["dst"])
        effective_edge_count = int(edge_arrays["edge_count"])
    elif edges_df is not None and len(edges_df) > 0:
        src_idx = torch.from_numpy(edges_df["src_idx"].to_numpy(dtype=np.int64, copy=False))
        dst_idx = torch.from_numpy(edges_df["dst_idx"].to_numpy(dtype=np.int64, copy=False))
        effective_edge_count = int(len(edges_df))
    else:
        effective_edge_count = 0
    if effective_edge_count > 0:
        edge_index = torch.stack([src_idx, dst_idx], dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr if feature_mode == "text" else None,
        num_nodes=effective_num_nodes,
    )
    data.node_type_names = node_type_names
    data.graph_feature_mode = feature_mode
    data.source_num_nodes = effective_num_nodes
    data.graph_storage_backend = storage_backend
    data.graph_feature_dim = int(feature_dim)
    if edge_arrays is not None:
        data.seed_nodes = _build_persisted_seed_nodes_from_degree(edge_arrays["degree"], max_seed_nodes=GRBENCH_PERSISTED_SEED_NODES)
    else:
        data.seed_nodes = _build_persisted_seed_nodes(len(nodes_df), edges_df, max_seed_nodes=GRBENCH_PERSISTED_SEED_NODES)
    data.seed_node_budget = int(data.seed_nodes.numel())
    if edge_arrays is not None and effective_edge_count > 0:
        data.edge_type = torch.from_numpy(edge_arrays["edge_type_codes"])
        data.edge_type_names = list(edge_arrays["edge_type_names"])
    elif edges_df is not None and len(edges_df) > 0:
        edge_type_codes, unique_edge_types = pd.factorize(edges_df["edge_attr"].astype(str), sort=True)
        data.edge_type = torch.from_numpy(edge_type_codes.astype(np.int64, copy=False))
        data.edge_type_names = unique_edge_types.astype(str).tolist()
    if x_store_path:
        data.x_store_path = str(x_store_path)
        data.x_store_shape = tuple(int(v) for v in x_store_shape)
        data.x_store_dtype = str(x_store_dtype)
    if storage_backend == "split_mmap":
        edge_type_embed_path = Path(edge_type_embed_path)
        if not edge_type_embed_path.exists():
            edge_type_values = edge_arrays["edge_type_names"] if edge_arrays is not None else []
            if edge_type_values:
                edge_type_embeds = bundle.text2embedding(
                    bundle.model,
                    bundle.tokenizer,
                    bundle.device,
                    edge_type_values,
                )
            else:
                edge_type_embeds = torch.zeros((0, int(feature_dim)), dtype=torch.float32)
            torch.save(edge_type_embeds, edge_type_embed_path)
        data.edge_type_embed_path = str(edge_type_embed_path)
    if infection_enabled and int(infection_k) > 0:
        if edge_arrays is not None:
            node_type_codes, _ = _load_node_type_codes(nodes_csv_path, effective_num_nodes)
            anchor_ids = _select_infection_anchor_nodes_from_degree(
                edge_arrays["degree"],
                infection_k,
                node_type_codes=node_type_codes,
            )
            infection_edges_df = {"src": edge_arrays["src"], "dst": edge_arrays["dst"]}
        else:
            node_type_codes, _ = pd.factorize(nodes_df["node_type"].astype(str), sort=True)
            anchor_ids = _select_infection_anchor_nodes(
                effective_num_nodes,
                edges_df,
                infection_k,
                node_type_codes=node_type_codes.astype(np.int32, copy=False),
            )
            infection_edges_df = edges_df
        data.infection_proximity = _compute_infection_proximity(
            effective_num_nodes,
            infection_edges_df,
            anchor_ids,
            clip_max=int(infection_clip_max),
        )
        data.infection_anchor_node_ids = anchor_ids
        data.infection_clip_max = int(infection_clip_max)
        data.infection_anchor_strategy = "type_diverse_degree"
    torch.save(data, graph_pt_path)
    actual_embed_dim = int(feature_dim)
    _write_preprocess_meta(
        root,
        "grbench",
        feature_mode,
        actual_embed_dim,
        extra={
            "embedding_model": str(model_name) if feature_mode == "text" else "",
            "node_count": effective_num_nodes,
            "edge_count": effective_edge_count,
            "cache_embeddings": bool(cache_embeddings),
            "infection_enabled": bool(infection_enabled and int(infection_k) > 0),
            "infection_k": int(infection_k),
            "infection_clip_max": int(infection_clip_max),
            "infection_anchor_strategy": "type_diverse_degree" if infection_enabled and int(infection_k) > 0 else "",
            "storage_backend": storage_backend,
            "node_type_names": list(node_type_names),
            "text_sanitizer_version": GRBENCH_TEXT_SANITIZER_VERSION
            if str(feature_mode).lower() == "text" and str(domain).lower() in {"goodreads", "biomedical"}
            else 0,
        },
    )


__all__ = [
    "build_grbench",
    "refresh_grbench_seed_nodes",
]

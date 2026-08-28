"""Lazy GRBench/Core-HopQA dataset and split metadata loader.

The global graph is loaded once per dataset instance. JSONL examples are read
through byte offsets, which keeps the large-domain memory footprint bounded
and lets DataLoader workers reopen their own file handles safely.
"""

from array import array
import json
import os
from pathlib import Path
import random
from collections import defaultdict, deque

import pandas as pd
import torch
from torch.utils.data import Dataset

from s2ge.data.datasets.common import read_split_indices
from s2ge.data.datasets.torch_load import load_graph_file

try:
    from s2ge.data.tasks.prompts import get_grbench_prompt, infer_grbench_task_family, infer_grbench_task_type
    from s2ge.data.tasks.specs import ENTITY_FAMILIES
except ModuleNotFoundError:
    ENTITY_FAMILIES = {"author_lookup", "venue_lookup", "collab_rank", "citation_rank", "graph_lookup", "other"}

    _SINGLE_INTEGER_PROMPT = (
        "Please answer the question using the graph. "
        "Output only a single integer as the answer. Do not output JSON or extra explanation."
    )
    _TOPIC_PROMPT = (
        "Please answer the question using the graph. "
        "Output exactly 3 specific research keywords separated by commas only. "
        "Do not output JSON, sentences, author names, or broad research fields. "
        'Format: "keyword one, keyword two, keyword three".'
    )
    _PLAIN_TEXT_PROMPT = (
        "Please answer the question using the graph. "
        "Output only the answer as a plain string. Do not output JSON or extra explanation."
    )
    _GENERIC_HOP_KEYWORDS = (
        "minimum number of graph hops",
        "minimum hop count",
        "shortest hop count",
        "graph hops at minimum",
        "hops at minimum separate",
    )

    def infer_grbench_task_type(question):
        q_lower = str(question).lower()
        if any(
            kw in q_lower
            for kw in (
                "witness node",
                "output yes and one witness node",
                "output yes plus one witness node",
                "or no_path",
                "within",
                "reachable from",
                "connect to",
            )
        ) and ("no_path" in q_lower or "witness" in q_lower):
            return "path_witness"
        if any(kw in q_lower for kw in ("keyword", "research interest", "area of research", "research area")):
            return "topic"
        return "hop"

    def infer_grbench_task_family(question):
        q_lower = str(question).lower()
        task_type = infer_grbench_task_type(question)
        if task_type == "topic":
            return "topic"
        if any(kw in q_lower for kw in ("how many people at minimum", "least count of people", "to get introduced to", "to be familiar with")):
            return "hop"
        if any(kw in q_lower for kw in _GENERIC_HOP_KEYWORDS):
            return "hop"
        if any(kw in q_lower for kw in ("who are the authors of", "could you tell me the authors of", "who wrote the paper")):
            return "author_lookup"
        if any(kw in q_lower for kw in ("published", "venue", "journal", "conference")):
            return "venue_lookup"
        if any(kw in q_lower for kw in ("highest number of collaborations", "closest collaborator", "closest co-author")):
            return "collab_rank"
        if any(kw in q_lower for kw in ("most frequently cited", "most cited", "highest number of citations")):
            return "citation_rank"
        if any(kw in q_lower for kw in ("how many", "number of", "count of")):
            return "counting"
        if any(kw in q_lower for kw in ("who within", "which author", "who is", "who ")):
            return "graph_lookup"
        return "other"

    def get_grbench_prompt(task_family):
        if task_family in {"hop", "counting"}:
            return _SINGLE_INTEGER_PROMPT
        if task_family == "topic":
            return _TOPIC_PROMPT
        return _PLAIN_TEXT_PROMPT


class GRBenchDataset(Dataset):
    """Expose one GRBench domain as graph-conditioned generation examples.

    Each item contains the shared graph, an exact-answer question, and query
    node metadata consumed by query-aware sampling and role-based perception.
    """
    _ENTITY_FAMILIES = ENTITY_FAMILIES
    _TASK_FILTERS = {
        "all": None,
        "counting": {"counting"},
        "entity": _ENTITY_FAMILIES,
        "hop": {"hop"},
        "path_witness": {"path_witness"},
        "topic": {"topic"},
        "author_lookup": {"author_lookup"},
        "venue_lookup": {"venue_lookup"},
        "collab_rank": {"collab_rank"},
        "citation_rank": {"citation_rank"},
        "graph_lookup": {"graph_lookup"},
        "other": {"other"},
    }

    _PATH_WITNESS_DOMAIN_HINTS = ("pathexist_witness", "path_witness")

    def __init__(self, root, domain, desc_mode="summary", desc_max_nodes=200, metric_max_nodes=5000, task_filter="all"):
        """Open a processed domain and build a lazy sample index."""
        self.root = Path(root)
        self.domain = domain
        self.desc_mode = desc_mode
        self.desc_max_nodes = desc_max_nodes
        self.metric_max_nodes = metric_max_nodes
        self.task_filter = self._resolve_task_filter(task_filter, domain)
        self.processed_root = self.root / "processed_data" / domain
        self.data_path = self.processed_root / "data.json"
        if not self.data_path.exists():
            raise FileNotFoundError(f"GRBENCH data.json not found: {self.data_path}")

        self._data_fp = None
        self._data_fp_pid = None
        self.sample_offsets = self._build_sample_offsets()
        self.sample_indices = self._build_sample_index()
        self.num_samples = len(self.sample_indices)

        self.graph_path = self.processed_root / "graph.pt"
        self.nodes_path = self.processed_root / "nodes.csv"
        self.edges_path = self.processed_root / "edges.csv"
        self.graph = load_graph_file(self.graph_path)
        self.textual_nodes = None
        self.textual_edges = None
        self.node_id_set = None
        self.edge_set = None
        self.node_idx_by_id = None
        self._split_cache = None
        self._split_seed = 0
        self._prompt_hop = get_grbench_prompt("hop")
        self._prompt_topic = get_grbench_prompt("topic")
        self._prompt_entity = get_grbench_prompt("author_lookup")
        self._prompt_counting = get_grbench_prompt("counting")
        self._FAMILY_PROMPTS = {
            "hop": self._prompt_hop,
            "path_witness": get_grbench_prompt("path_witness"),
            "topic": self._prompt_topic,
            "counting": self._prompt_counting,
            "author_lookup": self._prompt_entity,
            "venue_lookup": self._prompt_entity,
            "collab_rank": self._prompt_entity,
            "citation_rank": self._prompt_entity,
            "graph_lookup": self._prompt_entity,
            "other": self._prompt_entity,
        }
        self.graph_type = f"GRBENCH-{domain}"

    def __len__(self):
        return self.num_samples

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_data_fp"] = None
        state["_data_fp_pid"] = None
        return state

    def __del__(self):
        data_fp = getattr(self, "_data_fp", None)
        if data_fp is not None and not data_fp.closed:
            data_fp.close()

    def _build_sample_offsets(self):
        offsets = array("Q")
        with self.data_path.open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def _data_handle(self):
        current_pid = os.getpid()
        if self._data_fp is not None and self._data_fp_pid != current_pid:
            self._data_fp.close()
            self._data_fp = None
            self._data_fp_pid = None
        if self._data_fp is None or self._data_fp.closed:
            self._data_fp = self.data_path.open("rb")
            self._data_fp_pid = current_pid
        return self._data_fp

    @classmethod
    def _normalize_task_filter(cls, task_filter):
        normalized = str(task_filter or "all").strip().lower()
        if normalized not in cls._TASK_FILTERS:
            raise ValueError(f"Unsupported GRBench task filter: {task_filter}")
        return normalized

    @classmethod
    def _infer_task_filter_from_domain(cls, domain):
        domain_text = str(domain or "").strip().lower()
        if any(hint in domain_text for hint in cls._PATH_WITNESS_DOMAIN_HINTS):
            return "path_witness"
        return None

    @classmethod
    def _resolve_task_filter(cls, task_filter, domain):
        normalized = cls._normalize_task_filter(task_filter)
        inferred = cls._infer_task_filter_from_domain(domain)
        if inferred is None:
            return normalized
        if normalized in {"all", inferred}:
            return inferred
        if normalized == "hop":
            return inferred
        return normalized

    @classmethod
    def _match_task_filter(cls, task_family, task_filter):
        allowed = cls._TASK_FILTERS[task_filter]
        return allowed is None or task_family in allowed

    def _build_sample_index(self):
        if self.task_filter == "all":
            return list(range(len(self.sample_offsets)))
        kept_indices = []
        family_counter = {}
        for raw_index in range(len(self.sample_offsets)):
            sample = self._load_raw_sample(raw_index)
            task_family = self._infer_task_family(sample["question"])
            family_counter[task_family] = family_counter.get(task_family, 0) + 1
            if self._match_task_filter(task_family, self.task_filter):
                kept_indices.append(raw_index)
        if not kept_indices:
            counts = ", ".join(f"{family}={count}" for family, count in sorted(family_counter.items())) or "none"
            raise ValueError(
                f"No GRBench samples matched task filter '{self.task_filter}' for domain '{self.domain}'. "
                f"Observed task families: {counts}"
            )
        return kept_indices

    def _load_raw_sample(self, raw_index):
        handle = self._data_handle()
        handle.seek(int(self.sample_offsets[raw_index]))
        line = handle.readline()
        if not line:
            raise IndexError(f"GRBENCH sample index out of range: {raw_index}")
        return json.loads(line.decode("utf-8"))

    def _load_sample(self, index):
        if index < 0 or index >= self.num_samples:
            raise IndexError(f"GRBENCH filtered sample index out of range: {index}")
        raw_index = self.sample_indices[index]
        return self._load_raw_sample(raw_index)

    def _build_desc(self):
        if self.desc_mode == "none":
            return ""
        if self.desc_mode == "summary":
            return f"Graph Domain: {self.domain}. Nodes: {int(self.graph.num_nodes)}. Edges: {int(self.graph.num_edges)}."
        if self.desc_mode == "sampled_nodes" and self.nodes_path.exists():
            textual_nodes = pd.read_csv(self.nodes_path, nrows=self.desc_max_nodes)
            return textual_nodes.to_csv(index=False)
        return ""

    def _build_query_adjacency_desc(self, query_nodes):
        if not query_nodes:
            return self._build_desc()

        safe_query_nodes = []
        seen_query_nodes = set()
        for node_id in query_nodes:
            node_idx = self._coerce_node_index(node_id)
            if node_idx is None or node_idx in seen_query_nodes:
                continue
            if node_idx < 0 or node_idx >= int(self.graph.num_nodes):
                continue
            seen_query_nodes.add(node_idx)
            safe_query_nodes.append(node_idx)
        if not safe_query_nodes:
            return self._build_desc()

        adjacency = defaultdict(list)
        edge_index = self.graph.edge_index
        if edge_index is None or edge_index.numel() == 0:
            return self._build_desc()
        src_nodes = edge_index[0].detach().cpu().tolist()
        dst_nodes = edge_index[1].detach().cpu().tolist()
        for src, dst in zip(src_nodes, dst_nodes):
            adjacency[int(src)].append(int(dst))

        max_nodes = max(4, int(self.desc_max_nodes))
        queue = deque(safe_query_nodes)
        seen = set(safe_query_nodes)
        ordered_nodes = []
        while queue and len(ordered_nodes) < max_nodes:
            node = queue.popleft()
            ordered_nodes.append(int(node))
            for neigh in adjacency.get(int(node), []):
                if neigh in seen:
                    continue
                seen.add(neigh)
                queue.append(int(neigh))
                if len(seen) >= max_nodes:
                    break

        ordered_node_set = set(ordered_nodes)
        query_lines = []
        for pos, node_idx in enumerate(safe_query_nodes):
            role = "src" if pos == 0 else "dst" if pos == 1 else f"query_{pos}"
            query_lines.append(f"{role}: node_idx={node_idx}")

        node_lines = [f"node_idx={node_idx}" for node_idx in ordered_nodes]

        edge_lines = []
        edge_budget = max(4, max_nodes * 4)
        for src in ordered_nodes:
            for dst in adjacency.get(int(src), []):
                if dst not in ordered_node_set:
                    continue
                edge_lines.append(f"{src} -> {dst}")
                if len(edge_lines) >= edge_budget:
                    break
            if len(edge_lines) >= edge_budget:
                break

        if not node_lines or not edge_lines:
            return self._build_desc()

        parts = [
            f"Serialized query-centered subgraph for domain {self.domain}.",
            "Query nodes:",
            "\n".join(query_lines) if query_lines else "unknown",
            "Node table:",
            "\n".join(node_lines),
            "Adjacency list:",
            "\n".join(edge_lines),
        ]
        return "\n".join(part for part in parts if part)

    def _load_node_idx_lookup(self):
        if self.node_idx_by_id is not None:
            return self.node_idx_by_id
        if not self.nodes_path.exists():
            self.node_idx_by_id = {}
            return self.node_idx_by_id
        nodes_df = pd.read_csv(self.nodes_path)
        if "node_id" not in nodes_df.columns:
            self.node_idx_by_id = {}
            return self.node_idx_by_id
        if "node_idx" in nodes_df.columns:
            indices = nodes_df["node_idx"].astype(int).tolist()
        else:
            indices = list(range(len(nodes_df)))
        node_ids = nodes_df["node_id"].astype(str).tolist()
        self.node_idx_by_id = {node_id: int(node_idx) for node_id, node_idx in zip(node_ids, indices)}
        return self.node_idx_by_id

    @staticmethod
    def _coerce_node_index(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return int(value)
        text = str(value).strip()
        if text.isdigit():
            return int(text)
        return None

    def _resolve_query_nodes(self, sample, task_family):
        if task_family not in {"hop", "path_witness"}:
            return None
        sample_meta = sample.get("meta", {})
        meta = sample_meta if isinstance(sample_meta, dict) else {}
        resolved = []
        for key in ("src_idx", "dst_idx"):
            node_idx = self._coerce_node_index(meta.get(key))
            if node_idx is not None:
                resolved.append(node_idx)
        if resolved:
            return list(dict.fromkeys(resolved))

        node_lookup = self._load_node_idx_lookup()
        for key in ("src_node_id", "dst_node_id"):
            node_id = meta.get(key)
            if node_id is None:
                continue
            node_idx = node_lookup.get(str(node_id))
            if node_idx is not None:
                resolved.append(int(node_idx))
        if resolved:
            return list(dict.fromkeys(resolved))

        gt_path = sample.get("path", None) or sample.get("gt_path", None)
        if isinstance(gt_path, list) and len(gt_path) >= 2:
            path_endpoints = [gt_path[0], gt_path[-1]]
            for node_ref in path_endpoints:
                node_idx = self._coerce_node_index(node_ref)
                if node_idx is None:
                    node_idx = node_lookup.get(str(node_ref))
                if node_idx is not None:
                    resolved.append(int(node_idx))
        if resolved:
            return list(dict.fromkeys(resolved))
        return None

    @staticmethod
    def _infer_task_type(question):
        return infer_grbench_task_type(question)

    @classmethod
    def _infer_task_family(cls, question):
        return infer_grbench_task_family(question)

    @staticmethod
    def _bucket_split_counts(size, train_ratio=0.8, val_ratio=0.1):
        if size <= 0:
            return 0, 0, 0
        if size == 1:
            return 1, 0, 0
        if size == 2:
            return 1, 0, 1
        if size == 3:
            return 1, 1, 1
        train_count = int(size * train_ratio)
        val_count = int(size * val_ratio)
        test_count = size - train_count - val_count

        if val_count == 0:
            val_count = 1
            train_count -= 1
        if test_count == 0:
            test_count = 1
            train_count -= 1
        if train_count <= 0:
            train_count = 1
            if val_count > test_count:
                val_count -= 1
            else:
                test_count -= 1
        return train_count, val_count, test_count

    def __getitem__(self, index):
        """Return one example while preserving global graph node identifiers."""
        sample = self._load_sample(index)
        qid = sample.get("qid", index)
        q_text = sample['question']
        task_family = self._infer_task_family(q_text)
        prompt = self._FAMILY_PROMPTS.get(task_family, self._prompt_entity)
        question = f"{q_text}\n\n{prompt}"
        raw_meta = sample.get("meta", {})
        merged_meta = {"dataset": "grbench", "domain": self.domain}
        if isinstance(raw_meta, dict):
            merged_meta.update(raw_meta)
        query_nodes = self._resolve_query_nodes(sample, task_family)
        if self.desc_mode == "query_adjacency":
            desc = ""
        else:
            desc = self._build_desc()
        return {
            "id": qid,
            "label": sample["answer"],
            "desc": desc,
            "graph": self.graph,
            "question": question,
            "task_type": task_family,
            "gt_path": sample.get("path", None) or sample.get("gt_path", None),
            "query_nodes": query_nodes,
            "domain": self.domain,
            "meta": merged_meta,
        }

    def get_idx_split(self):
        """Return train/validation/test indices for the filtered sample view.

        Explicit release split files take precedence. The deterministic
        fallback supports compatible custom domains that omit those files.
        """
        if self._split_cache is not None:
            return self._split_cache

        explicit_split = self._load_explicit_split()
        if explicit_split is not None:
            self._split_cache = explicit_split
            return self._split_cache

        rng = random.Random(self._split_seed)
        family_to_indices = {}
        for index in range(self.num_samples):
            sample = self._load_sample(index)
            family = self._infer_task_family(sample["question"])
            family_to_indices.setdefault(family, []).append(index)

        train_indices, val_indices, test_indices = [], [], []
        for family in sorted(family_to_indices):
            indices = family_to_indices[family][:]
            rng.shuffle(indices)
            train_count, val_count, _ = self._bucket_split_counts(len(indices))
            val_end = train_count + val_count
            train_indices.extend(indices[:train_count])
            val_indices.extend(indices[train_count:val_end])
            test_indices.extend(indices[val_end:])

        train_indices.sort()
        val_indices.sort()
        test_indices.sort()
        self._split_cache = {"train": train_indices, "val": val_indices, "test": test_indices}
        return self._split_cache

    def _load_explicit_split(self):
        split_dir = self.processed_root / "split"
        if not split_dir.exists():
            return None
        required = [
            split_dir / "train_indices.txt",
            split_dir / "val_indices.txt",
            split_dir / "test_indices.txt",
        ]
        if not all(path.exists() for path in required):
            return None

        raw_splits = read_split_indices(self.processed_root)
        raw_to_filtered = {
            int(raw_index): filtered_index for filtered_index, raw_index in enumerate(self.sample_indices)
        }
        filtered_splits = {}

        for split_name, indices in raw_splits.items():
            mapped = [raw_to_filtered[idx] for idx in indices if idx in raw_to_filtered]
            if len(mapped) == len(indices):
                resolved = mapped
            elif all(0 <= int(idx) < self.num_samples for idx in indices):
                resolved = [int(idx) for idx in indices]
            else:
                raise ValueError(
                    f"Explicit GRBench split '{split_name}' under {split_dir} "
                    "contains indices outside both raw and filtered sample ranges."
                )
            deduped = sorted(dict.fromkeys(int(idx) for idx in resolved))
            filtered_splits[split_name] = deduped

        overlap = (
            set(filtered_splits["train"]) & set(filtered_splits["val"])
            or set(filtered_splits["train"]) & set(filtered_splits["test"])
            or set(filtered_splits["val"]) & set(filtered_splits["test"])
        )
        if overlap:
            raise ValueError(f"Explicit GRBench split under {split_dir} contains overlapping indices.")
        if not filtered_splits["train"]:
            raise ValueError(f"Explicit GRBench split under {split_dir} has an empty train split.")
        if not filtered_splits["test"]:
            raise ValueError(f"Explicit GRBench split under {split_dir} has an empty test split.")
        return filtered_splits

    def get_graph_meta(self):
        """Return graph dimensions and storage metadata required by models."""
        if int(self.graph.num_nodes) > int(self.metric_max_nodes):
            return set(), set()
        if self.node_id_set is None:
            if self.nodes_path.exists():
                textual_nodes = pd.read_csv(self.nodes_path, usecols=["node_id"])
                self.node_id_set = set(textual_nodes["node_id"].astype(str).tolist())
            else:
                self.node_id_set = set()
        if self.edge_set is None:
            if self.edges_path.exists():
                textual_edges = pd.read_csv(self.edges_path, usecols=["src_id", "dst_id"])
                self.edge_set = set(zip(textual_edges["src_id"].astype(str).tolist(), textual_edges["dst_id"].astype(str).tolist()))
            else:
                self.edge_set = set()
        return self.node_id_set, self.edge_set

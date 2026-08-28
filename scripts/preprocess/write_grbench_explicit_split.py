#!/usr/bin/env python3
"""Write deterministic answer-stratified train/validation/test indices."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write explicit split files for a GRBench domain.")
    parser.add_argument("--grbench-root", type=Path, default=Path("GRBENCH"))
    parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_records(data_path: Path):
    records = []
    with data_path.open("r", encoding="utf-8") as handle:
        for raw_index, line in enumerate(handle):
            if line.strip():
                row = json.loads(line)
                row["_raw_index"] = raw_index
                records.append(row)
    return records


def _apportion(capacities: dict[str, int], target: int) -> dict[str, int]:
    if target <= 0:
        return {key: 0 for key in capacities}
    total = sum(capacities.values())
    if target > total:
        raise ValueError(f"Requested split size {target} exceeds remaining capacity {total}")
    raw = {key: capacities[key] * target / total for key in capacities}
    counts = {key: min(capacities[key], math.floor(raw[key])) for key in capacities}
    while sum(counts.values()) < target:
        candidates = [key for key in capacities if counts[key] < capacities[key]]
        candidates.sort(
            key=lambda key: (raw[key] - counts[key], capacities[key] - counts[key], key),
            reverse=True,
        )
        counts[candidates[0]] += 1
    return counts


def _take(bucket_to_indices: dict[str, list[int]], allocation: dict[str, int]) -> list[int]:
    selected = []
    for key in sorted(bucket_to_indices):
        count = int(allocation.get(key, 0))
        selected.extend(bucket_to_indices[key][:count])
        del bucket_to_indices[key][:count]
    return sorted(selected)


def _write(path: Path, indices: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(int(index)) for index in indices)
    path.write_text(text + ("\n" if text else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if min(args.train_count, args.val_count, args.test_count) < 0:
        raise ValueError("Split counts must be non-negative")
    processed_root = args.grbench_root / "processed_data" / args.domain
    data_path = processed_root / "data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"GRBench data file not found: {data_path}")

    records = _load_records(data_path)
    requested_total = args.train_count + args.val_count + args.test_count
    if requested_total != len(records):
        raise ValueError(
            f"Requested split total {requested_total} must equal dataset size {len(records)} "
            "to avoid silently unused examples."
        )

    rng = random.Random(args.seed)
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in records:
        buckets[str(row.get("answer", "unknown")).strip() or "unknown"].append(int(row["_raw_index"]))
    for indices in buckets.values():
        rng.shuffle(indices)

    train = _take(buckets, _apportion({key: len(value) for key, value in buckets.items()}, args.train_count))
    val = _take(buckets, _apportion({key: len(value) for key, value in buckets.items()}, args.val_count))
    test = _take(buckets, _apportion({key: len(value) for key, value in buckets.items()}, args.test_count))

    split_dir = processed_root / "split"
    _write(split_dir / "train_indices.txt", train)
    _write(split_dir / "val_indices.txt", val)
    _write(split_dir / "test_indices.txt", test)
    manifest = {
        "domain": args.domain,
        "dataset_size": len(records),
        "actual": {"train": len(train), "val": len(val), "test": len(test)},
        "seed": args.seed,
        "stratify_by": "answer",
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

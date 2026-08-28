"""Small UTF-8 JSONL and YAML file helpers."""

import json
from pathlib import Path

import yaml


def load_jsonl(path):
    """Load non-empty JSONL records into a list."""
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def dump_jsonl(records, path):
    """Write iterable records as UTF-8 JSONL, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_yaml(path):
    """Load one YAML document with ``safe_load``."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

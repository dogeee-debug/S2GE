import csv
import json
from pathlib import Path

import pandas as pd


def _read_index_file(path: Path):
    """Read one integer sample index per non-empty line."""

    indices = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                indices.append(int(line))
    return indices


def read_split_indices(root: Path):
    """Load train, validation, and test split indices from a processed dataset."""

    split_dir = root / 'split'
    return {
        'train': _read_index_file(split_dir / 'train_indices.txt'),
        'val': _read_index_file(split_dir / 'val_indices.txt'),
        'test': _read_index_file(split_dir / 'test_indices.txt'),
    }


def read_tabular_subset(path: Path, required_columns, optional_columns=(), sep=","):
    """Read required columns and any available optional columns from a table."""

    header = pd.read_csv(path, sep=sep, nrows=0)
    available = set(header.columns.tolist())
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    selected = list(required_columns) + [column for column in optional_columns if column in available]
    return pd.read_csv(path, sep=sep, usecols=selected)


def load_metadata_records(root: Path):
    """Load sample metadata from the first supported JSONL or CSV file."""

    candidates = [
        root / 'samples.jsonl',
        root / 'dataset.jsonl',
        root / 'questions.jsonl',
    ]
    for candidate in candidates:
        if candidate.exists():
            records = []
            with candidate.open('r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            return records
    csv_candidates = [root / 'questions.csv', root / 'samples.csv']
    for candidate in csv_candidates:
        if candidate.exists():
            with candidate.open('r', encoding='utf-8', newline='') as f:
                return list(csv.DictReader(f))
    raise FileNotFoundError(f'No offline metadata file found under {root}')


def resolve_dataset_root(base_root: Path, dataset_name: str):
    """Return the dataset subdirectory when present, otherwise the base root."""

    candidates = [base_root / dataset_name, base_root]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

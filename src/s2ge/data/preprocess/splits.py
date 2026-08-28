"""Deterministic dataset split generation and persistence."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def write_split_files(split_dir: Path, train_indices, val_indices, test_indices):
    """Write train/validation/test indices to a processed dataset directory."""
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train_indices.txt").write_text("\n".join(map(str, train_indices)), encoding="utf-8")
    (split_dir / "val_indices.txt").write_text("\n".join(map(str, val_indices)), encoding="utf-8")
    (split_dir / "test_indices.txt").write_text("\n".join(map(str, test_indices)), encoding="utf-8")


def generate_random_split(num_samples: int, split_dir: Path, seed: int = 42):
    """Generate seeded 60/20/20 split files for compatible custom data."""
    indices = np.arange(num_samples)
    train_indices, temp_indices = train_test_split(indices, test_size=0.4, random_state=seed)
    val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=seed)
    write_split_files(split_dir, train_indices, val_indices, test_indices)

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def datasets_raw_root() -> Path:
    return repo_root() / "datasets_raw"


def datasets_processed_root() -> Path:
    return repo_root() / "datasets_processed"


def docs_root() -> Path:
    return repo_root()

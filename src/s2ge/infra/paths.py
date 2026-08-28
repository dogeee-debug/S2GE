"""Repository-relative default paths used by CLI resolution."""

from pathlib import Path


def repo_root() -> Path:
    """Return the root of the installed/editable source checkout."""
    return Path(__file__).resolve().parents[3]


def datasets_raw_root() -> Path:
    """Return the default raw-dataset directory."""
    return repo_root() / "datasets_raw"


def datasets_processed_root() -> Path:
    """Return the default processed-dataset directory."""
    return repo_root() / "datasets_processed"


def docs_root() -> Path:
    """Return the release root retained by the historical docs API."""
    return repo_root()

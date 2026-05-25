from pathlib import Path


def _assert_exists(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label}: missing -> {path}")


def validate_dataset_ready(dataset_name: str, root: Path, grbench_root: Path | None = None, grbench_domain: str = "dblp"):
    if dataset_name == "grbench":
        if grbench_root is None:
            raise ValueError("grbench_root is required for grbench validation")
        processed_root = grbench_root / "processed_data" / grbench_domain
        for filename in ["data.json", "graph.pt", "nodes.csv", "edges.csv"]:
            _assert_exists(processed_root / filename, f"grbench {filename}")
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

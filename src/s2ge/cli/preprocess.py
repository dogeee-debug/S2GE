"""Dataset preprocessing entry point used by the release scripts."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for candidate in [ROOT, ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from s2ge.data.preprocess import (
    build_grbench,
    validate_dataset_ready,
)
from s2ge.infra.paths import datasets_processed_root, datasets_raw_root

S2GE_DOMAINS = ("dblp", "biomedical", "goodreads", "pubmed")


def main(argv=None):
    """Build the requested processed dataset after validating source paths."""
    parser = argparse.ArgumentParser(description="Dispatch local preprocessing jobs")
    parser.add_argument("--dataset", required=True, choices=["grbench"])
    parser.add_argument("--data-root", default=str(datasets_processed_root()))
    parser.add_argument("--raw-root", default=str(datasets_raw_root()))
    parser.add_argument("--grbench-root", default=str(ROOT / "GRBENCH"))
    parser.add_argument(
        "--domain",
        default="dblp",
        help="GRBench domain name, including derived domains such as pubmed_hopqa_train2000_val200_test1000",
    )
    parser.add_argument("--model-name", default="sbert")
    parser.add_argument("--grbench-feature-mode", choices=["light", "text"], default="light")
    parser.add_argument("--grbench-embed-dim", type=int, default=1024)
    parser.add_argument("--grbench-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--grbench-cache-embeddings", action="store_true")
    parser.add_argument("--infection-enabled", action="store_true")
    parser.add_argument("--infection-k", type=int, default=0)
    parser.add_argument("--infection-clip-max", type=int, default=255)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Build graph.pt/nodes.csv/edges.csv without requiring QA data.json yet",
    )
    args = parser.parse_args(argv)

    if args.validate_only and args.graph_only:
        parser.error("--validate-only and --graph-only cannot be used together")

    data_root = Path(args.data_root)
    grbench_root = Path(args.grbench_root)

    if args.validate_only:
        validate_dataset_ready("grbench", data_root, grbench_root=grbench_root, grbench_domain=args.domain)
        print("[OK] Validation completed.")
        return

    if args.dataset == "grbench":
        build_grbench(
            grbench_root,
            domain=args.domain,
            model_name=args.grbench_embedding_model if args.grbench_feature_mode == "text" else args.model_name,
            feature_mode=args.grbench_feature_mode,
            embed_dim=args.grbench_embed_dim,
            cache_embeddings=args.grbench_cache_embeddings,
            infection_enabled=args.infection_enabled,
            infection_k=args.infection_k,
            infection_clip_max=args.infection_clip_max,
        )
        if args.graph_only:
            graph_root = grbench_root / "processed_data" / args.domain
            for filename in ("graph.pt", "nodes.csv", "edges.csv"):
                path = graph_root / filename
                if not path.exists():
                    raise FileNotFoundError(f"grbench {filename}: missing -> {path}")
            print("[OK] Graph preprocessing completed; QA data can now be generated.")
            return
        validate_dataset_ready("grbench", data_root, grbench_root=grbench_root, grbench_domain=args.domain)
        return
    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == '__main__':
    main()

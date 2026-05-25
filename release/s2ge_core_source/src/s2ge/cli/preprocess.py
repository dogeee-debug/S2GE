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
    parser = argparse.ArgumentParser(description="Dispatch local preprocessing jobs")
    parser.add_argument("--dataset", required=True, choices=["grbench"])
    parser.add_argument("--data-root", default=str(datasets_processed_root()))
    parser.add_argument("--raw-root", default=str(datasets_raw_root()))
    parser.add_argument("--grbench-root", default=str(ROOT / "GRBENCH"))
    parser.add_argument("--domain", default="dblp", choices=S2GE_DOMAINS)
    parser.add_argument("--model-name", default="sbert")
    parser.add_argument("--grbench-feature-mode", choices=["light", "text"], default="light")
    parser.add_argument("--grbench-embed-dim", type=int, default=1024)
    parser.add_argument("--grbench-embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--grbench-cache-embeddings", action="store_true")
    parser.add_argument("--infection-enabled", action="store_true")
    parser.add_argument("--infection-k", type=int, default=0)
    parser.add_argument("--infection-clip-max", type=int, default=255)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

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
        validate_dataset_ready("grbench", data_root, grbench_root=grbench_root, grbench_domain=args.domain)
        return
    raise ValueError(f"Unknown dataset: {args.dataset}")


if __name__ == '__main__':
    main()

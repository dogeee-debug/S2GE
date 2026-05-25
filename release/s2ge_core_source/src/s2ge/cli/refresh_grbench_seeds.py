import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for candidate in [ROOT, ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from s2ge.data.preprocess import refresh_grbench_seed_nodes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refresh GRBench persisted seed nodes from collected node scores.")
    parser.add_argument("--grbench-root", default=str(ROOT / "GRBENCH"))
    parser.add_argument("--domain", default="dblp")
    parser.add_argument("--score-path", required=True)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--score-weight", type=float, default=0.7)
    parser.add_argument("--prior-weight", type=float, default=0.3)
    parser.add_argument("--exploration-ratio", type=float, default=0.25)
    args = parser.parse_args(argv)

    seed_nodes = refresh_grbench_seed_nodes(
        args.grbench_root,
        domain=args.domain,
        score_path=args.score_path,
        budget=args.budget,
        score_weight=args.score_weight,
        prior_weight=args.prior_weight,
        exploration_ratio=args.exploration_ratio,
    )
    print({"updated_seed_nodes": int(seed_nodes.numel()), "domain": args.domain})


if __name__ == "__main__":
    main()

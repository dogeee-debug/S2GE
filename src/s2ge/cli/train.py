"""Training entry point for released S2GE configurations."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for candidate in [ROOT, ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from s2ge.cli.common import parse_args
from s2ge.infra.distributed import init_distributed
from s2ge.infra.env import apply_offline_env, validate_required_files


def main(argv=None):
    """Initialize distributed state, run training, and release resources."""
    args = parse_args(argv)
    from s2ge.engine.trainer import Trainer

    apply_offline_env(args)
    init_distributed(args)
    if args.local_rank >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    validate_required_files(args)
    trainer = Trainer(args)
    trainer.fit()


if __name__ == '__main__':
    main()

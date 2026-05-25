import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for candidate in [ROOT, ROOT / "src"]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import torch

from s2ge.cli.common import parse_args
from s2ge.engine.trainer import Trainer
from s2ge.infra.env import apply_offline_env, validate_required_files


def main(argv=None):
    args = parse_args(argv)
    apply_offline_env(args)

    # Evaluation runs on one local device by default. This avoids launching a
    # distributed DeepSpeed job when only checkpoint scoring is needed.
    args.use_deepspeed = False
    args.local_rank = -1
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    validate_required_files(args)
    trainer = Trainer(args)
    if not str(getattr(args, "checkpoint_path", "")).strip():
        trainer.raw_model = trainer.checkpoints.load_best(trainer.raw_model)
    acc = trainer.test()
    print(f"Test Acc {acc}")


if __name__ == "__main__":
    main()

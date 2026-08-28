"""Runtime infrastructure for paths, logging, checkpoints, and distribution."""

from s2ge.infra.checkpoint import CheckpointManager
from s2ge.infra.distributed import barrier, init_distributed, is_deepspeed_enabled, is_main_process, local_rank, rank, world_size
from s2ge.infra.env import apply_offline_env, require_exists, validate_required_files
from s2ge.infra.logging import RankZeroLogger
from s2ge.infra.paths import datasets_processed_root, datasets_raw_root, docs_root, repo_root

__all__ = [
    "CheckpointManager",
    "rank",
    "world_size",
    "local_rank",
    "barrier",
    "is_main_process",
    "is_deepspeed_enabled",
    "init_distributed",
    "apply_offline_env",
    "require_exists",
    "validate_required_files",
    "RankZeroLogger",
    "repo_root",
    "datasets_raw_root",
    "datasets_processed_root",
    "docs_root",
]

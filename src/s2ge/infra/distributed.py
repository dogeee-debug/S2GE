"""Minimal distributed-process helpers shared by CLI and Trainer."""

import os
import torch.distributed as dist

try:
    import deepspeed
except ImportError:
    deepspeed = None


def is_deepspeed_enabled(args) -> bool:
    """Return whether this run explicitly requested DeepSpeed."""
    return bool(getattr(args, 'use_deepspeed', False) or getattr(args, 'deepspeed', ''))


def init_distributed(args):
    """Initialize DeepSpeed communication only for an enabled launch."""
    if is_deepspeed_enabled(args):
        if deepspeed is None:
            raise ImportError('DeepSpeed requested but not installed.')
        if not deepspeed.comm.is_initialized():
            deepspeed.init_distributed()
        args.use_deepspeed = True
        args.local_rank = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', -1)))


def rank() -> int:
    """Return the distributed rank, or zero outside a distributed run."""
    if deepspeed is not None and deepspeed.comm.is_initialized():
        return deepspeed.comm.get_rank()
    return 0


def world_size() -> int:
    """Return the distributed world size, or one for local execution."""
    if deepspeed is not None and deepspeed.comm.is_initialized():
        return deepspeed.comm.get_world_size()
    return 1


def local_rank(default=-1) -> int:
    """Read the launcher-provided local rank."""
    return int(os.environ.get('LOCAL_RANK', default))


def is_main_process() -> bool:
    """Return whether the caller is rank zero."""
    return rank() == 0


def barrier():
    """Synchronize initialized DeepSpeed workers; otherwise do nothing."""
    if deepspeed is not None and deepspeed.comm.is_initialized():
        deepspeed.comm.barrier()


def shutdown_distributed():
    """Destroy an initialized torch.distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

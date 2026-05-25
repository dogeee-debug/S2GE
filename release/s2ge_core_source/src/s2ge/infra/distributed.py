import os
import torch.distributed as dist

try:
    import deepspeed
except ImportError:
    deepspeed = None


def is_deepspeed_enabled(args) -> bool:
    return bool(getattr(args, 'use_deepspeed', False) or getattr(args, 'deepspeed', ''))


def init_distributed(args):
    if is_deepspeed_enabled(args):
        if deepspeed is None:
            raise ImportError('DeepSpeed requested but not installed.')
        if not deepspeed.comm.is_initialized():
            deepspeed.init_distributed()
        args.use_deepspeed = True
        args.local_rank = int(os.environ.get('LOCAL_RANK', getattr(args, 'local_rank', -1)))


def rank() -> int:
    if deepspeed is not None and deepspeed.comm.is_initialized():
        return deepspeed.comm.get_rank()
    return 0


def world_size() -> int:
    if deepspeed is not None and deepspeed.comm.is_initialized():
        return deepspeed.comm.get_world_size()
    return 1


def local_rank(default=-1) -> int:
    return int(os.environ.get('LOCAL_RANK', default))


def is_main_process() -> bool:
    return rank() == 0


def barrier():
    if deepspeed is not None and deepspeed.comm.is_initialized():
        deepspeed.comm.barrier()


def shutdown_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

import os
from pathlib import Path


def apply_offline_env(args):
    if getattr(args, 'offline', False):
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def require_exists(path_like, label: str):
    path = Path(path_like)
    if not path.exists():
        raise FileNotFoundError(f'{label} not found: {path}')
    return path


def validate_required_files(args):
    """Fail early when offline mode needs local model or checkpoint files."""

    if getattr(args, 'offline', False):
        require_exists(args.llm_model_path, 'Offline LLM model path')
        if args.graph_projector_mode == 'llaga' and args.llaga_projector_ckpt:
            require_exists(args.llaga_projector_ckpt, 'LLaGA projector checkpoint')

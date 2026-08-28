"""Offline-mode setup and required-file validation."""

import os
from pathlib import Path


def apply_offline_env(args):
    """Set Hugging Face offline flags when requested by the release config."""
    if getattr(args, 'offline', False):
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def require_exists(path_like, label: str):
    """Return an existing path or raise a label-rich error."""
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

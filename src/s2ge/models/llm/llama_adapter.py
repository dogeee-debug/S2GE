"""Hugging Face causal-LM loading with explicit offline validation."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_llm_loading_kwargs(args):
    """Translate release precision/device options into HF loading kwargs."""
    kwargs = {"revision": "main"}
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", -1)))
    if getattr(args, "use_deepspeed", False) and local_rank >= 0:
        kwargs["device_map"] = {"": local_rank}
    else:
        kwargs["max_memory"] = {i: f"{size}GiB" for i, size in enumerate(args.max_memory)}
        kwargs["device_map"] = "auto"
    return kwargs


def _offline_mode_enabled(args=None):
    return bool(getattr(args, "offline", False)) or any(
        os.getenv(name, "0") == "1"
        for name in ("S2GE_OFFLINE_MODE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _local_files_only(args, model_path):
    return os.path.exists(str(model_path)) or _offline_mode_enabled(args)


def _validate_local_dir_exists(model_path, label: str):
    path = os.fspath(model_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found locally: {path}")
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label} must be a directory: {path}")
    return path


def _validate_local_config_file(model_path, label: str):
    path = _validate_local_dir_exists(model_path, label)
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"{label} is incomplete, missing config.json: {path}")
    return path


def _validate_local_tokenizer_dir(model_path, label: str):
    path = _validate_local_config_file(model_path, label)
    tokenizer_candidates = (
        os.path.join(path, "tokenizer.json"),
        os.path.join(path, "tokenizer.model"),
        os.path.join(path, "tokenizer_config.json"),
        os.path.join(path, "vocab.json"),
    )
    if not any(os.path.exists(candidate) for candidate in tokenizer_candidates):
        raise FileNotFoundError(f"{label} is incomplete, missing tokenizer files: {path}")
    return path


def _validate_local_model_dir(model_path, label: str):
    path = _validate_local_config_file(model_path, label)
    weight_candidates = (
        os.path.join(path, "pytorch_model.bin"),
        os.path.join(path, "model.safetensors"),
        os.path.join(path, "model.safetensors.index.json"),
    )
    if not any(os.path.exists(candidate) for candidate in weight_candidates):
        raise FileNotFoundError(f"{label} is incomplete, missing model weights: {path}")
    return path


def load_quantization_config(args):
    """Build the optional bitsandbytes 4-bit quantization configuration."""
    if not getattr(args, "llm_load_4bit", False):
        return None
    from transformers import BitsAndBytesConfig

    compute_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_path, use_fast=False):
    """Load a tokenizer and ensure that a usable pad token is configured."""
    if os.path.exists(str(model_path)):
        _validate_local_tokenizer_dir(model_path, "LLM tokenizer path")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=use_fast,
        revision="main",
        local_files_only=_local_files_only(None, model_path),
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return tokenizer


def load_causal_lm(args, model_path):
    """Load a causal LM while enforcing local-only behavior in offline mode."""
    quantization_config = load_quantization_config(args)
    kwargs = resolve_llm_loading_kwargs(args)
    model_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16
    if os.path.exists(str(model_path)):
        _validate_local_model_dir(model_path, "LLM model path")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        local_files_only=_local_files_only(args, model_path),
        **kwargs,
    )
    return model

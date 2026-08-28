"""Optional DeepSpeed wrapper that preserves a single Trainer interface."""
import json
from pathlib import Path

import torch

from s2ge.infra.distributed import is_deepspeed_enabled

try:
    import deepspeed
except ImportError:
    deepspeed = None


class DeepSpeedRuntime:
    """Initialize either a local PyTorch model or a DeepSpeed engine."""
    def __init__(self, args):
        self.args = args

    def _has_hf_device_map(self, module):
        if module is None:
            return False
        device_map = getattr(module, "hf_device_map", None)
        return isinstance(device_map, dict) and len(device_map) > 0

    def _is_hf_sharded_model(self, model):
        candidates = [
            model,
            getattr(model, "model", None),
            getattr(getattr(model, "model", None), "base_model", None),
            getattr(getattr(getattr(model, "model", None), "base_model", None), "model", None),
        ]
        for candidate in candidates:
            if self._has_hf_device_map(candidate):
                return True
        return False

    def _resolve_model_device(self, model):
        if hasattr(model, "llm_input_device"):
            return model.llm_input_device
        if hasattr(model, "device"):
            return model.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _resolve_config(self):
        """Return an explicit DeepSpeed config or a protocol-equivalent default.

        The generated default deliberately controls only batching, precision,
        and gradient clipping. Optimizer construction and the learning-rate
        schedule remain owned by S2GE, matching the local PyTorch path.
        """
        configured = str(getattr(self.args, "deepspeed", "") or "").strip()
        if configured:
            config_path = Path(configured)
            if not config_path.exists():
                raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")
            return json.loads(config_path.read_text(encoding="utf-8"))
        config = {
            "train_micro_batch_size_per_gpu": int(self.args.batch_size),
            "gradient_accumulation_steps": max(1, int(self.args.grad_steps)),
            "gradient_clipping": 0.1,
            "zero_optimization": {"stage": 0},
        }
        if bool(getattr(self.args, "bf16", False)):
            config["bf16"] = {"enabled": True}
        return config

    def initialize(self, model, optimizer, model_parameters):
        """Return ``(runtime_model, optimizer, device, unwrapped_model)``."""
        if not is_deepspeed_enabled(self.args):
            if self._is_hf_sharded_model(model):
                device = self._resolve_model_device(model)
                return model, optimizer, device, model
            device = self._resolve_model_device(model)
            model = model.to(device)
            return model, optimizer, device, model
        if deepspeed is None:
            raise ImportError('DeepSpeed requested but not installed.')
        model_engine, optimizer, _, _ = deepspeed.initialize(
            args=self.args,
            model=model,
            optimizer=optimizer,
            model_parameters=model_parameters,
            config=self._resolve_config(),
        )
        device = torch.device(f'cuda:{model_engine.local_rank}')
        return model_engine, optimizer, device, model_engine.module

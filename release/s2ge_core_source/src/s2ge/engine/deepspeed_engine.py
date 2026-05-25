'''
'''
import torch

from s2ge.infra.distributed import is_deepspeed_enabled

try:
    import deepspeed
except ImportError:
    deepspeed = None


class DeepSpeedRuntime:
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

    def initialize(self, model, optimizer, model_parameters):
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
            config=self.args.deepspeed or None,
        )
        device = torch.device(f'cuda:{model_engine.local_rank}')
        return model_engine, optimizer, device, model_engine.module

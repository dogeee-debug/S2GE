from pathlib import Path
import torch
import os


def _fs_path(path: Path) -> str:
    path_str = str(path)
    if os.name == 'nt' and not path_str.startswith('\\\\?\\'):
        return '\\\\?\\' + str(path.resolve())
    return path_str


class CheckpointManager:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.output_dir) / args.dataset
        self.root.mkdir(parents=True, exist_ok=True)

    def _stem(self):
        return (
            f"model_{self.args.model_name}"
            f"_llm_{self.args.llm_model_name}"
            f"_frozen_{self.args.llm_frozen}"
            f"_txt_{self.args.max_txt_len}"
            f"_new_{self.args.max_new_tokens}"
            f"_gnn_{self.args.gnn_model_name}"
            f"_layers_{getattr(self.args, 'gnn_num_layers', 'na')}"
            f"_graph_tokens_{getattr(self.args, 'graph_token_mode', 'single')}"
            f"_epochs_{self.args.num_epochs}"
            f"_seed{self.args.seed}"
        )

    def checkpoint_path(self, suffix='best'):
        return self.root / f"{self._stem()}_checkpoint_{suffix}.pth"

    def prediction_path(self):
        return self.root / f"{self._stem()}_predictions.jsonl"

    def epoch_prediction_path(self, split: str, epoch: int):
        split = str(split).strip().lower()
        return self.root / f"{self._stem()}_{split}_epoch_{epoch}_predictions.jsonl"

    def node_score_path(self):
        return self.root / f"{self._stem()}_node_scores.pt"

    def save(self, model, optimizer, epoch: int, is_best=False, train_state=None, suffix=None):
        trainable_only = bool(getattr(self.args, 'checkpoint_trainable_only', True))
        trainable_names = {name for name, param in model.named_parameters() if param.requires_grad} if trainable_only else set()
        full_state_dict = model.state_dict()
        state_dict = {}
        for name, tensor in full_state_dict.items():
            if trainable_only and trainable_names and name not in trainable_names:
                continue
            if hasattr(tensor, "detach"):
                state_dict[name] = tensor.detach().cpu()
            else:
                state_dict[name] = tensor
        payload = {
            'model': state_dict,
            'optimizer': (
                None
                if optimizer is None or getattr(self.args, 'skip_checkpoint_optimizer', False)
                else optimizer.state_dict()
            ),
            'config': vars(self.args),
            'epoch': epoch,
            'trainable_only': trainable_only,
            'train_state': train_state or {},
        }
        resolved_suffix = str(suffix) if suffix is not None else ('best' if is_best else str(epoch))
        path = self.checkpoint_path(resolved_suffix)
        # Keep the temporary name short so atomic saves work on Windows paths
        # even when the final checkpoint stem records a long experiment config.
        tmp_path = path.with_name(f".tmp_checkpoint_{resolved_suffix}.pth")
        try:
            torch.save(payload, _fs_path(tmp_path))
            os.replace(_fs_path(tmp_path), _fs_path(path))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise
        return Path(_fs_path(path))

    def load_best(self, model):
        path = self.checkpoint_path('best')
        # PyTorch 2.6+: weights_only=True by default, set to False for checkpoints with custom classes
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        return model

"""Core S2GE training orchestration.

The :class:`Trainer` owns the standard release path: data loaders, model and
optimizer setup, checkpoint resume/save, epoch training, validation, early
stopping, and final evaluation.

The file also retains several opt-in research controls used during development
(sub-epoch monitoring, dynamic alignment weighting, alignment LR compensation,
and axis-momentum reset).  They are disabled by the paper configurations and
are grouped and labelled below so that they cannot be mistaken for the fixed
camera-ready training protocol.
"""

import gc
import random
from contextlib import suppress

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from s2ge.data.registry import build_collate_fn, build_dataset
from s2ge.engine.deepspeed_engine import DeepSpeedRuntime
from s2ge.engine.evaluator import compute_prediction_diagnostics, evaluate_dataset, evaluate_loss, move_batch_to_device, predict_to_file
from s2ge.engine.hooks import EarlyStopHook
from s2ge.engine.optim import adjust_learning_rate, build_optimizer
from s2ge.engine.state import TrainState
from s2ge.infra.checkpoint import CheckpointManager
from s2ge.infra.distributed import barrier, is_deepspeed_enabled, is_main_process, rank, shutdown_distributed, world_size
from s2ge.infra.logging import RankZeroLogger
from s2ge.models import build_model
from s2ge.utils.seed import seed_everything


def _accumulation_window(step: int, num_batches: int, grad_steps: int) -> tuple[int, bool]:
    """Return the current accumulation-window size and optimizer boundary.

    The final window may be shorter than ``grad_steps``. Scaling by its actual
    size prevents the last partial window from receiving a smaller update.
    """
    grad_steps = max(int(grad_steps), 1)
    window_start = (int(step) // grad_steps) * grad_steps
    window_size = min(grad_steps, max(int(num_batches) - window_start, 1))
    is_boundary = (int(step) + 1) % grad_steps == 0 or (int(step) + 1) == int(num_batches)
    return window_size, is_boundary


class Trainer:
    """Run standard S2GE training with optional, explicitly enabled diagnostics."""

    def __init__(self, args):
        """Build data, model, optimizer, checkpointing, and opt-in controllers.

        Public ``args`` names mirror YAML keys for release compatibility.
        Mutable runtime/controller fields use a leading underscore so they are
        not confused with user-configurable protocol settings.
        """
        self.args = args
        seed_everything(seed=args.seed + rank())
        self.logger = RankZeroLogger(args)
        self.dataset = build_dataset(args)
        self.idx_split = self.dataset.get_idx_split()
        self.train_dataset = Subset(self.dataset, self.idx_split['train'])
        self.val_dataset = Subset(self.dataset, self.idx_split['val'])
        self.test_dataset = Subset(self.dataset, self.idx_split['test'])
        self.collate_fn = build_collate_fn(args)
        self.train_sampler = None
        train_num_workers = max(0, int(getattr(args, "train_num_workers", 0)))
        eval_num_workers = max(0, int(getattr(args, "eval_num_workers", 0)))
        persistent_workers = bool(getattr(args, "persistent_workers", False))
        train_loader_kwargs = {
            "batch_size": args.batch_size,
            "drop_last": True,
            "pin_memory": True,
            "shuffle": (self.train_sampler is None),
            "sampler": self.train_sampler,
            "collate_fn": self.collate_fn,
            "num_workers": train_num_workers,
            "persistent_workers": persistent_workers and train_num_workers > 0,
        }
        eval_loader_kwargs = {
            "batch_size": args.eval_batch_size,
            "drop_last": False,
            "pin_memory": True,
            "shuffle": False,
            "collate_fn": self.collate_fn,
            "num_workers": eval_num_workers,
            "persistent_workers": persistent_workers and eval_num_workers > 0,
        }
        if is_deepspeed_enabled(args) and world_size() > 1:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size(), rank=rank(), shuffle=True, drop_last=False)
        train_loader_kwargs["shuffle"] = (self.train_sampler is None)
        train_loader_kwargs["sampler"] = self.train_sampler
        self.train_loader = DataLoader(self.train_dataset, **train_loader_kwargs)
        self.val_loader = DataLoader(self.val_dataset, **eval_loader_kwargs)
        self.test_loader = DataLoader(self.test_dataset, **eval_loader_kwargs)
        # Optional research monitor. This is not part of the fixed paper
        # protocol and runs only when monitor_subset_enabled is explicitly set.
        self.monitor_subset_enabled = bool(getattr(args, "monitor_subset_enabled", False))
        self.monitor_subset_size = max(0, int(getattr(args, "monitor_subset_size", 0)))
        self.monitor_every_optimizer_steps = max(0, int(getattr(args, "monitor_every_optimizer_steps", 0)))
        self.monitor_eval_batch_size = max(1, int(getattr(args, "monitor_eval_batch_size", 0) or args.eval_batch_size))
        self.monitor_subset_seed = int(getattr(args, "monitor_subset_seed", args.seed))
        self.monitor_digit_min_length = max(1, int(getattr(args, "monitor_digit_min_length", 6)))
        self.monitor_keep_predictions = bool(getattr(args, "monitor_keep_predictions", False))
        self.monitor_save_best_checkpoint = bool(getattr(args, "monitor_save_best_checkpoint", True))
        self.monitor_drives_axis_controller = bool(getattr(args, "monitor_drives_axis_controller", True))
        self.monitor_best_digit8_threshold = float(
            getattr(args, "monitor_best_digit8_threshold", getattr(args, "best_checkpoint_digit8_threshold", 0.1))
        )
        self._monitor_best_optimizer_step = -1
        self._monitor_best_score = float("-inf")
        self._monitor_best_digit8 = None
        self._monitor_best_legal_integer = None
        self.monitor_loader = None
        self.monitor_dataset = None
        if self.monitor_subset_enabled and self.monitor_subset_size > 0:
            monitor_rng = random.Random(self.monitor_subset_seed)
            monitor_indices = list(range(len(self.val_dataset)))
            monitor_rng.shuffle(monitor_indices)
            monitor_indices = sorted(monitor_indices[: min(self.monitor_subset_size, len(monitor_indices))])
            self.monitor_dataset = Subset(self.val_dataset, monitor_indices)
            monitor_loader_kwargs = dict(eval_loader_kwargs)
            monitor_loader_kwargs["batch_size"] = self.monitor_eval_batch_size
            self.monitor_loader = DataLoader(self.monitor_dataset, **monitor_loader_kwargs)
        model = build_model(args, graph_type=self.dataset.graph_type, init_prompt=getattr(self.dataset, 'prompt', None))
        self.optimizer, trainable_params = build_optimizer(model, args)
        self.runtime = DeepSpeedRuntime(args)
        self.runtime_model, self.optimizer, self.device, self.model = self.runtime.initialize(
            model,
            self.optimizer,
            trainable_params,
        )
        self.state = TrainState()
        self.checkpoints = CheckpointManager(args)
        self.early_stop = EarlyStopHook(args.patience)
        self._checkpoint_failed = False
        self.best_checkpoint_metric = str(getattr(args, "best_checkpoint_metric", "val_loss"))
        self.best_checkpoint_digit8_threshold = float(getattr(args, "best_checkpoint_digit8_threshold", 0.1))
        self.best_checkpoint_fallback_to_val_loss = bool(getattr(args, "best_checkpoint_fallback_to_val_loss", False))
        self._best_checkpoint_epoch = -1
        self._best_checkpoint_score = float("inf") if self.best_checkpoint_metric == "val_loss" else float("-inf")
        self._best_checkpoint_digit8 = None
        self._best_checkpoint_legal_integer = None
        # Optional research controllers. All are opt-in and disabled in the
        # released main configurations, which use a fixed align_lambda=0.25.
        self.dynamic_align_lambda_enabled = bool(getattr(args, "dynamic_align_lambda_enabled", False))
        self._dynamic_align_progress_ema = None
        self.dynamic_align_progress_ema_beta = float(getattr(args, "dynamic_align_progress_ema_beta", 0.7))
        self.dynamic_align_progress_clip_min = float(getattr(args, "dynamic_align_progress_clip_min", 0.0))
        self.dynamic_align_progress_clip_max = float(getattr(args, "dynamic_align_progress_clip_max", 1.0))
        self.dynamic_align_progress_intercept = float(getattr(args, "dynamic_align_progress_intercept", 0.3167507165736947))
        self.dynamic_align_progress_coef_em = float(getattr(args, "dynamic_align_progress_coef_em", 1.13560542))
        self.dynamic_align_progress_coef_digit8 = float(getattr(args, "dynamic_align_progress_coef_digit8", -0.38246612))
        self.dynamic_align_progress_coef_single_digit = float(getattr(args, "dynamic_align_progress_coef_single_digit", 0.36927145))
        self.dynamic_align_lambda_delta_max = max(0.0, float(getattr(args, "dynamic_align_lambda_delta_max", 0.03)))
        self.dynamic_align_hold_progress_threshold = float(getattr(args, "dynamic_align_hold_progress_threshold", 1.1))
        self.dynamic_align_hold_lambda = max(0.0, float(getattr(args, "dynamic_align_hold_lambda", 0.0)))
        self.dynamic_align_digit8_target_enabled = bool(getattr(args, "dynamic_align_digit8_target_enabled", False))
        self.dynamic_align_digit8_target = float(getattr(args, "dynamic_align_digit8_target", 0.0))
        self.dynamic_align_digit8_target_gain = float(getattr(args, "dynamic_align_digit8_target_gain", 0.0))
        self.dynamic_align_digit8_target_deadband = max(0.0, float(getattr(args, "dynamic_align_digit8_target_deadband", 0.0)))
        self.dynamic_align_digit8_band_enabled = bool(getattr(args, "dynamic_align_digit8_band_enabled", False))
        self.dynamic_align_digit8_band_low = float(getattr(args, "dynamic_align_digit8_band_low", 0.0))
        self.dynamic_align_digit8_band_high = float(getattr(args, "dynamic_align_digit8_band_high", 0.0))
        if self.dynamic_align_digit8_band_high < self.dynamic_align_digit8_band_low:
            self.dynamic_align_digit8_band_low, self.dynamic_align_digit8_band_high = (
                self.dynamic_align_digit8_band_high,
                self.dynamic_align_digit8_band_low,
            )
        self.dynamic_align_digit8_band_gain_above = float(getattr(args, "dynamic_align_digit8_band_gain_above", 0.0))
        self.dynamic_align_digit8_band_gain_below = float(getattr(args, "dynamic_align_digit8_band_gain_below", 0.0))
        self.dynamic_align_digit8_band_deadband = max(0.0, float(getattr(args, "dynamic_align_digit8_band_deadband", 0.0)))
        self.align_lr_compensation_enabled = bool(getattr(args, "align_lr_compensation_enabled", False))
        self.align_lr_compensation_start_epoch = float(getattr(args, "align_lr_compensation_start_epoch", 2.0))
        self.align_lr_compensation_ref_lr = float(getattr(args, "align_lr_compensation_ref_lr", 0.0)) or float(args.lr)
        self.align_lr_compensation_min_scale = float(getattr(args, "align_lr_compensation_min_scale", 1.0))
        self.align_lr_compensation_max_scale = float(getattr(args, "align_lr_compensation_max_scale", 4.0))
        if self.align_lr_compensation_max_scale < self.align_lr_compensation_min_scale:
            self.align_lr_compensation_min_scale, self.align_lr_compensation_max_scale = (
                self.align_lr_compensation_max_scale,
                self.align_lr_compensation_min_scale,
            )
        self.axis_momentum_reset_enabled = bool(getattr(args, "axis_momentum_reset_enabled", False))
        self.axis_momentum_reset_threshold = float(getattr(args, "axis_momentum_reset_threshold", 0.55))
        self.axis_momentum_reset_guard_band = max(0.0, float(getattr(args, "axis_momentum_reset_guard_band", 0.05)))
        self.axis_momentum_reset_delta_tol = max(0.0, float(getattr(args, "axis_momentum_reset_delta_tol", 0.03)))
        self.axis_momentum_reset_cooldown_epochs = max(0, int(getattr(args, "axis_momentum_reset_cooldown_epochs", 1)))
        self.axis_momentum_reset_clear_exp_avg_sq = bool(
            getattr(args, "axis_momentum_reset_clear_exp_avg_sq", False)
        )
        # Mutable controller state is private; checkpoint payload keys remain
        # stable for backward compatibility with existing experimental runs.
        self._axis_progress_previous = None
        self._axis_momentum_reset_cooldown = 0
        self._axis_momentum_reset_count = 0
        base_align_lambda = float(getattr(self.model, "align_lambda", getattr(args, "align_lambda", 0.0)))
        self._base_align_lambda = base_align_lambda
        self.dynamic_align_lambda_min = float(getattr(args, "dynamic_align_lambda_min", base_align_lambda))
        self.dynamic_align_lambda_max = float(getattr(args, "dynamic_align_lambda_max", base_align_lambda))
        if self.dynamic_align_lambda_max < self.dynamic_align_lambda_min:
            self.dynamic_align_lambda_min, self.dynamic_align_lambda_max = (
                self.dynamic_align_lambda_max,
                self.dynamic_align_lambda_min,
            )
        if self.dynamic_align_lambda_enabled and not bool(getattr(args, "epoch_diagnostics_enabled", False)):
            raise ValueError("dynamic_align_lambda_enabled requires epoch_diagnostics_enabled=True")
        if self.axis_momentum_reset_enabled and not bool(getattr(args, "epoch_diagnostics_enabled", False)):
            raise ValueError("axis_momentum_reset_enabled requires epoch_diagnostics_enabled=True")
        if self.monitor_subset_enabled and (self.monitor_loader is None or self.monitor_every_optimizer_steps <= 0):
            raise ValueError(
                "monitor_subset_enabled requires monitor_subset_size > 0 and monitor_every_optimizer_steps > 0"
            )
        self._set_align_lambda(base_align_lambda)
        self._warm_start_from_checkpoint()
        if is_main_process() and self.dynamic_align_lambda_enabled:
            self.logger.print(
                "Dynamic align controller enabled: "
                f"lambda_range=[{self.dynamic_align_lambda_min:.4f}, {self.dynamic_align_lambda_max:.4f}], "
                f"delta_max={self.dynamic_align_lambda_delta_max:.4f}, "
                f"ema_beta={self.dynamic_align_progress_ema_beta:.4f}, "
                f"hold_threshold={self.dynamic_align_hold_progress_threshold:.4f}, "
                f"hold_lambda={self.dynamic_align_hold_lambda:.4f}"
            )
            if self.dynamic_align_digit8_target_enabled:
                self.logger.print(
                    "Dynamic digit8 target enabled: "
                    f"target={self.dynamic_align_digit8_target:.4f}, "
                    f"gain={self.dynamic_align_digit8_target_gain:.4f}, "
                    f"deadband={self.dynamic_align_digit8_target_deadband:.4f}"
                )
            if self.dynamic_align_digit8_band_enabled:
                self.logger.print(
                    "Dynamic digit8 band enabled: "
                    f"band=[{self.dynamic_align_digit8_band_low:.4f}, {self.dynamic_align_digit8_band_high:.4f}], "
                    f"gain_above={self.dynamic_align_digit8_band_gain_above:.4f}, "
                    f"gain_below={self.dynamic_align_digit8_band_gain_below:.4f}, "
                    f"deadband={self.dynamic_align_digit8_band_deadband:.4f}"
                )
        if is_main_process() and self.align_lr_compensation_enabled:
            self.logger.print(
                "Align LR compensation enabled: "
                f"start_epoch={self.align_lr_compensation_start_epoch:.2f}, "
                f"ref_lr={self.align_lr_compensation_ref_lr:.6g}, "
                f"scale_range=[{self.align_lr_compensation_min_scale:.4f}, {self.align_lr_compensation_max_scale:.4f}]"
            )
        if is_main_process() and self.axis_momentum_reset_enabled:
            self.logger.print(
                "Axis momentum reset enabled: "
                f"threshold={self.axis_momentum_reset_threshold:.4f}, "
                f"guard_band={self.axis_momentum_reset_guard_band:.4f}, "
                f"delta_tol={self.axis_momentum_reset_delta_tol:.4f}, "
                f"cooldown_epochs={self.axis_momentum_reset_cooldown_epochs}, "
                f"clear_exp_avg_sq={self.axis_momentum_reset_clear_exp_avg_sq}"
            )
        if is_main_process() and self.monitor_subset_enabled:
            self.logger.print(
                "Sub-epoch monitor enabled: "
                f"subset_size={len(self.monitor_dataset)}, "
                f"every_optimizer_steps={self.monitor_every_optimizer_steps}, "
                f"eval_batch_size={self.monitor_eval_batch_size}, "
                f"digit8_threshold={self.monitor_best_digit8_threshold:.4f}, "
                f"save_best_checkpoint={self.monitor_save_best_checkpoint}, "
                f"drives_axis_controller={self.monitor_drives_axis_controller}"
            )

    def _warm_start_from_checkpoint(self):
        """Load model state and, when requested, resume optimizer/train state."""
        checkpoint_path = str(getattr(self.args, "checkpoint_path", "")).strip()
        if not checkpoint_path:
            return
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if bool(getattr(self.args, "resume_training", False)) and isinstance(checkpoint, dict):
            optimizer_state = checkpoint.get("optimizer")
            if optimizer_state is not None:
                try:
                    self.optimizer.load_state_dict(optimizer_state)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.args.lr
                except Exception as exc:
                    if is_main_process():
                        self.logger.print(f"Resume optimizer state skipped: {type(exc).__name__}: {exc}")
            train_state = checkpoint.get("train_state") or {}
            loaded_epoch = int(checkpoint.get("epoch", -1))
            self.state.epoch = loaded_epoch + 1
            self.state.best_epoch = int(train_state.get("best_epoch", loaded_epoch))
            self.state.best_val_loss = float(train_state.get("best_val_loss", float("inf")))
            self.state.global_step = int(train_state.get("global_step", max(self.state.epoch, 0) * max(len(self.train_loader), 1)))
            if "base_align_lambda" in train_state:
                self._set_align_lambda(float(train_state["base_align_lambda"]))
            elif "align_lambda" in train_state:
                self._set_align_lambda(float(train_state["align_lambda"]))
            if "dynamic_align_progress_ema" in train_state and train_state["dynamic_align_progress_ema"] is not None:
                self._dynamic_align_progress_ema = float(train_state["dynamic_align_progress_ema"])
            if "axis_progress_prev" in train_state and train_state["axis_progress_prev"] is not None:
                self._axis_progress_previous = float(train_state["axis_progress_prev"])
            if "axis_momentum_reset_cooldown" in train_state:
                self._axis_momentum_reset_cooldown = max(0, int(train_state["axis_momentum_reset_cooldown"]))
            if "axis_momentum_reset_fire_count" in train_state:
                self._axis_momentum_reset_count = max(0, int(train_state["axis_momentum_reset_fire_count"]))
            if "monitor_best_optimizer_step" in train_state:
                self._monitor_best_optimizer_step = int(train_state["monitor_best_optimizer_step"])
            if "monitor_best_score" in train_state:
                self._monitor_best_score = float(train_state["monitor_best_score"])
            if "monitor_best_digit8" in train_state and train_state["monitor_best_digit8"] is not None:
                self._monitor_best_digit8 = float(train_state["monitor_best_digit8"])
            if "monitor_best_legal_integer" in train_state and train_state["monitor_best_legal_integer"] is not None:
                self._monitor_best_legal_integer = float(train_state["monitor_best_legal_integer"])
        if is_main_process():
            epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
            epoch_text = f", source epoch={epoch}" if epoch is not None else ""
            self.logger.print(f"Warm-started from checkpoint: {checkpoint_path}{epoch_text}")
            if missing:
                self.logger.print(f"Warm-start missing keys: {len(missing)}")
            if unexpected:
                self.logger.print(f"Warm-start unexpected keys: {len(unexpected)}")
            if bool(getattr(self.args, "resume_training", False)):
                self.logger.print(f"Resume training start epoch: {self.state.epoch}")

    def _train_state_payload(self):
        """Serialize core state plus optional-controller state for safe resume."""
        return {
            "epoch": int(self.state.epoch),
            "best_epoch": int(self.state.best_epoch),
            "best_val_loss": float(self.state.best_val_loss),
            "global_step": int(self.state.global_step),
            "align_lambda": float(getattr(self.model, "align_lambda", getattr(self.args, "align_lambda", 0.0))),
            "base_align_lambda": float(self._base_align_lambda),
            "dynamic_align_progress_ema": (
                float(self._dynamic_align_progress_ema) if self._dynamic_align_progress_ema is not None else None
            ),
            "axis_progress_prev": float(self._axis_progress_previous) if self._axis_progress_previous is not None else None,
            "axis_momentum_reset_cooldown": int(self._axis_momentum_reset_cooldown),
            "axis_momentum_reset_fire_count": int(self._axis_momentum_reset_count),
            "monitor_best_optimizer_step": int(self._monitor_best_optimizer_step),
            "monitor_best_score": float(self._monitor_best_score),
            "monitor_best_digit8": float(self._monitor_best_digit8) if self._monitor_best_digit8 is not None else None,
            "monitor_best_legal_integer": (
                float(self._monitor_best_legal_integer) if self._monitor_best_legal_integer is not None else None
            ),
        }

    def _set_runtime_align_lambda(self, value):
        """Apply the current alignment weight to every runtime model wrapper."""
        value = float(value)
        self.model.align_lambda = value
        loss_model = getattr(self.runtime_model, "module", self.runtime_model)
        if hasattr(loss_model, "align_lambda"):
            loss_model.align_lambda = value

    def _current_epoch_float(self):
        """Return the current epoch in the numeric form used by schedules."""
        return float(getattr(self.state, "epoch", 0))

    def _compute_align_lr_scale(self):
        """Return the opt-in alignment LR compensation factor."""
        if not self.align_lr_compensation_enabled:
            return 1.0
        if self._current_epoch_float() < self.align_lr_compensation_start_epoch:
            return 1.0
        current_lr = float(self.optimizer.param_groups[0]["lr"])
        if current_lr <= 0.0:
            return self.align_lr_compensation_max_scale
        scale = self.align_lr_compensation_ref_lr / current_lr
        return max(self.align_lr_compensation_min_scale, min(self.align_lr_compensation_max_scale, scale))

    def _refresh_align_lambda(self):
        """Apply LR compensation to the base alignment weight, if enabled."""
        actual_align_lambda = float(self._base_align_lambda) * self._compute_align_lr_scale()
        self._set_runtime_align_lambda(actual_align_lambda)
        return actual_align_lambda

    def _set_align_lambda(self, value):
        """Set the base alignment weight and propagate its runtime value."""
        self._base_align_lambda = float(value)
        return self._refresh_align_lambda()

    def _zero_optimizer_momentum(self):
        """Clear optimizer momentum for the experimental axis reset control."""
        optimizers = []
        for candidate in (
            self.optimizer,
            getattr(self.optimizer, "optimizer", None),
            getattr(self.optimizer, "optim", None),
            getattr(self.optimizer, "basic_optimizer", None),
        ):
            if candidate is not None and hasattr(candidate, "param_groups") and hasattr(candidate, "state"):
                optimizers.append(candidate)

        zeroed_params = 0
        seen_params = set()
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for param in group.get("params", []):
                    if param is None or id(param) in seen_params:
                        continue
                    seen_params.add(id(param))
                    state = optimizer.state.get(param)
                    if not state:
                        continue
                    exp_avg = state.get("exp_avg")
                    if torch.is_tensor(exp_avg):
                        exp_avg.zero_()
                        zeroed_params += 1
                    if self.axis_momentum_reset_clear_exp_avg_sq:
                        exp_avg_sq = state.get("exp_avg_sq")
                        if torch.is_tensor(exp_avg_sq):
                            exp_avg_sq.zero_()
        return zeroed_params

    def _compute_progress_hat(self, diagnostics):
        """Map validation diagnostics to the experimental progress estimate."""
        progress_hat = (
            self.dynamic_align_progress_intercept
            + self.dynamic_align_progress_coef_em * float(diagnostics["exact_match"])
            + self.dynamic_align_progress_coef_digit8 * float(diagnostics["digit8_rate"])
            + self.dynamic_align_progress_coef_single_digit * float(diagnostics["single_digit_rate"])
        )
        return max(self.dynamic_align_progress_clip_min, min(self.dynamic_align_progress_clip_max, progress_hat))

    def _run_monitor_diagnostics(self, optimizer_step: int):
        """Evaluate the opt-in fixed validation subset during an epoch."""
        if not self.monitor_subset_enabled or self.monitor_loader is None:
            return None
        prediction_path = self.checkpoints.epoch_prediction_path("monitor", optimizer_step)
        predict_to_file(self.model, self.monitor_loader, self.device, prediction_path)
        diagnostics = compute_prediction_diagnostics(
            self.args.dataset,
            prediction_path,
            dataset=self.dataset,
            args=self.args,
            digit_min_length=self.monitor_digit_min_length,
        )
        if not self.monitor_keep_predictions:
            with suppress(FileNotFoundError):
                prediction_path.unlink()
        return diagnostics

    def _maybe_save_monitor_checkpoint(self, optimizer_step: int, diagnostics):
        """Save a sub-epoch checkpoint only when its configured score improves."""
        if not self.monitor_save_best_checkpoint or diagnostics is None or self._checkpoint_failed:
            return
        legal_integer_rate = float(diagnostics.get("legal_integer_rate", 0.0))
        if self.best_checkpoint_metric == "val_em_legal_integer":
            val_em = float(diagnostics["exact_match"])
            better = (
                val_em > self._monitor_best_score
                or (
                    abs(val_em - self._monitor_best_score) <= 1e-12
                    and (
                        self._monitor_best_legal_integer is None
                        or legal_integer_rate > self._monitor_best_legal_integer
                    )
                )
            )
            if not better:
                return
            try:
                self.checkpoints.save(
                    self.model,
                    self.optimizer,
                    epoch=self.state.epoch,
                    is_best=False,
                    train_state=self._train_state_payload(),
                    suffix="monitor_best",
                )
                self._monitor_best_optimizer_step = int(optimizer_step)
                self._monitor_best_score = val_em
                self._monitor_best_legal_integer = legal_integer_rate
                self._monitor_best_digit8 = float(diagnostics["digit8_rate"])
                self.logger.print(
                    "Monitor checkpoint updated: "
                    f"optimizer_step={optimizer_step}, "
                    f"em={val_em:.4f}, "
                    f"legal_integer={legal_integer_rate:.4f}"
                )
            except Exception as exc:
                self._checkpoint_failed = True
                self.logger.print(f"Monitor checkpoint save failed at optimizer_step {optimizer_step}: {exc}")
            return
        digit8_rate = float(diagnostics["digit8_rate"])
        val_em = float(diagnostics["exact_match"])
        qualifies = digit8_rate <= self.monitor_best_digit8_threshold
        if not qualifies:
            return
        better = (
            val_em > self._monitor_best_score
            or (
                abs(val_em - self._monitor_best_score) <= 1e-12
                and (self._monitor_best_digit8 is None or digit8_rate < self._monitor_best_digit8)
            )
        )
        if not better:
            return
        try:
            self.checkpoints.save(
                self.model,
                self.optimizer,
                epoch=self.state.epoch,
                is_best=False,
                train_state=self._train_state_payload(),
                suffix="monitor_best",
            )
            self._monitor_best_optimizer_step = int(optimizer_step)
            self._monitor_best_score = val_em
            self._monitor_best_digit8 = digit8_rate
            self._monitor_best_legal_integer = legal_integer_rate
            self.logger.print(
                "Monitor checkpoint updated: "
                f"optimizer_step={optimizer_step}, "
                f"em={val_em:.4f}, "
                f"digit8={digit8_rate:.4f}"
            )
        except Exception as exc:
            self._checkpoint_failed = True
            self.logger.print(f"Monitor checkpoint save failed at optimizer_step {optimizer_step}: {exc}")

    def _run_subepoch_monitor(self, optimizer_step: int):
        """Run the opt-in fixed validation subset at optimizer-step intervals."""
        if not self.monitor_subset_enabled or self.monitor_loader is None:
            return None
        if optimizer_step <= 0 or optimizer_step % self.monitor_every_optimizer_steps != 0:
            return None

        diagnostics = None
        progress_hat = None
        barrier()
        if is_main_process():
            diagnostics = self._run_monitor_diagnostics(optimizer_step)
            progress_hat = self._compute_progress_hat(diagnostics) if diagnostics is not None else None
        if self.axis_momentum_reset_enabled and self.monitor_drives_axis_controller:
            self._maybe_reset_axis_momentum(diagnostics)
        if is_main_process() and diagnostics is not None:
            self.logger.print(
                "Sub-epoch monitor: "
                f"optimizer_step={optimizer_step}, "
                f"EM={diagnostics['exact_match']:.4f}, "
                f"digit8={diagnostics['digit8_rate']:.4f}, "
                f"legal_integer={diagnostics.get('legal_integer_rate', 0.0):.4f}, "
                f"single={diagnostics['single_digit_rate']:.4f}"
            )
            log_payload = {
                "Monitor EM": diagnostics["exact_match"],
                "Monitor Digit8 Rate": diagnostics["digit8_rate"],
                "Monitor Legal Integer Rate": diagnostics.get("legal_integer_rate", 0.0),
                "Monitor Single-Digit Rate": diagnostics["single_digit_rate"],
                "Monitor Optimizer Step": float(optimizer_step),
            }
            if progress_hat is not None:
                log_payload["Monitor Progress Hat"] = progress_hat
            self.logger.log(log_payload)
            self._maybe_save_monitor_checkpoint(optimizer_step, diagnostics)
        barrier()
        self.runtime_model.train()
        return diagnostics

    def _maybe_reset_axis_momentum(self, diagnostics):
        """Apply the experimental optimizer-momentum reset controller."""
        if not self.axis_momentum_reset_enabled:
            return

        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        trigger_flag = 0.0
        progress_value = -1.0
        prev_value = -1.0 if self._axis_progress_previous is None else float(self._axis_progress_previous)
        delta_value = 0.0
        cooldown_value = float(self._axis_momentum_reset_cooldown)

        if is_main_process():
            if diagnostics is None:
                self.logger.print("Axis momentum reset skipped: diagnostics unavailable.")
            else:
                progress = self._compute_progress_hat(diagnostics)
                progress_value = float(progress)
                previous = self._axis_progress_previous
                if self._axis_momentum_reset_cooldown > 0:
                    self._axis_momentum_reset_cooldown -= 1
                if previous is not None:
                    prev_value = float(previous)
                    delta_value = progress_value - prev_value
                    crossed_threshold = (
                        prev_value >= self.axis_momentum_reset_threshold
                        and progress_value < self.axis_momentum_reset_threshold
                    )
                    entered_guard_band = (
                        prev_value >= self.axis_momentum_reset_threshold
                        and progress_value
                        < (self.axis_momentum_reset_threshold + self.axis_momentum_reset_guard_band)
                    )
                    falling = delta_value <= -self.axis_momentum_reset_delta_tol
                    if self._axis_momentum_reset_cooldown <= 0 and falling and (crossed_threshold or entered_guard_band):
                        trigger_flag = 1.0
                        self._axis_momentum_reset_cooldown = self.axis_momentum_reset_cooldown_epochs
                self._axis_progress_previous = progress_value
                cooldown_value = float(self._axis_momentum_reset_cooldown)

        sync_tensor = torch.tensor(
            [trigger_flag, progress_value, prev_value, delta_value, cooldown_value],
            device=device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized() and world_size() > 1:
            dist.broadcast(sync_tensor, src=0)

        trigger_flag, progress_value, prev_value, delta_value, cooldown_value = [float(x) for x in sync_tensor.tolist()]
        self._axis_progress_previous = None if progress_value < 0.0 else progress_value
        self._axis_momentum_reset_cooldown = max(0, int(round(cooldown_value)))

        if trigger_flag >= 0.5:
            zeroed_params = self._zero_optimizer_momentum()
            self._axis_momentum_reset_count += 1
            if is_main_process():
                self.logger.print(
                    "Axis momentum reset fired: "
                    f"prev_progress={prev_value:.4f}, "
                    f"progress={progress_value:.4f}, "
                    f"delta={delta_value:.4f}, "
                    f"threshold={self.axis_momentum_reset_threshold:.4f}, "
                    f"zeroed_params={zeroed_params}"
                )
                self.logger.log(
                    {
                        "Axis Progress": progress_value,
                        "Axis Progress Prev": prev_value,
                        "Axis Progress Delta": delta_value,
                        "Axis Momentum Reset Fired": 1.0,
                        "Axis Momentum Reset Cooldown": self._axis_momentum_reset_cooldown,
                        "Axis Momentum Reset Count": self._axis_momentum_reset_count,
                    }
                )
        elif is_main_process() and progress_value >= 0.0:
            self.logger.print(
                "Axis momentum monitor: "
                f"prev_progress={prev_value:.4f}, "
                f"progress={progress_value:.4f}, "
                f"delta={delta_value:.4f}, "
                f"cooldown={self._axis_momentum_reset_cooldown}"
            )
            self.logger.log(
                {
                    "Axis Progress": progress_value,
                    "Axis Progress Prev": prev_value,
                    "Axis Progress Delta": delta_value,
                    "Axis Momentum Reset Fired": 0.0,
                    "Axis Momentum Reset Cooldown": self._axis_momentum_reset_cooldown,
                    "Axis Momentum Reset Count": self._axis_momentum_reset_count,
                }
            )

    def _update_dynamic_align_lambda(self, diagnostics):
        """Update alignment weight for the experimental dynamic controller."""
        if not self.dynamic_align_lambda_enabled:
            return

        device = self.device if torch.cuda.is_available() else torch.device("cpu")
        current_base_lambda = float(self._base_align_lambda)
        next_base_lambda = current_base_lambda
        progress_hat = None
        progress_ema = self._dynamic_align_progress_ema
        target_lambda = current_base_lambda
        digit8_error = None
        digit8_band_error = None
        digit8_band_adjustment = 0.0

        if is_main_process():
            if diagnostics is None:
                self.logger.print('Dynamic align controller skipped: diagnostics unavailable.')
            else:
                progress_hat = self._compute_progress_hat(diagnostics)
                if progress_ema is None:
                    progress_ema = progress_hat
                else:
                    progress_ema = (
                        self.dynamic_align_progress_ema_beta * progress_ema
                        + (1.0 - self.dynamic_align_progress_ema_beta) * progress_hat
                    )
                target_lambda = self.dynamic_align_lambda_min + (
                    self.dynamic_align_lambda_max - self.dynamic_align_lambda_min
                ) * (1.0 - progress_ema)
                if progress_ema >= self.dynamic_align_hold_progress_threshold:
                    target_lambda = max(target_lambda, self.dynamic_align_hold_lambda)
                if self.dynamic_align_digit8_target_enabled:
                    digit8_error = float(diagnostics["digit8_rate"]) - self.dynamic_align_digit8_target
                    if abs(digit8_error) <= self.dynamic_align_digit8_target_deadband:
                        digit8_error = 0.0
                    target_lambda += self.dynamic_align_digit8_target_gain * digit8_error
                if self.dynamic_align_digit8_band_enabled:
                    digit8_rate = float(diagnostics["digit8_rate"])
                    upper = self.dynamic_align_digit8_band_high + self.dynamic_align_digit8_band_deadband
                    lower = self.dynamic_align_digit8_band_low - self.dynamic_align_digit8_band_deadband
                    if digit8_rate > upper:
                        digit8_band_error = digit8_rate - upper
                        digit8_band_adjustment = self.dynamic_align_digit8_band_gain_above * digit8_band_error
                    elif digit8_rate < lower:
                        digit8_band_error = digit8_rate - lower
                        digit8_band_adjustment = self.dynamic_align_digit8_band_gain_below * digit8_band_error
                    else:
                        digit8_band_error = 0.0
                        digit8_band_adjustment = 0.0
                    target_lambda += digit8_band_adjustment
                delta = max(-self.dynamic_align_lambda_delta_max, min(self.dynamic_align_lambda_delta_max, target_lambda - current_base_lambda))
                next_base_lambda = current_base_lambda + delta
                next_base_lambda = max(self.dynamic_align_lambda_min, min(self.dynamic_align_lambda_max, next_base_lambda))

        lambda_tensor = torch.tensor(next_base_lambda, device=device, dtype=torch.float32)
        ema_tensor = torch.tensor(
            -1.0 if progress_ema is None else progress_ema,
            device=device,
            dtype=torch.float32,
        )
        if dist.is_available() and dist.is_initialized() and world_size() > 1:
            dist.broadcast(lambda_tensor, src=0)
            dist.broadcast(ema_tensor, src=0)

        actual_align_lambda = self._set_align_lambda(float(lambda_tensor.item()))
        self._dynamic_align_progress_ema = None if ema_tensor.item() < 0.0 else float(ema_tensor.item())

        if is_main_process() and progress_hat is not None:
            self.logger.print(
                "Dynamic align update: "
                f"progress_hat={progress_hat:.4f}, "
                f"progress_ema={self._dynamic_align_progress_ema:.4f}, "
                f"target_lambda={target_lambda:.4f}, "
                f"base_align_lambda={float(lambda_tensor.item()):.4f}, "
                f"align_lambda={actual_align_lambda:.4f}"
            )
            if digit8_error is not None:
                self.logger.print(
                    "Dynamic digit8 target: "
                    f"target={self.dynamic_align_digit8_target:.4f}, "
                    f"error={digit8_error:.4f}"
                )
            if self.dynamic_align_digit8_band_enabled:
                self.logger.print(
                    "Dynamic digit8 band: "
                    f"band=[{self.dynamic_align_digit8_band_low:.4f}, {self.dynamic_align_digit8_band_high:.4f}], "
                    f"error={0.0 if digit8_band_error is None else digit8_band_error:.4f}, "
                    f"adjustment={digit8_band_adjustment:.4f}"
                )
            self.logger.log(
                {
                    "Dynamic Align Progress Hat": progress_hat,
                    "Dynamic Align Progress EMA": self._dynamic_align_progress_ema,
                    "Dynamic Align Target Lambda": target_lambda,
                    "Dynamic Align Base Lambda": float(lambda_tensor.item()),
                    "Dynamic Align Lambda": actual_align_lambda,
                }
            )
            if digit8_error is not None:
                self.logger.log(
                    {
                        "Dynamic Align Digit8 Target": self.dynamic_align_digit8_target,
                        "Dynamic Align Digit8 Error": digit8_error,
                    }
                )
            if self.dynamic_align_digit8_band_enabled:
                self.logger.log(
                    {
                        "Dynamic Align Digit8 Band Low": self.dynamic_align_digit8_band_low,
                        "Dynamic Align Digit8 Band High": self.dynamic_align_digit8_band_high,
                        "Dynamic Align Digit8 Band Error": 0.0 if digit8_band_error is None else digit8_band_error,
                        "Dynamic Align Digit8 Band Adjustment": digit8_band_adjustment,
                    }
                )

    def _should_update_best_checkpoint(self, epoch, val_loss, diagnostics=None):
        """Apply the configured, deterministic best-checkpoint policy."""
        if self.best_checkpoint_metric == "val_loss":
            return val_loss < self._best_checkpoint_score, f"val_loss={val_loss:.4f}"

        if diagnostics is None:
            return False, "diagnostics unavailable"

        if self.best_checkpoint_metric == "val_em":
            val_em = float(diagnostics["exact_match"])
            better = val_em > self._best_checkpoint_score
            return better, f"val_em={val_em:.4f}"

        if self.best_checkpoint_metric == "val_em_legal_integer":
            val_em = float(diagnostics["exact_match"])
            legal_integer_rate = float(diagnostics.get("legal_integer_rate", 0.0))
            better = (
                val_em > self._best_checkpoint_score
                or (
                    abs(val_em - self._best_checkpoint_score) <= 1e-12
                    and (
                        self._best_checkpoint_legal_integer is None
                        or legal_integer_rate > self._best_checkpoint_legal_integer
                    )
                )
            )
            return better, f"val_em={val_em:.4f}, legal_integer={legal_integer_rate:.4f}"

        val_em = float(diagnostics["exact_match"])
        digit8_rate = float(diagnostics["digit8_rate"])
        qualifies = digit8_rate <= self.best_checkpoint_digit8_threshold
        if qualifies:
            better = (
                val_em > self._best_checkpoint_score
                or (
                    abs(val_em - self._best_checkpoint_score) <= 1e-12
                    and (self._best_checkpoint_digit8 is None or digit8_rate < self._best_checkpoint_digit8)
                )
            )
            return better, f"val_em={val_em:.4f}, digit8={digit8_rate:.4f}"

        if self.best_checkpoint_fallback_to_val_loss and self._best_checkpoint_epoch < 0:
            return val_loss < float("inf"), f"fallback_val_loss={val_loss:.4f}"
        return False, f"digit8={digit8_rate:.4f} exceeds threshold={self.best_checkpoint_digit8_threshold:.4f}"

    def _mark_best_checkpoint(self, epoch, val_loss, diagnostics=None):
        """Update the in-memory best-checkpoint statistics after saving."""
        self._best_checkpoint_epoch = int(epoch)
        if self.best_checkpoint_metric == "val_loss" or diagnostics is None:
            self._best_checkpoint_score = float(val_loss)
            self._best_checkpoint_digit8 = float(diagnostics["digit8_rate"]) if diagnostics is not None else None
            self._best_checkpoint_legal_integer = (
                float(diagnostics.get("legal_integer_rate", 0.0)) if diagnostics is not None else None
            )
            return
        if self.best_checkpoint_metric == "val_em":
            self._best_checkpoint_score = float(diagnostics["exact_match"])
            self._best_checkpoint_digit8 = float(diagnostics["digit8_rate"])
            self._best_checkpoint_legal_integer = float(diagnostics.get("legal_integer_rate", 0.0))
            return
        if self.best_checkpoint_metric == "val_em_legal_integer":
            self._best_checkpoint_score = float(diagnostics["exact_match"])
            self._best_checkpoint_legal_integer = float(diagnostics.get("legal_integer_rate", 0.0))
            self._best_checkpoint_digit8 = float(diagnostics["digit8_rate"])
            return
        digit8_rate = float(diagnostics["digit8_rate"])
        val_em = float(diagnostics["exact_match"])
        qualifies = digit8_rate <= self.best_checkpoint_digit8_threshold
        if qualifies:
            self._best_checkpoint_score = val_em
            self._best_checkpoint_digit8 = digit8_rate
            self._best_checkpoint_legal_integer = float(diagnostics.get("legal_integer_rate", 0.0))
        else:
            self._best_checkpoint_score = float(val_loss)
            self._best_checkpoint_digit8 = digit8_rate
            self._best_checkpoint_legal_integer = float(diagnostics.get("legal_integer_rate", 0.0))

    def train_one_epoch(self, epoch):
        """Train for one epoch and return the mean loss."""
        loss_model = getattr(self.runtime_model, "module", self.runtime_model)
        if hasattr(loss_model, "update_distance_branch_freeze_state"):
            loss_model.update_distance_branch_freeze_state(epoch)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        self.runtime_model.train()
        epoch_loss, accum_loss, accumulated_batches = 0.0, 0.0, 0
        progress_bar = tqdm(range(len(self.train_loader)), disable=not is_main_process())
        if not is_deepspeed_enabled(self.args):
            self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader):
            batch = move_batch_to_device(batch, self.device)
            optimizer_step = self.state.global_step // max(self.args.grad_steps, 1)
            optimizer_step_completed = False
            actual_align_lambda = self._refresh_align_lambda()
            if hasattr(loss_model, "update_projector_freeze_state"):
                loss_model.update_projector_freeze_state(optimizer_step)
            if is_deepspeed_enabled(self.args):
                loss = self.runtime_model(batch)
                self.runtime_model.backward(loss)
                boundary_fn = getattr(self.runtime_model, "is_gradient_accumulation_boundary", None)
                if callable(boundary_fn):
                    optimizer_step_completed = bool(boundary_fn())
                else:
                    optimizer_step_completed = (step + 1) % max(self.args.grad_steps, 1) == 0
                if optimizer_step_completed:
                    adjust_learning_rate(self.optimizer.param_groups[0], self.args.lr, step / max(len(self.train_loader), 1) + epoch, self.args)
                self.runtime_model.step()
            else:
                loss = self.runtime_model(batch)
                window_size, optimizer_step_completed = _accumulation_window(
                    step,
                    len(self.train_loader),
                    self.args.grad_steps,
                )
                (loss / max(window_size, 1)).backward()
                if optimizer_step_completed:
                    params_to_clip = [
                        param
                        for group in self.optimizer.param_groups
                        for param in group['params']
                        if param.grad is not None
                    ]
                    if params_to_clip:
                        torch.nn.utils.clip_grad_norm_(params_to_clip, 0.1)
                    adjust_learning_rate(self.optimizer.param_groups[0], self.args.lr, step / max(len(self.train_loader), 1) + epoch, self.args)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            loss_item = float(loss.detach().item())
            epoch_loss += loss_item
            accum_loss += loss_item
            accumulated_batches += 1
            self.state.global_step += 1
            if is_main_process() and optimizer_step_completed:
                lr = self.optimizer.param_groups[0]['lr']
                metrics = {
                    'Lr': lr,
                    'Accum Loss': accum_loss / max(accumulated_batches, 1),
                    'Align Lambda': actual_align_lambda,
                    'Base Align Lambda': float(self._base_align_lambda),
                    'Align LR Compensation Scale': self._compute_align_lr_scale(),
                }
                aux_metrics = {}
                if hasattr(loss_model, "latest_aux_metrics"):
                    aux_metrics = loss_model.latest_aux_metrics()
                metrics.update(aux_metrics)
                if "lm_loss" in aux_metrics:
                    self.logger.print(f"LM loss: {aux_metrics['lm_loss']:.4f}")
                if "align_loss" in aux_metrics:
                    self.logger.print(f"Align loss: {aux_metrics['align_loss']:.4f}")
                if "align_grad_norm" in aux_metrics:
                    self.logger.print(f"Align grad norm: {aux_metrics['align_grad_norm']:.6f}")
                if "distance_residual_alpha" in aux_metrics:
                    self.logger.print(f"Distance residual alpha: {aux_metrics['distance_residual_alpha']:.6f}")
                if "distance_branch_frozen" in aux_metrics:
                    self.logger.print(f"Distance branch frozen: {bool(aux_metrics['distance_branch_frozen'])}")
                self.logger.log(metrics)
                accum_loss = 0.0
                accumulated_batches = 0
            if optimizer_step_completed:
                current_optimizer_step = self.state.global_step // max(self.args.grad_steps, 1)
                self._run_subepoch_monitor(current_optimizer_step)
            if is_main_process():
                progress_bar.update(1)
        return epoch_loss / max(len(self.train_loader), 1)

    def validate(self):
        """Evaluate mean loss on the validation split."""
        return evaluate_loss(self.runtime_model, self.val_loader, self.device)

    def test(self):
        """Generate test predictions and return dataset-specific metrics."""
        prediction_path = self.checkpoints.prediction_path()
        predict_to_file(self.model, self.test_loader, self.device, prediction_path)
        return evaluate_dataset(self.args.dataset, prediction_path, dataset=self.dataset, args=self.args)

    def run_epoch_diagnostics(self, epoch):
        """Generate temporary validation predictions for output-state metrics."""
        prediction_path = self.checkpoints.epoch_prediction_path("val", epoch)
        predict_to_file(self.model, self.val_loader, self.device, prediction_path)
        diagnostics = compute_prediction_diagnostics(
            self.args.dataset,
            prediction_path,
            dataset=self.dataset,
            args=self.args,
            digit_min_length=getattr(self.args, "epoch_diagnostics_digit_min_length", 6),
        )
        if not getattr(self.args, "epoch_diagnostics_keep_predictions", False):
            with suppress(FileNotFoundError):
                prediction_path.unlink()
        return diagnostics

    def fit(self):
        """Run train/validation/early-stop/test orchestration."""
        self.logger.init()
        if is_main_process():
            trainable_params, all_param = self.model.print_trainable_params()
            self.logger.print(f'trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}')
        start_epoch = int(getattr(self.state, "epoch", 0))
        for epoch in range(start_epoch, self.args.num_epochs):
            self.state.epoch = epoch
            train_loss = self.train_one_epoch(epoch)
            barrier()
            stop_training = False
            diagnostics = None
            if is_main_process():
                self.logger.print(f'Epoch: {epoch}|{self.args.num_epochs}: Train Loss (Epoch Mean): {train_loss}')
                self.logger.log({'Train Loss (Epoch Mean)': train_loss})
                val_loss = self.validate()
                self.logger.print(f'Epoch: {epoch}|{self.args.num_epochs}: Val Loss: {val_loss}')
                self.logger.log({'Val Loss': val_loss})
                if getattr(self.args, "epoch_diagnostics_enabled", False):
                    diagnostics = self.run_epoch_diagnostics(epoch)
                    self.logger.print(
                        f"Epoch: {epoch}|{self.args.num_epochs}: "
                        f"Val EM: {diagnostics['exact_match']:.4f} | "
                        f"Val Digit8 Rate: {diagnostics['digit8_rate']:.4f} | "
                        f"Val Legal Integer Rate: {diagnostics.get('legal_integer_rate', 0.0):.4f} | "
                        f"Val Single-Digit Rate: {diagnostics['single_digit_rate']:.4f}"
                    )
                    self.logger.log(
                        {
                            'Val EM': diagnostics['exact_match'],
                            'Val Digit8 Rate': diagnostics['digit8_rate'],
                            'Val Legal Integer Rate': diagnostics.get('legal_integer_rate', 0.0),
                            'Val Single-Digit Rate': diagnostics['single_digit_rate'],
                        }
                    )
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    self.state.best_epoch = epoch
            if self.dynamic_align_lambda_enabled:
                self._update_dynamic_align_lambda(diagnostics)
            if self.axis_momentum_reset_enabled:
                self._maybe_reset_axis_momentum(diagnostics)
            if is_main_process():
                if getattr(self.args, "save_every_epoch", False):
                    self.checkpoints.save(self.model, self.optimizer, epoch, is_best=False, train_state=self._train_state_payload())
                save_best, reason = self._should_update_best_checkpoint(epoch, val_loss, diagnostics=diagnostics)
                if save_best and not self._checkpoint_failed:
                    try:
                        self.checkpoints.save(self.model, self.optimizer, epoch, is_best=True, train_state=self._train_state_payload())
                        self._mark_best_checkpoint(epoch, val_loss, diagnostics=diagnostics)
                        self.logger.print(f'Best checkpoint updated at epoch {epoch} ({reason})')
                    except Exception as exc:
                        self._checkpoint_failed = True
                        self.logger.print(f'Checkpoint save failed at epoch {epoch}: {exc}')
                if self.early_stop.should_stop(epoch, self.state.best_epoch):
                    self.logger.print(f'Early stop at epoch {epoch}')
                    stop_training = True
            if dist.is_available() and dist.is_initialized() and world_size() > 1:
                stop_tensor = torch.tensor(
                    int(stop_training),
                    device=self.device if torch.cuda.is_available() else torch.device("cpu"),
                )
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
            barrier()
            if stop_training:
                break
        if getattr(self.args, "collect_hsgs_node_scores", False) and hasattr(self.model, "export_hsgs_node_scores_from_tensors"):
            score_sum, score_count = self.model.get_hsgs_node_score_tensors()
            if score_sum is not None and score_count is not None and dist.is_available() and dist.is_initialized() and world_size() > 1:
                reduce_device = self.device if torch.cuda.is_available() else torch.device("cpu")
                score_sum_dev = score_sum.to(reduce_device)
                score_count_dev = score_count.to(reduce_device)
                dist.all_reduce(score_sum_dev, op=dist.ReduceOp.SUM)
                dist.all_reduce(score_count_dev, op=dist.ReduceOp.SUM)
                score_sum = score_sum_dev.cpu()
                score_count = score_count_dev.cpu()
            if is_main_process() and score_sum is not None and score_count is not None:
                stats_path = self.checkpoints.node_score_path()
                self.model.export_hsgs_node_scores_from_tensors(stats_path, score_sum, score_count)
                self.logger.print(f'HSGS node scores saved to {stats_path}')
            elif is_main_process():
                self.logger.print('HSGS node score collection was enabled, but no node scores were produced.')
        barrier()
        if is_main_process():
            if getattr(self.args, "skip_test", False):
                self.logger.print('Skipping final test evaluation (--skip_test).')
            else:
                checkpoint_path = self.checkpoints.checkpoint_path('best')
                if not self._checkpoint_failed and checkpoint_path.exists():
                    self.model = self.checkpoints.load_best(self.model)
                else:
                    self.logger.print('Best checkpoint unavailable; evaluating current in-memory model.')
                acc = self.test()
                self.logger.print(f'Test Acc {acc}')
                self.logger.log({'Test Acc': acc})
        barrier()
        if is_main_process():
            self.logger.finish()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        gc.collect()
        shutdown_distributed()

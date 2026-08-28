"""Rank-zero console and optional Weights & Biases logging."""

import os

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover - exercised by import-only environments
    wandb = None

from s2ge.infra.distributed import is_main_process


class RankZeroLogger:
    """Prevent duplicate logs in distributed runs and tolerate missing W&B."""
    def __init__(self, args):
        self.args = args
        self.enabled = is_main_process() and not getattr(args, 'no_wandb', False)
        self.wandb_run = None

    def init(self):
        """Start an offline-by-default W&B run on the main process."""
        if not self.enabled:
            return
        if wandb is None:
            self.enabled = False
            self.print("wandb is not installed; disabling W&B logging for this run.")
            return
        self.wandb_run = wandb.init(
            project=self.args.project,
            name=self.args.run_name,
            config=vars(self.args),
            mode=os.environ.get('WANDB_MODE', 'offline'),
        )

    def log(self, payload):
        """Log a scalar payload when W&B is enabled."""
        if self.enabled and wandb is not None:
            wandb.log(payload)

    def print(self, *args, **kwargs):
        """Print only from rank zero."""
        if is_main_process():
            print(*args, **kwargs)

    def finish(self):
        """Finish the active W&B run, if any."""
        if self.enabled and self.wandb_run is not None and wandb is not None:
            wandb.finish()

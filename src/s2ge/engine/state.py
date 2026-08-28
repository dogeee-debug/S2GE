"""Serializable training and evaluation state records."""

from dataclasses import dataclass


@dataclass
class TrainState:
    """Mutable counters and best-validation state owned by ``Trainer``."""
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    best_epoch: int = -1


@dataclass
class EvalState:
    """Scalar results produced by one evaluation pass."""
    loss: float = 0.0
    metric: float = 0.0

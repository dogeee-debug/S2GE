from dataclasses import dataclass


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_loss: float = float('inf')
    best_epoch: int = -1


@dataclass
class EvalState:
    loss: float = 0.0
    metric: float = 0.0

"""Small state-free training hooks."""


class EarlyStopHook:
    """Stop after ``patience`` epochs without a new best checkpoint."""
    def __init__(self, patience: int):
        self.patience = patience

    def should_stop(self, epoch: int, best_epoch: int) -> bool:
        """Return whether the zero-based epoch distance reaches patience."""
        if best_epoch < 0:
            return False
        return epoch - best_epoch >= self.patience

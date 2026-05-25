class EarlyStopHook:
    def __init__(self, patience: int):
        self.patience = patience

    def should_stop(self, epoch: int, best_epoch: int) -> bool:
        if best_epoch < 0:
            return False
        return epoch - best_epoch >= self.patience

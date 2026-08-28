"""Regression tests for the evaluation CLI checkpoint contract."""

from types import SimpleNamespace

from s2ge.cli.eval import _load_default_checkpoint


class _CheckpointStub:
    def __init__(self):
        self.loaded = []

    def load_best(self, model):
        self.loaded.append(model)
        return "best-model"


def test_eval_loads_best_checkpoint_into_canonical_model_field():
    trainer = SimpleNamespace(model="initial-model", checkpoints=_CheckpointStub())

    _load_default_checkpoint(trainer, checkpoint_path="")

    assert trainer.checkpoints.loaded == ["initial-model"]
    assert trainer.model == "best-model"


def test_eval_preserves_explicit_checkpoint_loaded_by_trainer():
    trainer = SimpleNamespace(model="explicit-model", checkpoints=_CheckpointStub())

    _load_default_checkpoint(trainer, checkpoint_path="checkpoint.pth")

    assert trainer.checkpoints.loaded == []
    assert trainer.model == "explicit-model"

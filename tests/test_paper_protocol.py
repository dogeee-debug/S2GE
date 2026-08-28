"""Guards connecting camera-ready claims to released training configs."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from s2ge.cli.common import parse_args
from s2ge.engine.deepspeed_engine import DeepSpeedRuntime
from s2ge.engine.trainer import _accumulation_window


ROOT = Path(__file__).resolve().parents[1]
MAIN_CONFIGS = (
    "dblp_main.yaml",
    "biomedical_main.yaml",
    "goodreads_main.yaml",
    "pubmed_main.yaml",
)


@pytest.mark.parametrize("config_name", MAIN_CONFIGS)
def test_main_configs_match_camera_ready_protocol(config_name):
    args = parse_args(["--config", str(ROOT / "configs" / "train" / config_name)])

    assert args.dataset == "grbench"
    assert args.llm_model_name == "llama3_8b_instruct"
    assert args.num_epochs == 12
    assert args.patience == 6
    assert args.bf16 is True
    assert args.grad_checkpointing is False
    assert args.graph_token_mode == "nodes"
    assert args.hsgs_enabled is True
    assert args.graph_query_syntax_enabled is True
    assert args.graph_role_post_projector_enabled is True
    assert args.graph_distance_post_projector_enabled is True
    assert args.align_lambda == pytest.approx(0.25)
    assert args.align_on_projected_tokens is True

    # Experimental mechanisms are not part of the camera-ready main method.
    assert args.moco_enabled is False
    assert args.dual_view_enabled is False
    assert args.graph_token_shuffle_enabled is False
    assert args.graph_token_corrupt_enabled is False
    assert args.graph_tokens_disabled is False
    assert args.dynamic_align_lambda_enabled is False
    assert args.align_lr_compensation_enabled is False
    assert args.axis_momentum_reset_enabled is False
    assert args.monitor_subset_enabled is False


def test_local_runtime_override_is_available_for_paper_configs():
    config = ROOT / "configs" / "train" / "dblp_main.yaml"

    args = parse_args(["--config", str(config), "--no-use-deepspeed"])

    assert args.use_deepspeed is False


def test_gradient_accumulation_flushes_a_partial_final_window():
    windows = [_accumulation_window(step, num_batches=5, grad_steps=2) for step in range(5)]

    assert windows == [(2, False), (2, True), (2, False), (2, True), (1, True)]


def test_default_deepspeed_config_preserves_protocol_batching():
    args = SimpleNamespace(deepspeed="", batch_size=1, grad_steps=2, bf16=False)

    config = DeepSpeedRuntime(args)._resolve_config()

    assert config["train_micro_batch_size_per_gpu"] == 1
    assert config["gradient_accumulation_steps"] == 2
    assert config["gradient_clipping"] == pytest.approx(0.1)
    assert config["zero_optimization"] == {"stage": 0}


def test_default_deepspeed_config_enables_camera_ready_bf16():
    args = SimpleNamespace(deepspeed="", batch_size=1, grad_steps=2, bf16=True)

    config = DeepSpeedRuntime(args)._resolve_config()

    assert config["bf16"] == {"enabled": True}

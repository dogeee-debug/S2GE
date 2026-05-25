import argparse
import os
from pathlib import Path

import yaml

from s2ge.infra.paths import datasets_processed_root, repo_root

DEFAULT_LLM_MODEL_PATHS = {
    '7b': 'meta-llama/Llama-2-7b-hf',
    '7b_chat': 'meta-llama/Llama-2-7b-chat-hf',
    '13b': 'meta-llama/Llama-2-13b-hf',
    '13b_chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama3_8b_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'qwen2_5_1_5b_instruct': 'Qwen/Qwen2.5-1.5B-Instruct',
}

_DISABLED_PROJECTOR_CKPT_VALUES = {"", "none", "null", "false", "off", "disable", "disabled"}


def csv_list(string):
    if isinstance(string, list):
        return string
    return [item for item in str(string).split(',') if item]


def parse_bool(value):
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_config_defaults(config_path: str):
    if not config_path:
        return {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data


def build_parser(defaults=None):
    defaults = defaults or {}
    parser = argparse.ArgumentParser(description="S2GE 2.0 trainer")
    parser.add_argument('--config', type=str, default='')
    parser.add_argument('--model_name', '--model-name', dest='model_name', type=str, default='graph_llm')
    parser.add_argument('--project', type=str, default='project_g_retriever')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='grbench')
    parser.add_argument('--data-root', dest='data_root', type=str, default='')
    parser.add_argument('--model-root', dest='model_root', type=str, default='')
    parser.add_argument('--output-root', dest='output_root', type=str, default='')
    parser.add_argument('--offline', action='store_true', default=defaults.get('offline', True))
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--batch_size', '--batch-size', dest='batch_size', type=int, default=8)
    parser.add_argument('--grad_steps', '--grad-steps', dest='grad_steps', type=int, default=2)
    parser.add_argument('--num_epochs', '--num-epochs', dest='num_epochs', type=int, default=10)
    parser.add_argument('--warmup_epochs', '--warmup-epochs', dest='warmup_epochs', type=float, default=1)
    parser.add_argument('--eval_batch_size', '--eval-batch-size', dest='eval_batch_size', type=int, default=16)
    parser.add_argument('--train_num_workers', '--train-num-workers', dest='train_num_workers', type=int, default=0)
    parser.add_argument('--eval_num_workers', '--eval-num-workers', dest='eval_num_workers', type=int, default=0)
    parser.add_argument('--persistent_workers', '--persistent-workers', dest='persistent_workers', type=parse_bool, default=False)
    parser.add_argument('--llm_model_name', '--llm-model-name', dest='llm_model_name', type=str, default='llama3_8b_instruct')
    parser.add_argument('--llm_model_path', '--llm-model-path', dest='llm_model_path', type=str, default='')
    parser.add_argument('--llm_frozen', type=str, default='True')
    parser.add_argument('--llm_num_virtual_tokens', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--max_txt_len', type=int, default=512)
    parser.add_argument('--max_new_tokens', type=int, default=32)
    parser.add_argument('--max_memory', type=csv_list, default=[80, 80])
    parser.add_argument('--llm_load_4bit', action='store_true')
    parser.add_argument('--grad_checkpointing', action='store_true')
    parser.add_argument('--gradient_checkpointing', dest='grad_checkpointing', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--micro_batch_size', type=int, default=0)
    parser.add_argument('--use_deepspeed', action='store_true')
    parser.add_argument('--deepspeed', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--use_fsdp', action='store_true')
    parser.add_argument('--gnn_model_name', type=str, default='gt')
    parser.add_argument('--gnn_num_layers', type=int, default=4)
    parser.add_argument('--gnn_in_dim', type=int, default=1024)
    parser.add_argument('--gnn_hidden_dim', type=int, default=1024)
    parser.add_argument('--gnn_num_heads', type=int, default=4)
    parser.add_argument('--gnn_dropout', type=float, default=0.0)
    parser.add_argument('--hsgs_enabled', action='store_true')
    parser.add_argument('--hsgs_k_init', type=int, default=25)
    parser.add_argument('--hsgs_gamma', type=float, default=0.7)
    parser.add_argument('--hsgs_seed_budget', type=int, default=8)
    parser.add_argument('--hsgs_num_hops', type=int, default=0)
    parser.add_argument('--graph_token_mode', type=str, default='single', choices=['single', 'nodes'])
    parser.add_argument('--graph_tokens_disabled', type=parse_bool, default=False)
    parser.add_argument('--graph_token_budget', type=int, default=32)
    parser.add_argument('--graph_bfs_order_enabled', type=parse_bool, default=False)
    parser.add_argument('--graph_query_syntax_enabled', type=parse_bool, default=True)
    parser.add_argument('--graph_token_shuffle_enabled', type=parse_bool, default=False)
    parser.add_argument('--graph_token_corrupt_enabled', type=parse_bool, default=False)
    parser.add_argument('--graph_token_corrupt_std', type=float, default=1.0)
    parser.add_argument('--legal_integer_decode_enabled', type=parse_bool, default=False)
    parser.add_argument('--legal_integer_min', type=int, default=1)
    parser.add_argument('--legal_integer_max', type=int, default=5)
    parser.add_argument('--graph_role_encoding_enabled', type=parse_bool, default=True)
    parser.add_argument('--graph_role_post_projector_enabled', type=parse_bool, default=False)
    parser.add_argument('--graph_distance_post_projector_enabled', type=parse_bool, default=False)
    parser.add_argument('--graph_role_residual_gated_enabled', type=parse_bool, default=False)
    parser.add_argument('--graph_role_residual_alpha_init', type=float, default=0.0)
    parser.add_argument('--graph_role_residual_layernorm_enabled', type=parse_bool, default=False)
    parser.add_argument('--graph_role_residual_clip_ratio', type=float, default=0.0)
    parser.add_argument('--graph_distance_encoding_enabled', type=parse_bool, default=True)
    parser.add_argument('--graph_distance_residual_bucket_count', type=int, default=6)
    parser.add_argument('--graph_distance_residual_role_conditioned_enabled', type=parse_bool, default=False)
    parser.add_argument('--graph_distance_residual_alpha_init', type=float, default=0.0)
    parser.add_argument('--graph_distance_residual_clip_ratio', type=float, default=0.0)
    parser.add_argument('--distance_branch_freeze_epochs', type=int, default=0)
    parser.add_argument('--graph_corridor_alpha', type=float, default=1.0)
    parser.add_argument('--graph_projector_freeze_steps', type=int, default=0)
    parser.add_argument('--graph_projector_mode', type=str, default='mlp', choices=['mlp', 'llaga'])
    parser.add_argument('--llaga_projector_type', type=str, default='2-layer-mlp')
    parser.add_argument('--llaga_projector_ckpt', type=str, default='')
    parser.add_argument('--moco_enabled', action='store_true')
    parser.add_argument('--moco_queue_size', type=int, default=65536)
    parser.add_argument('--moco_momentum', type=float, default=0.999)
    parser.add_argument('--moco_temperature', type=float, default=0.07)
    parser.add_argument('--moco_lambda', type=float, default=0.05)
    parser.add_argument('--dual_view_enabled', action='store_true')
    parser.add_argument('--query_dropedge_rate', type=float, default=0.15)
    parser.add_argument('--key_dropedge_rate', type=float, default=0.20)
    parser.add_argument('--key_feature_mask_rate', type=float, default=0.10)
    parser.add_argument('--key_view_mode', type=str, default='feature_mask', choices=['feature_mask', 'dropedge'])
    parser.add_argument('--grbench_root', '--grbench-root', dest='grbench_root', type=str, default='')
    parser.add_argument('--grbench_domain', '--grbench-domain', '--domain', dest='grbench_domain', type=str, default='dblp')
    parser.add_argument('--grbench_desc_mode', '--grbench-desc-mode', dest='grbench_desc_mode', type=str, default='summary')
    parser.add_argument('--grbench_desc_max_nodes', '--grbench-desc-max-nodes', dest='grbench_desc_max_nodes', type=int, default=200)
    parser.add_argument(
        '--grbench_task_filter',
        type=str,
        default='all',
        choices=[
            'all',
            'counting',
            'entity',
            'hop',
            'path_witness',
            'topic',
            'author_lookup',
            'venue_lookup',
            'collab_rank',
            'citation_rank',
            'graph_lookup',
            'other',
        ],
    )
    parser.add_argument('--grbench_embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--grbench_cache_embeddings', action='store_true')
    parser.add_argument('--infection_enabled', action='store_true')
    parser.add_argument('--infection_k', type=int, default=0)
    parser.add_argument('--infection_clip_max', type=int, default=255)
    parser.add_argument('--collect_hsgs_node_scores', action='store_true')
    parser.add_argument('--hsgs_score_tau', type=float, default=128.0)
    parser.add_argument('--degree_bias_enabled', type=parse_bool, default=False)
    parser.add_argument('--node_identity_enabled', type=parse_bool, default=False)
    parser.add_argument('--align_lambda', type=float, default=0.0)
    parser.add_argument('--align_on_projected_tokens', type=parse_bool, default=True)
    parser.add_argument('--dynamic_align_lambda_enabled', type=parse_bool, default=False)
    parser.add_argument('--dynamic_align_lambda_min', type=float, default=0.0)
    parser.add_argument('--dynamic_align_lambda_max', type=float, default=0.0)
    parser.add_argument('--dynamic_align_lambda_delta_max', type=float, default=0.03)
    parser.add_argument('--dynamic_align_progress_ema_beta', type=float, default=0.7)
    parser.add_argument('--dynamic_align_progress_clip_min', type=float, default=0.0)
    parser.add_argument('--dynamic_align_progress_clip_max', type=float, default=1.0)
    parser.add_argument('--dynamic_align_progress_intercept', type=float, default=0.3167507165736947)
    parser.add_argument('--dynamic_align_progress_coef_em', type=float, default=1.13560542)
    parser.add_argument('--dynamic_align_progress_coef_digit8', type=float, default=-0.38246612)
    parser.add_argument('--dynamic_align_progress_coef_single_digit', type=float, default=0.36927145)
    parser.add_argument('--dynamic_align_hold_progress_threshold', type=float, default=1.1)
    parser.add_argument('--dynamic_align_hold_lambda', type=float, default=0.0)
    parser.add_argument('--dynamic_align_digit8_target_enabled', type=parse_bool, default=False)
    parser.add_argument('--dynamic_align_digit8_target', type=float, default=0.0)
    parser.add_argument('--dynamic_align_digit8_target_gain', type=float, default=0.0)
    parser.add_argument('--dynamic_align_digit8_target_deadband', type=float, default=0.0)
    parser.add_argument('--dynamic_align_digit8_band_enabled', type=parse_bool, default=False)
    parser.add_argument('--dynamic_align_digit8_band_low', type=float, default=0.0)
    parser.add_argument('--dynamic_align_digit8_band_high', type=float, default=0.0)
    parser.add_argument('--dynamic_align_digit8_band_gain_above', type=float, default=0.0)
    parser.add_argument('--dynamic_align_digit8_band_gain_below', type=float, default=0.0)
    parser.add_argument('--dynamic_align_digit8_band_deadband', type=float, default=0.0)
    parser.add_argument('--align_lr_compensation_enabled', type=parse_bool, default=False)
    parser.add_argument('--align_lr_compensation_start_epoch', type=float, default=2.0)
    parser.add_argument('--align_lr_compensation_ref_lr', type=float, default=0.0)
    parser.add_argument('--align_lr_compensation_min_scale', type=float, default=1.0)
    parser.add_argument('--align_lr_compensation_max_scale', type=float, default=4.0)
    parser.add_argument('--axis_momentum_reset_enabled', type=parse_bool, default=False)
    parser.add_argument('--axis_momentum_reset_threshold', type=float, default=0.55)
    parser.add_argument('--axis_momentum_reset_guard_band', type=float, default=0.05)
    parser.add_argument('--axis_momentum_reset_delta_tol', type=float, default=0.03)
    parser.add_argument('--axis_momentum_reset_cooldown_epochs', type=int, default=1)
    parser.add_argument('--axis_momentum_reset_clear_exp_avg_sq', type=parse_bool, default=False)
    parser.add_argument('--graph_metric_max_nodes', type=int, default=5000)
    parser.add_argument('--run_name', '--run-name', dest='run_name', type=str, default='')
    parser.add_argument('--no_wandb', action='store_true')
    parser.add_argument('--checkpoint_path', '--checkpoint-path', dest='checkpoint_path', type=str, default='')
    parser.add_argument('--resume_training', type=parse_bool, default=False)
    parser.add_argument('--checkpoint_trainable_only', type=parse_bool, default=True)
    parser.add_argument('--save_every_epoch', type=parse_bool, default=False)
    parser.add_argument(
        '--best_checkpoint_metric',
        type=str,
        default='val_loss',
        choices=['val_loss', 'val_em', 'val_em_digit8', 'val_em_legal_integer'],
    )
    parser.add_argument('--best_checkpoint_digit8_threshold', type=float, default=0.1)
    parser.add_argument('--best_checkpoint_fallback_to_val_loss', type=parse_bool, default=False)
    parser.add_argument('--skip_test', action='store_true')
    parser.add_argument('--skip_checkpoint_optimizer', action='store_true')
    parser.add_argument('--epoch_diagnostics_enabled', type=parse_bool, default=False)
    parser.add_argument('--epoch_diagnostics_digit_min_length', type=int, default=6)
    parser.add_argument('--epoch_diagnostics_keep_predictions', type=parse_bool, default=False)
    parser.add_argument('--monitor_subset_enabled', type=parse_bool, default=False)
    parser.add_argument('--monitor_subset_size', type=int, default=0)
    parser.add_argument('--monitor_every_optimizer_steps', type=int, default=0)
    parser.add_argument('--monitor_eval_batch_size', type=int, default=0)
    parser.add_argument('--monitor_subset_seed', type=int, default=0)
    parser.add_argument('--monitor_digit_min_length', type=int, default=6)
    parser.add_argument('--monitor_keep_predictions', type=parse_bool, default=False)
    parser.add_argument('--monitor_save_best_checkpoint', type=parse_bool, default=True)
    parser.add_argument('--monitor_best_digit8_threshold', type=float, default=0.1)
    parser.add_argument('--monitor_drives_axis_controller', type=parse_bool, default=True)
    parser.set_defaults(**defaults)
    return parser


def _resolve_model_path(args):
    if args.llm_model_path:
        return args.llm_model_path
    model_root = Path(args.model_root)
    default_ref = DEFAULT_LLM_MODEL_PATHS.get(args.llm_model_name, args.llm_model_name)
    default_basename = Path(default_ref).name

    local_candidates = [
        model_root / args.llm_model_name,
        model_root / default_basename,
        model_root.parent / args.llm_model_name,
        model_root.parent / default_basename,
        model_root.parent.parent / args.llm_model_name,
        model_root.parent.parent / default_basename,
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)
    return default_ref


def finalize_args(args):
    repo = repo_root()
    args.data_root = args.data_root or os.environ.get("DATA_ROOT", str(datasets_processed_root()))
    args.model_root = args.model_root or os.environ.get('MODEL_ROOT', str(repo / 'models'))
    args.output_root = args.output_root or os.environ.get('OUTPUT_ROOT', str(repo / 'outputs'))
    if args.output_dir == 'output':
        args.output_dir = args.output_root
    args.llm_model_path = _resolve_model_path(args)
    if not args.grbench_root:
        env_grbench = os.environ.get('GRBENCH_ROOT')
        if env_grbench:
            args.grbench_root = env_grbench
        else:
            candidates = [
                Path(args.data_root) / 'grbench',
                Path(args.data_root) / 'GRBENCH',
                repo / 'GRBENCH',
            ]
            for candidate in candidates:
                if candidate.exists():
                    args.grbench_root = str(candidate)
                    break
            else:
                args.grbench_root = str(Path(args.data_root) / 'GRBENCH')
    projector_ckpt_value = str(getattr(args, "llaga_projector_ckpt", "")).strip()
    if projector_ckpt_value.lower() in _DISABLED_PROJECTOR_CKPT_VALUES:
        args.llaga_projector_ckpt = ''
        args.llaga_projector_ckpt_disabled = projector_ckpt_value.lower() not in {""}
    else:
        args.llaga_projector_ckpt = projector_ckpt_value
        args.llaga_projector_ckpt_disabled = False
    if args.graph_projector_mode == 'llaga' and not args.llaga_projector_ckpt and not args.llaga_projector_ckpt_disabled:
        model_root = Path(args.model_root)
        candidates = [
            model_root / 'llaga_projector' / 'model.pt',
            model_root / 'llaga_projector' / 'pytorch_model.bin',
            model_root / 'llaga_projector.pt',
        ]
        for candidate in candidates:
            if candidate.exists():
                args.llaga_projector_ckpt = str(candidate)
                break
    if not args.run_name:
        args.run_name = f"{args.dataset}_{args.model_name}_seed{args.seed}"
    return args


def parse_args(argv=None):
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', type=str, default='')
    known, _ = bootstrap.parse_known_args(argv)
    defaults = load_config_defaults(known.config)
    parser = build_parser(defaults=defaults)
    args = parser.parse_args(argv)
    return finalize_args(args)

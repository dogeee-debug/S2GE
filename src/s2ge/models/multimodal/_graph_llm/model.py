"""Graph encoder/projector/native-decoder assembly for S2GE.

The graph-language interface is explicit: a sampled PyG graph is encoded into
ordered graph tokens, projected into the decoder embedding space, and inserted
into reserved prompt positions before native causal generation.
"""

import contextlib
import math
import os

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from s2ge.data.transforms.dual_view import build_views
from s2ge.models.contrastive import MoCoGraph
from s2ge.models.encoders import DeepS2GEEncoderFactory, GraphReadout
from s2ge.models.llm import load_causal_lm, load_tokenizer
from s2ge.models.multimodal._graph_llm.graph_encoding import (
    _add_node_identity_encoding,
    _build_query_distance_residual,
    _build_query_role_residual,
    _build_query_syntax_bias,
    _coarse_bucketize_query_distance,
    _build_degree_token_bias,
    _bucketize_query_distance,
    _corrupt_graph_token_sequence,
    _encode_graph_features,
    _encode_graph_token_batch,
    _encode_node_embeddings,
    _ensure_node_score_storage,
    _inject_infection_features,
    _pool_graph_embeds,
    _resolve_graph_edge_attr,
    _resolve_graph_node_features,
    _shuffle_graph_token_sequence,
    _update_node_score_stats,
    export_hsgs_node_scores,
    export_hsgs_node_scores_from_tensors,
    get_hsgs_node_score_tensors,
)
from s2ge.models.multimodal._graph_llm.losses import _build_subgraph_adj_from_batch, _compute_align_loss, _compute_moco_loss
from s2ge.models.multimodal._graph_llm.postprocess import _extract_answer_text, _postprocess_prediction
from s2ge.models.multimodal._graph_llm.text_layout import _build_text_tensors, _prepare_text_layout
from s2ge.models.projectors import build_projector


class GraphLLM(nn.Module):
    """Combine the graph encoder, interface projector, and causal LM.

    The released path uses node graph tokens with query-aware ordering and the
    adjacency-based alignment objective. MoCo, token corruption, and auxiliary
    residual branches remain opt-in experimental controls.
    """
    _pool_graph_embeds = _pool_graph_embeds
    _encode_node_embeddings = _encode_node_embeddings
    _resolve_graph_node_features = _resolve_graph_node_features
    _resolve_graph_edge_attr = _resolve_graph_edge_attr
    _inject_infection_features = _inject_infection_features
    _ensure_node_score_storage = _ensure_node_score_storage
    _update_node_score_stats = _update_node_score_stats
    export_hsgs_node_scores = export_hsgs_node_scores
    get_hsgs_node_score_tensors = get_hsgs_node_score_tensors
    export_hsgs_node_scores_from_tensors = export_hsgs_node_scores_from_tensors
    _encode_graph_features = _encode_graph_features
    _build_degree_token_bias = _build_degree_token_bias
    _add_node_identity_encoding = _add_node_identity_encoding
    _build_query_distance_residual = _build_query_distance_residual
    _build_query_role_residual = _build_query_role_residual
    _bucketize_query_distance = _bucketize_query_distance
    _coarse_bucketize_query_distance = _coarse_bucketize_query_distance
    _build_query_syntax_bias = _build_query_syntax_bias
    _encode_graph_token_batch = _encode_graph_token_batch
    _shuffle_graph_token_sequence = _shuffle_graph_token_sequence
    _corrupt_graph_token_sequence = _corrupt_graph_token_sequence
    _prepare_text_layout = _prepare_text_layout
    _build_text_tensors = _build_text_tensors
    _build_subgraph_adj_from_batch = _build_subgraph_adj_from_batch
    _compute_align_loss = _compute_align_loss
    _compute_moco_loss = _compute_moco_loss
    _extract_answer_text = staticmethod(_extract_answer_text)
    _postprocess_prediction = classmethod(_postprocess_prediction)

    def __init__(self, args, graph_type=None, init_prompt=None, **kwargs):
        """Build all graph, interface, and decoder components from CLI args."""
        super().__init__()
        self.args = args
        self.graph_type = graph_type
        self.init_prompt = init_prompt
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens
        # Interface mode and controlled interventions.
        self.graph_tokens_disabled = bool(getattr(args, "graph_tokens_disabled", False))
        self.moco_enabled = args.moco_enabled
        self.moco_lambda = args.moco_lambda
        self.graph_token_mode = getattr(args, "graph_token_mode", "single")
        self.graph_token_budget = max(1, int(getattr(args, "graph_token_budget", 32)))
        self.graph_query_syntax_enabled = bool(getattr(args, "graph_query_syntax_enabled", True))
        self.graph_token_shuffle_enabled = bool(getattr(args, "graph_token_shuffle_enabled", False))
        self.graph_token_corrupt_enabled = bool(getattr(args, "graph_token_corrupt_enabled", False))
        self.graph_token_corrupt_std = max(0.0, float(getattr(args, "graph_token_corrupt_std", 1.0)))
        self.legal_integer_decode_enabled = bool(getattr(args, "legal_integer_decode_enabled", False))
        self.legal_integer_min = int(getattr(args, "legal_integer_min", 1))
        self.legal_integer_max = int(getattr(args, "legal_integer_max", self.legal_integer_min))
        # Role-based perception and proximity encodings.
        self.graph_role_encoding_enabled = bool(getattr(args, "graph_role_encoding_enabled", True))
        self.graph_role_post_projector_enabled = bool(getattr(args, "graph_role_post_projector_enabled", False))
        self.graph_distance_post_projector_enabled = bool(getattr(args, "graph_distance_post_projector_enabled", False))
        self.graph_role_residual_gated_enabled = bool(getattr(args, "graph_role_residual_gated_enabled", False))
        self.graph_role_residual_alpha_init = float(getattr(args, "graph_role_residual_alpha_init", 0.0))
        self.graph_role_residual_layernorm_enabled = bool(getattr(args, "graph_role_residual_layernorm_enabled", False))
        self.graph_role_residual_clip_ratio = max(0.0, float(getattr(args, "graph_role_residual_clip_ratio", 0.0)))
        self.graph_distance_encoding_enabled = bool(getattr(args, "graph_distance_encoding_enabled", True))
        self.graph_distance_residual_role_conditioned_enabled = bool(
            getattr(args, "graph_distance_residual_role_conditioned_enabled", False)
        )
        self.distance_branch_freeze_epochs = max(0, int(getattr(args, "distance_branch_freeze_epochs", 0)))
        self.graph_corridor_alpha = float(getattr(args, "graph_corridor_alpha", 1.0))
        self.graph_projector_freeze_steps = max(0, int(getattr(args, "graph_projector_freeze_steps", 0)))
        # Auxiliary structural features and objectives.
        self.degree_bias_enabled = bool(getattr(args, "degree_bias_enabled", False))
        self.node_identity_enabled = bool(getattr(args, "node_identity_enabled", False))
        self.align_lambda = float(getattr(args, "align_lambda", 0.0))
        self.align_on_projected_tokens = bool(getattr(args, "align_on_projected_tokens", True))
        self.infection_enabled = bool(getattr(args, "infection_enabled", False))
        self.infection_k = max(0, int(getattr(args, "infection_k", 0)))
        self.infection_clip_max = max(1, int(getattr(args, "infection_clip_max", 255)))
        self.collect_hsgs_node_scores = bool(getattr(args, "collect_hsgs_node_scores", False))
        self.hsgs_score_tau = float(getattr(args, "hsgs_score_tau", 128.0))
        # Optional contrastive dual-view path. It is inactive in main configs.
        self.dual_view_enabled = getattr(args, "dual_view_enabled", False)
        self.query_dropedge_rate = getattr(args, "query_dropedge_rate", 0.15)
        self.key_dropedge_rate = getattr(args, "key_dropedge_rate", 0.15)
        self.key_feature_mask_rate = getattr(args, "key_feature_mask_rate", 0.1)
        self.key_view_mode = getattr(args, "key_view_mode", "feature_mask")
        self.autocast_dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16
        self.use_deepspeed = getattr(args, "use_deepspeed", False)
        self.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", -1)))

        self.tokenizer = load_tokenizer(args.llm_model_path, use_fast=False)
        self.model = self._build_llm(args)
        self.word_embedding = self.model.get_input_embeddings()
        llm_input_device = self.llm_input_device
        self.graph_encoder = DeepS2GEEncoderFactory.build(args).to(llm_input_device)
        self.graph_encoder = self.graph_encoder.to(torch.float32)
        self.readout = GraphReadout(
            bfs_order_enabled=getattr(args, "graph_bfs_order_enabled", False),
            query_syntax_enabled=self.graph_query_syntax_enabled,
        )
        self.llm_embed_dim = self.word_embedding.weight.shape[1]
        self.llm_embed_dtype = self.word_embedding.weight.dtype
        self.projector = build_projector(args, self.llm_embed_dim).to(llm_input_device)
        self._maybe_load_projector_ckpt(args)
        self.degree_projector = nn.Linear(1, args.gnn_hidden_dim, bias=False).to(llm_input_device) if self.degree_bias_enabled else None
        self.node_id_embed = nn.Embedding(self.graph_token_budget, args.gnn_hidden_dim).to(llm_input_device) if self.node_identity_enabled and self.graph_token_mode == "nodes" else None
        distance_bucket_count = self.graph_token_budget + 2
        self.query_role_vocab_size = 4
        self.query_role_embed = nn.Embedding(4, args.gnn_hidden_dim).to(llm_input_device) if self.graph_query_syntax_enabled and self.graph_role_encoding_enabled and self.graph_token_mode == "nodes" else None
        if self.query_role_embed is not None:
            nn.init.zeros_(self.query_role_embed.weight)
        self.query_role_residual_embed = nn.Embedding(4, self.llm_embed_dim).to(llm_input_device) if self.graph_query_syntax_enabled and self.graph_role_post_projector_enabled and self.graph_token_mode == "nodes" else None
        if self.query_role_residual_embed is not None:
            nn.init.zeros_(self.query_role_residual_embed.weight)
        self.query_role_residual_alpha = nn.Parameter(torch.tensor(self.graph_role_residual_alpha_init, device=llm_input_device, dtype=torch.float32)) if self.query_role_residual_embed is not None and self.graph_role_residual_gated_enabled else None
        self.query_role_residual_norm = nn.LayerNorm(self.llm_embed_dim).to(llm_input_device) if self.query_role_residual_embed is not None and self.graph_role_residual_layernorm_enabled else None
        # Coarse distance bins: 0, 1, 2, 3, >=4, disconnected = 6 buckets.
        distance_residual_bucket_count = int(getattr(args, "graph_distance_residual_bucket_count", 6))
        self.graph_distance_residual_alpha_init = float(getattr(args, "graph_distance_residual_alpha_init", 0.0))
        self.graph_distance_residual_clip_ratio = float(getattr(args, "graph_distance_residual_clip_ratio", 0.0))
        if self.graph_query_syntax_enabled and self.graph_distance_post_projector_enabled and self.graph_token_mode == "nodes":
            distance_residual_vocab_size = distance_residual_bucket_count
            if self.graph_distance_residual_role_conditioned_enabled:
                distance_residual_vocab_size *= self.query_role_vocab_size
            self.distance_src_residual_embed = nn.Embedding(distance_residual_vocab_size, self.llm_embed_dim).to(llm_input_device)
            self.distance_dst_residual_embed = nn.Embedding(distance_residual_vocab_size, self.llm_embed_dim).to(llm_input_device)
            nn.init.zeros_(self.distance_src_residual_embed.weight)
            nn.init.zeros_(self.distance_dst_residual_embed.weight)
            self.distance_residual_alpha = nn.Parameter(
                torch.tensor(self.graph_distance_residual_alpha_init, device=llm_input_device, dtype=torch.float32)
            )
        else:
            self.distance_src_residual_embed = None
            self.distance_dst_residual_embed = None
            self.distance_residual_alpha = None
        if self.graph_query_syntax_enabled and self.graph_distance_encoding_enabled and self.graph_token_mode == "nodes":
            self.distance_src_embed = nn.Embedding(distance_bucket_count, args.gnn_hidden_dim).to(llm_input_device)
            self.distance_dst_embed = nn.Embedding(distance_bucket_count, args.gnn_hidden_dim).to(llm_input_device)
            self.corridor_projector = nn.Linear(1, args.gnn_hidden_dim, bias=False).to(llm_input_device)
            nn.init.zeros_(self.distance_src_embed.weight)
            nn.init.zeros_(self.distance_dst_embed.weight)
            nn.init.zeros_(self.corridor_projector.weight)
        else:
            self.distance_src_embed = None
            self.distance_dst_embed = None
            self.corridor_projector = None
        self.infection_projector = nn.Linear(self.infection_k, args.gnn_in_dim).to(llm_input_device) if self.infection_enabled and self.infection_k > 0 else None
        self._node_score_sum = None
        self._node_score_count = None
        self._feature_store_cache = {}
        self._edge_type_embed_cache = {}
        self._latest_aux_metrics = {}
        self._projector_is_frozen = False
        self._distance_branch_is_frozen = False
        self.moco = None
        if self.moco_enabled:
            self.moco = MoCoGraph(self.graph_encoder, self.projector, embed_dim=self.llm_embed_dim, queue_size=args.moco_queue_size, momentum=args.moco_momentum, temperature=args.moco_temperature).to(llm_input_device)
        if args.grad_checkpointing:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
    def inference(self, samples):
        """Generate and task-postprocess native decoder answers for a batch."""
        graph_token_embeds, graph_token_mask = self.encode_graph_tokens(samples)
        graph_token_embeds = graph_token_embeds.to(dtype=self.llm_embed_dtype)
        graph_token_counts = graph_token_mask.sum(dim=1).tolist()
        token_ids, attention_mask, _, graph_positions, graph_token_counts = self._prepare_text_layout(samples, include_labels=False, graph_token_counts=graph_token_counts, return_graph_spans=True)
        inputs_embeds = self.word_embedding(token_ids)
        inputs_embeds = inputs_embeds.clone()
        for batch_idx in range(inputs_embeds.size(0)):
            token_count = int(graph_token_counts[batch_idx].item())
            graph_start = int(graph_positions[batch_idx].item())
            inputs_embeds[batch_idx, graph_start : graph_start + token_count] = graph_token_embeds[batch_idx, :token_count]
        eos_token_id = self.tokenizer.eos_token_id
        if isinstance(eos_token_id, list):
            eos_token_id = eos_token_id[0] if eos_token_id else 128001
        with self.maybe_autocast():
            outputs = self.model.generate(inputs_embeds=inputs_embeds, max_new_tokens=self.max_new_tokens, attention_mask=attention_mask, use_cache=True, eos_token_id=eos_token_id, pad_token_id=eos_token_id)
        raw_pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        task_types = samples.get("task_type", [None] * len(raw_pred))
        cleaned_pred = [self._postprocess_prediction(pred, task_type) for pred, task_type in zip(raw_pred, task_types)]
        return {
            "id": samples["id"],
            "pred": cleaned_pred,
            "label": samples["label"],
            "question": samples["question"],
            "task_type": task_types,
            "desc": samples["desc"],
            "gt_path": samples.get("gt_path", None),
            "domain": samples.get("domain", None),
            "meta": samples.get("meta", None),
        }

    def latest_aux_metrics(self):
        """Return a copy of diagnostics collected during the latest forward."""
        return dict(self._latest_aux_metrics)

    def update_projector_freeze_state(self, optimizer_step):
        """Apply the optional step-based projector warm-start freeze."""
        should_freeze = self.graph_projector_freeze_steps > 0 and optimizer_step < self.graph_projector_freeze_steps
        if should_freeze == self._projector_is_frozen:
            return
        for param in self.projector.parameters():
            param.requires_grad = not should_freeze
        self._projector_is_frozen = should_freeze

    def update_distance_branch_freeze_state(self, epoch):
        """Apply the optional epoch-based distance-residual freeze."""
        should_freeze = (
            self.distance_branch_freeze_epochs > 0
            and epoch < self.distance_branch_freeze_epochs
            and self.distance_src_residual_embed is not None
        )
        if should_freeze == self._distance_branch_is_frozen:
            return
        modules = [self.distance_src_residual_embed, self.distance_dst_residual_embed]
        for module in modules:
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = not should_freeze
        if self.distance_residual_alpha is not None:
            self.distance_residual_alpha.requires_grad = not should_freeze
        self._distance_branch_is_frozen = should_freeze

    def print_trainable_params(self):
        """Return ``(trainable_parameters, all_parameters)`` for logging."""
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        return trainable_params, all_param
    def _build_llm(self, args):
        model = load_causal_lm(args, args.llm_model_path)
        if args.llm_frozen == "True":
            for _, param in model.named_parameters():
                param.requires_grad = False
        else:
            if args.llm_load_4bit:
                model = prepare_model_for_kbit_training(model)
            lora_cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            model = get_peft_model(model, lora_cfg)
        return model

    def _maybe_load_projector_ckpt(self, args):
        projector_ckpt = getattr(args, "llaga_projector_ckpt", "")
        if getattr(args, "graph_projector_mode", "legacy") != "llaga" or not projector_ckpt:
            return
        state_dict = torch.load(projector_ckpt, map_location="cpu")
        if any("mm_projector" in key for key in state_dict):
            state_dict = {key.split("mm_projector.", 1)[1]: value for key, value in state_dict.items() if "mm_projector." in key}
        self.projector.load_state_dict(state_dict, strict=False)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def llm_device(self):
        return next(self.model.parameters()).device

    @property
    def llm_input_device(self):
        if hasattr(self, "word_embedding") and self.word_embedding is not None:
            return self.word_embedding.weight.device
        return self.llm_device

    def maybe_autocast(self, dtype=None):
        return contextlib.nullcontext()

    def _build_graph_views(self, graphs):
        return build_views(graphs, training=self.training, enabled=self.dual_view_enabled, query_dropedge_rate=self.query_dropedge_rate, key_view_mode=self.key_view_mode, key_dropedge_rate=self.key_dropedge_rate, key_feature_mask_rate=self.key_feature_mask_rate)

    def encode_graph(self, batch):
        """Return unprojected pooled graph embeddings for compatibility."""
        return self._encode_graph_features(batch["graph"], encoder=self.graph_encoder, projector=None)

    def encode_graph_tokens(self, batch):
        """Return projected graph-token sequences and their validity mask."""
        graph_token_embeds, graph_token_mask, _, _ = self._encode_graph_token_batch(batch["graph"], encoder=self.graph_encoder, projector=self.projector)
        return graph_token_embeds, graph_token_mask

    def project(self, graph_latent):
        """Project graph-space tensors into the decoder embedding space."""
        projector_dtype = next(self.projector.parameters()).dtype
        graph_latent = graph_latent.to(dtype=projector_dtype)
        return self.projector(graph_latent)

    def compute_loss(self, samples, include_contrastive=True):
        """Combine causal-LM, adjacency-alignment, and optional MoCo losses."""
        base_graphs = samples["graph"]
        query_graphs, key_graphs = self._build_graph_views(base_graphs)
        self._latest_aux_metrics = {}
        self._latest_aux_metrics["distance_branch_frozen"] = float(self._distance_branch_is_frozen)
        graph_token_embeds, graph_token_mask, pooled_graph_embeds, graph_token_node_ids = self._encode_graph_token_batch(query_graphs, encoder=self.graph_encoder, projector=self.projector)
        align_loss = self._compute_align_loss(graph_token_embeds, graph_token_mask, graph_token_node_ids, query_graphs)
        graph_token_embeds = graph_token_embeds.to(dtype=self.llm_embed_dtype)
        inputs_embeds, attention_mask, label_input_ids = self._build_text_tensors(samples, graph_token_embeds, graph_token_mask=graph_token_mask)
        with self.maybe_autocast():
            outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True, labels=label_input_ids)
        total_loss = outputs.loss
        self._latest_aux_metrics["lm_loss"] = float(outputs.loss.detach().item())
        if align_loss is not None:
            if self.training:
                align_grads = torch.autograd.grad(
                    self.align_lambda * align_loss,
                    [param for param in self.parameters() if param.requires_grad],
                    retain_graph=True,
                    allow_unused=True,
                )
                align_grad_sq_sum = 0.0
                for grad in align_grads:
                    if grad is None:
                        continue
                    align_grad_sq_sum += float(grad.detach().float().pow(2).sum().item())
                self._latest_aux_metrics["align_grad_norm"] = math.sqrt(align_grad_sq_sum)
            total_loss = total_loss + self.align_lambda * align_loss
        if include_contrastive:
            nce_loss = self._compute_moco_loss(query_graphs, key_graphs, self.project(pooled_graph_embeds))
            if nce_loss is not None:
                total_loss = total_loss + self.moco_lambda * nce_loss
                self._latest_aux_metrics["nce_loss"] = float(nce_loss.detach().item())
        return total_loss

    def forward(self, samples):
        """Compute the configured training objective for one batch."""
        return self.compute_loss(samples, include_contrastive=True)

"""Convenience exports for datasets, collators, and preprocessing."""

try:
    from s2ge.data.collators.graph_llm import build_graph_llm_collate_fn
except ModuleNotFoundError:
    build_graph_llm_collate_fn = None
try:
    from s2ge.data.datasets.grbench import GRBenchDataset
except ModuleNotFoundError:
    GRBenchDataset = None
try:
    from s2ge.data.preprocess import (
        build_grbench,
        validate_dataset_ready,
    )
except ModuleNotFoundError:
    build_grbench = None
    validate_dataset_ready = None
try:
    from s2ge.data.registry import DATASET_REGISTRY, build_collate_fn, build_dataset
except ModuleNotFoundError:
    DATASET_REGISTRY = None
    build_collate_fn = None
    build_dataset = None
try:
    from s2ge.data.samplers.hsgs import hsgs_fanouts
    from s2ge.data.samplers.pyg_neighbor import sample_graph_with_hsgs
    from s2ge.data.samplers.seed_select import select_seed_nodes
except ModuleNotFoundError:
    hsgs_fanouts = None
    sample_graph_with_hsgs = None
    select_seed_nodes = None
from s2ge.data.schemas import GraphTextBatch, GraphTextSample
try:
    from s2ge.data.transforms.dual_view import augment_dropedge, augment_feature_mask, build_views
except ModuleNotFoundError:
    augment_dropedge = None
    augment_feature_mask = None
    build_views = None

__all__ = [
    "GraphTextSample",
    "GraphTextBatch",
    "GRBenchDataset",
    "DATASET_REGISTRY",
    "build_dataset",
    "build_collate_fn",
    "build_graph_llm_collate_fn",
    "hsgs_fanouts",
    "select_seed_nodes",
    "sample_graph_with_hsgs",
    "augment_dropedge",
    "augment_feature_mask",
    "build_views",
    "build_grbench",
    "validate_dataset_ready",
]

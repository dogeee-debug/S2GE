"""Dataset and collator construction from public training arguments."""

from s2ge.data.collators.graph_llm import build_graph_llm_collate_fn
from s2ge.data.datasets.grbench import GRBenchDataset


DATASET_REGISTRY = {
    'grbench': GRBenchDataset,
}

def build_dataset(args):
    """Build the dataset selected by parsed CLI arguments."""

    name = args.dataset
    if name == 'grbench':
        return GRBenchDataset(
            args.grbench_root,
            args.grbench_domain,
            args.grbench_desc_mode,
            args.grbench_desc_max_nodes,
            getattr(args, "graph_metric_max_nodes", 5000),
            getattr(args, "grbench_task_filter", "all"),
        )
    raise ValueError(f"Unsupported S2GE dataset: {name}. Use 'grbench' with one of: dblp, biomedical, goodreads, pubmed.")


def build_collate_fn(args):
    """Build the collate function that matches the selected model path."""

    return build_graph_llm_collate_fn(args)

"""Lazy preprocessing-builder lookup by public dataset name."""

from importlib import import_module

PREPROCESS_BUILDERS = {
    "grbench": "s2ge.data.preprocess._builders.grbench:build_grbench",
}


def get_preprocess_builder(dataset_name):
    """Return the builder registered for ``dataset_name`` or raise clearly."""
    try:
        target = PREPROCESS_BUILDERS[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported preprocess dataset: {dataset_name}") from exc
    module_name, attr_name = target.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)

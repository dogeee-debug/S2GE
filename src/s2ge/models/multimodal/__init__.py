"""Lazy public export for the graph-language model."""

__all__ = ["GraphLLM"]


def __getattr__(name):
    if name == "GraphLLM":
        from s2ge.models.multimodal.graph_llm import GraphLLM

        return GraphLLM
    raise AttributeError(name)

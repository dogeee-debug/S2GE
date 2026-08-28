"""Lazy internal export for the split GraphLLM implementation."""

__all__ = ["GraphLLM"]


def __getattr__(name):
    if name == "GraphLLM":
        from s2ge.models.multimodal._graph_llm.model import GraphLLM

        return GraphLLM
    raise AttributeError(name)

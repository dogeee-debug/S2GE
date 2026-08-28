"""Stable public import path for :class:`GraphLLM`."""

from pathlib import Path
import sys

try:
    from s2ge.models.multimodal._graph_llm.model import GraphLLM
except ModuleNotFoundError:
    src_root = Path(__file__).resolve().parents[3]
    src_root_str = str(src_root)
    if src_root_str not in sys.path:
        sys.path.insert(0, src_root_str)
    from s2ge.models.multimodal._graph_llm.model import GraphLLM

__all__ = ["GraphLLM"]

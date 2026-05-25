from typing import Any, Dict, List, TypedDict


class GraphTextSample(TypedDict, total=False):
    """Single graph-language example returned by a dataset."""

    id: Any
    question: str
    label: str
    desc: str
    task_type: str
    query_nodes: Any
    graph: Any
    meta: Dict[str, Any]


class GraphTextBatch(TypedDict, total=False):
    """Mini-batch consumed by the graph-LLM collator and trainer."""

    id: List[Any]
    question: List[str]
    label: List[str]
    desc: List[str]
    task_type: List[str]
    query_nodes: List[Any]
    graph: Any
    target_node_mask: Any
    meta: Dict[str, Any]

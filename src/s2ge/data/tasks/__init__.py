"""Task prompts, protocols, postprocessing, and metrics."""

from s2ge.data.tasks.metrics import compute_grbench_metrics, get_accuracy_grbench, load_grbench_metrics
from s2ge.data.tasks.postprocess import extract_answer_text, postprocess_task_prediction
from s2ge.data.tasks.prompts import get_grbench_prompt, infer_grbench_task_family, infer_grbench_task_type
from s2ge.data.tasks.specs import ENTITY_FAMILIES, TASK_PROTOCOLS, TaskProtocol, resolve_task_protocol

__all__ = [
    "TaskProtocol",
    "ENTITY_FAMILIES",
    "TASK_PROTOCOLS",
    "resolve_task_protocol",
    "infer_grbench_task_family",
    "infer_grbench_task_type",
    "get_grbench_prompt",
    "extract_answer_text",
    "postprocess_task_prediction",
    "compute_grbench_metrics",
    "load_grbench_metrics",
    "get_accuracy_grbench",
]

"""Output-format contracts for GRBench task families."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskProtocol:
    """Name and native-generation output contract for one task family."""
    name: str
    output_mode: str
    description: str = ""


ENTITY_FAMILIES = {"author_lookup", "venue_lookup", "collab_rank", "citation_rank", "graph_lookup", "other"}

TASK_PROTOCOLS = {
    "single_integer": TaskProtocol("single_integer", "single_integer", "Return a single integer only."),
    "plain_text": TaskProtocol("plain_text", "plain_text", "Return the answer as plain text only."),
    "keyword_list": TaskProtocol("keyword_list", "keyword_list", "Return exactly three comma-separated keywords."),
    "path_witness": TaskProtocol("path_witness", "plain_text", "Return yes plus one witness node, or no_path."),
}

TASK_FAMILY_PROTOCOL = {
    "hop": "single_integer",
    "counting": "single_integer",
    "path_witness": "path_witness",
    "topic": "keyword_list",
    "author_lookup": "plain_text",
    "venue_lookup": "plain_text",
    "collab_rank": "plain_text",
    "citation_rank": "plain_text",
    "graph_lookup": "plain_text",
    "other": "plain_text",
    None: "plain_text",
}


def resolve_task_protocol(task_family):
    """Resolve a task family to its output-format protocol."""
    return TASK_PROTOCOLS[TASK_FAMILY_PROTOCOL.get(task_family, "plain_text")]

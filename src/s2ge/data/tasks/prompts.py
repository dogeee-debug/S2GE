"""Task-family inference and exact-output prompts for GRBench generation."""

try:
    from s2ge.data.tasks.specs import ENTITY_FAMILIES
except ModuleNotFoundError:
    ENTITY_FAMILIES = {"author_lookup", "venue_lookup", "collab_rank", "citation_rank", "graph_lookup", "other"}

_SINGLE_INTEGER_PROMPT = (
    "Please answer the question using the graph. "
    "Output only a single integer as the answer. Do not output JSON or extra explanation."
)
_TOPIC_PROMPT = (
    "Please answer the question using the graph. "
    "Output exactly 3 specific research keywords separated by commas only. "
    "Do not output JSON, sentences, author names, or broad research fields. "
    'Format: "keyword one, keyword two, keyword three".'
)
_PLAIN_TEXT_PROMPT = (
    "Please answer the question using the graph. "
    "Output only the answer as a plain string. Do not output JSON or extra explanation."
)
_PATH_WITNESS_PROMPT = (
    "Please answer the question using the graph. "
    "Output exactly one of the following formats only: "
    "'yes<TAB>witness_node_id' or 'no_path'. "
    "Do not output JSON, explanations, or extra text."
)

GRBENCH_PROMPTS = {
    "hop": _SINGLE_INTEGER_PROMPT,
    "counting": _SINGLE_INTEGER_PROMPT,
    "path_witness": _PATH_WITNESS_PROMPT,
    "topic": _TOPIC_PROMPT,
    "author_lookup": _PLAIN_TEXT_PROMPT,
    "venue_lookup": _PLAIN_TEXT_PROMPT,
    "collab_rank": _PLAIN_TEXT_PROMPT,
    "citation_rank": _PLAIN_TEXT_PROMPT,
    "graph_lookup": _PLAIN_TEXT_PROMPT,
    "other": _PLAIN_TEXT_PROMPT,
}

_GENERIC_HOP_KEYWORDS = (
    "minimum number of graph hops",
    "minimum hop count",
    "shortest hop count",
    "graph hops at minimum",
    "hops at minimum separate",
)


def infer_grbench_task_type(question):
    """Infer the coarse task type from the released question templates."""
    q_lower = str(question).lower()
    if any(
        kw in q_lower
        for kw in (
            "witness node",
            "output yes and one witness node",
            "output yes plus one witness node",
            "or no_path",
            "within",
            "reachable from",
            "connect to",
        )
    ) and ("no_path" in q_lower or "witness" in q_lower):
        return "path_witness"
    if any(kw in q_lower for kw in ("keyword", "research interest", "area of research", "research area")):
        return "topic"
    return "hop"


def infer_grbench_task_family(question):
    """Infer the metric and prompt family from a question string."""
    q_lower = str(question).lower()
    task_type = infer_grbench_task_type(question)
    if task_type == "topic":
        return "topic"
    if task_type == "path_witness":
        return "path_witness"
    if any(kw in q_lower for kw in ("how many people at minimum", "least count of people", "to get introduced to", "to be familiar with")):
        return "hop"
    if any(kw in q_lower for kw in _GENERIC_HOP_KEYWORDS):
        return "hop"
    if any(kw in q_lower for kw in ("who are the authors of", "could you tell me the authors of", "who wrote the paper")):
        return "author_lookup"
    if any(kw in q_lower for kw in ("published", "venue", "journal", "conference")):
        return "venue_lookup"
    if any(kw in q_lower for kw in ("highest number of collaborations", "closest collaborator", "closest co-author")):
        return "collab_rank"
    if any(kw in q_lower for kw in ("most frequently cited", "most cited", "highest number of citations")):
        return "citation_rank"
    if any(kw in q_lower for kw in ("how many", "number of", "count of")):
        return "counting"
    if any(kw in q_lower for kw in ("who within", "which author", "who is", "who ")):
        return "graph_lookup"
    return "other"


def get_grbench_prompt(task_family):
    """Return the strict native-generation instruction for a task family."""
    return GRBENCH_PROMPTS.get(task_family, _PLAIN_TEXT_PROMPT)

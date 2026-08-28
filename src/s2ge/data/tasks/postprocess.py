"""Conservative extraction of task-formatted answers from decoder output."""

import json
import re

try:
    from s2ge.data.tasks.specs import resolve_task_protocol
except ModuleNotFoundError:
    def resolve_task_protocol(task_family):
        if task_family in {"hop", "counting"}:
            return type("Proto", (), {"output_mode": "single_integer"})()
        if task_family == "topic":
            return type("Proto", (), {"output_mode": "keyword_list"})()
        return type("Proto", (), {"output_mode": "plain_text"})()


def extract_answer_text(prediction):
    """Extract a JSON answer field when present, otherwise preserve text."""
    if isinstance(prediction, dict):
        return str(prediction.get("answer", "")).strip()
    if not isinstance(prediction, str):
        return str(prediction).strip()
    text = prediction.strip()
    try:
        match_obj = re.search(r"\{.*\}", text, flags=re.S)
        if match_obj:
            payload = json.loads(match_obj.group(0))
            return str(payload.get("answer", text)).strip()
    except Exception:
        pass
    return text


def _clean_generation_text(text):
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = re.split(r"</s>|<\|eot_id\|>|<\|end_of_text\|>", text)[0].strip()
    text = re.split(r"\[/?INST\]", text)[0].strip()
    text = re.sub(r"\][\s]*\(?/?INST\)?", "", text, flags=re.IGNORECASE).strip()
    text = re.split(r"```", text)[0].strip()
    return text


def _postprocess_keyword_list(text):
    text = re.sub(r"(?i)^(answer|keywords?)\s*:\s*", "", text).strip()
    text = re.sub(r"[\[\]\{\}\"']", "", text)
    text = re.sub(r"\s*[\r\n]+\s*", ", ", text)
    text = re.sub(r"\s*;\s*", ", ", text)
    parts = []
    for chunk in text.split(","):
        chunk = re.sub(r"^\s*\d+[\)\.\-:]\s*", "", chunk).strip()
        if chunk:
            parts.append(chunk)
        if len(parts) == 3:
            break
    return ", ".join(parts) if parts else text


def postprocess_task_prediction(prediction, task_family=None):
    """Apply only the normalization allowed by the task output protocol."""
    protocol = resolve_task_protocol(task_family)
    text = _clean_generation_text(extract_answer_text(prediction))
    if protocol.output_mode == "single_integer":
        match_obj = re.search(r"-?\d+", text)
        return match_obj.group(0) if match_obj else text
    if protocol.output_mode == "keyword_list":
        return _postprocess_keyword_list(text)
    return text

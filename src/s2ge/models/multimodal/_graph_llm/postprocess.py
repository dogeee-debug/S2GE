"""Model-level generation cleanup and optional legal-integer filtering."""

import json
import re


def _extract_answer_text(prediction):
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


def _postprocess_prediction(cls, prediction, task_type):
    text = _extract_answer_text(prediction)
    text = re.split(r"</s>|<\|eot_id\|>|<\|end_of_text\|>", text)[0].strip()
    text = re.split(r"\[/?INST\]", text)[0].strip()
    text = re.split(r"```", text)[0].strip()
    text = re.sub(r"<[^>]+>", "", text).strip()
    text = re.sub(r"\][\s]*\(?/?INST\)?", "", text, flags=re.IGNORECASE).strip()
    if task_type in {"hop", "counting"}:
        match_obj = re.search(r"-?\d+", text)
        if not match_obj:
            return text
        value = match_obj.group(0)
        if not bool(getattr(cls, "legal_integer_decode_enabled", False)):
            return value
        try:
            number = int(value)
        except ValueError:
            return ""
        lower = int(getattr(cls, "legal_integer_min", 1))
        upper = int(getattr(cls, "legal_integer_max", lower))
        if lower > upper:
            lower, upper = upper, lower
        return str(number) if lower <= number <= upper else ""
    if task_type == "topic":
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
    return text

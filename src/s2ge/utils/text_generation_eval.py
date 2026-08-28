"""Dependency-light text-generation diagnostics used by release evaluation."""

from __future__ import annotations

import re
from collections import Counter


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l_score(prediction: str, reference: str) -> float:
    """Compute token-level ROUGE-L F1 for one prediction/reference pair."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def corpus_rouge_l(pairs: list[tuple[str, str]]) -> float:
    """Return the arithmetic mean of per-example ROUGE-L scores."""
    if not pairs:
        return 0.0
    return sum(rouge_l_score(pred, ref) for pred, ref in pairs) / len(pairs)


def top_answer_rows(counter: Counter[str], total: int, top_k: int = 10) -> list[dict[str, float | int | str]]:
    """Convert answer counts into JSON-serializable frequency rows."""
    rows: list[dict[str, float | int | str]] = []
    for answer, count in counter.most_common(top_k):
        rows.append(
            {
                "answer": answer,
                "count": int(count),
                "rate": (float(count) / float(total)) if total else 0.0,
            }
        )
    return rows

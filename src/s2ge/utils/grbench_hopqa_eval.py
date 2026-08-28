"""Unified strict-format and output-collapse report for Core-HopQA JSONL."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from s2ge.utils.metrics import _extract_hop_answer, _parse_pred_json, load_grbench_metrics, normalize


_INTEGER_ONLY_RE = re.compile(r"^\s*-?\d+\s*$")


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _is_single_integer(text: str) -> bool:
    return bool(_INTEGER_ONLY_RE.fullmatch(str(text or "").strip()))


def _as_hop_key(label: Any) -> str:
    label_text = str(label).strip()
    return f"hop{label_text}" if re.fullmatch(r"-?\d+", label_text) else f"label:{label_text}"


def _counter_to_rows(counter: Counter[str], total: int, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for answer, count in counter.most_common(top_k):
        rows.append(
            {
                "answer": answer,
                "count": int(count),
                "rate": _safe_rate(int(count), total),
            }
        )
    return rows


def load_grbench_hopqa_eval_report(prediction_path: str | Path, top_k: int = 10) -> dict[str, Any]:
    """Compute benchmark, strict, format, and answer-frequency diagnostics."""
    prediction_path = Path(prediction_path)
    df = pd.read_json(prediction_path, lines=True)
    benchmark_metrics = load_grbench_metrics(prediction_path, dataset="grbench")

    overall = {
        "count": int(len(df)),
        "benchmark_exact_match": float(benchmark_metrics.get("exact_match", 0.0)),
        "strict_raw_exact_match": 0,
        "strict_parsed_exact_match": 0,
        "single_integer_exact_match": 0,
        "single_integer_format_rate": 0,
    }
    extracted_counter: Counter[str] = Counter()
    parsed_counter: Counter[str] = Counter()
    by_hop: dict[str, dict[str, Any]] = {}

    for row in df.itertuples(index=False):
        raw_pred = row.pred
        label = str(row.label).strip()
        parsed_answer, _ = _parse_pred_json(raw_pred)
        parsed_answer = str(parsed_answer).strip()
        extracted_answer = str(_extract_hop_answer(parsed_answer)).strip()
        hop_key = _as_hop_key(label)

        stats = by_hop.setdefault(
            hop_key,
            {
                "count": 0,
                "benchmark_exact_match": 0,
                "strict_raw_exact_match": 0,
                "strict_parsed_exact_match": 0,
                "single_integer_exact_match": 0,
                "single_integer_format_rate": 0,
                "extracted_answers": Counter(),
            },
        )

        strict_raw = 1 if normalize(str(raw_pred)) == normalize(label) else 0
        strict_parsed = 1 if normalize(parsed_answer) == normalize(label) else 0
        extracted_match = 1 if normalize(extracted_answer) == normalize(label) else 0
        single_integer_ok = 1 if _is_single_integer(parsed_answer) else 0
        single_integer_exact = 1 if single_integer_ok and extracted_match else 0

        overall["strict_raw_exact_match"] += strict_raw
        overall["strict_parsed_exact_match"] += strict_parsed
        overall["single_integer_exact_match"] += single_integer_exact
        overall["single_integer_format_rate"] += single_integer_ok

        stats["count"] += 1
        stats["benchmark_exact_match"] += extracted_match
        stats["strict_raw_exact_match"] += strict_raw
        stats["strict_parsed_exact_match"] += strict_parsed
        stats["single_integer_exact_match"] += single_integer_exact
        stats["single_integer_format_rate"] += single_integer_ok
        stats["extracted_answers"][extracted_answer] += 1

        extracted_counter[extracted_answer] += 1
        parsed_counter[parsed_answer] += 1

    total = overall["count"]
    majority_answer, majority_count = ("", 0)
    if extracted_counter:
        majority_answer, majority_count = extracted_counter.most_common(1)[0]

    report = {
        "prediction_path": str(prediction_path),
        "count": total,
        "benchmark_exact_match": overall["benchmark_exact_match"],
        "strict_raw_exact_match": _safe_rate(overall["strict_raw_exact_match"], total),
        "strict_parsed_exact_match": _safe_rate(overall["strict_parsed_exact_match"], total),
        "single_integer_exact_match": _safe_rate(overall["single_integer_exact_match"], total),
        "single_integer_format_rate": _safe_rate(overall["single_integer_format_rate"], total),
        "degenerate_majority_answer": {
            "answer": majority_answer,
            "count": int(majority_count),
            "rate": _safe_rate(int(majority_count), total),
        },
        "most_common_extracted_answers": _counter_to_rows(extracted_counter, total, top_k=top_k),
        "most_common_parsed_answers": _counter_to_rows(parsed_counter, total, top_k=top_k),
        "by_hop": {},
    }

    for hop_key in sorted(by_hop.keys()):
        stats = by_hop[hop_key]
        hop_total = int(stats["count"])
        majority_hop_answer, majority_hop_count = ("", 0)
        if stats["extracted_answers"]:
            majority_hop_answer, majority_hop_count = stats["extracted_answers"].most_common(1)[0]
        report["by_hop"][hop_key] = {
            "count": hop_total,
            "benchmark_exact_match": _safe_rate(int(stats["benchmark_exact_match"]), hop_total),
            "strict_raw_exact_match": _safe_rate(int(stats["strict_raw_exact_match"]), hop_total),
            "strict_parsed_exact_match": _safe_rate(int(stats["strict_parsed_exact_match"]), hop_total),
            "single_integer_exact_match": _safe_rate(int(stats["single_integer_exact_match"]), hop_total),
            "single_integer_format_rate": _safe_rate(int(stats["single_integer_format_rate"]), hop_total),
            "degenerate_majority_answer": {
                "answer": majority_hop_answer,
                "count": int(majority_hop_count),
                "rate": _safe_rate(int(majority_hop_count), hop_total),
            },
            "most_common_extracted_answers": _counter_to_rows(stats["extracted_answers"], hop_total, top_k=5),
        }

    return report


def format_grbench_hopqa_eval_report(report: dict[str, Any]) -> str:
    """Format a compact human-readable diagnostic report."""
    lines = [
        "GRBench HopQA Unified Evaluation",
        f"- samples: {report['count']}",
        f"- benchmark_exact_match: {report['benchmark_exact_match']:.4f}",
        f"- strict_raw_exact_match: {report['strict_raw_exact_match']:.4f}",
        f"- strict_parsed_exact_match: {report['strict_parsed_exact_match']:.4f}",
        f"- single_integer_exact_match: {report['single_integer_exact_match']:.4f}",
        f"- single_integer_format_rate: {report['single_integer_format_rate']:.4f}",
    ]
    majority = report.get("degenerate_majority_answer", {})
    if majority.get("count", 0) > 0:
        lines.append(
            "- majority_extracted_answer: "
            f"{majority.get('answer', '')} ({majority.get('count', 0)}/{report['count']}, {majority.get('rate', 0.0):.4f})"
        )
    for hop_key, hop_report in report.get("by_hop", {}).items():
        lines.append(
            f"- {hop_key}: benchmark={hop_report['benchmark_exact_match']:.4f}, "
            f"strict_parsed={hop_report['strict_parsed_exact_match']:.4f}, "
            f"single_integer={hop_report['single_integer_exact_match']:.4f}, "
            f"format={hop_report['single_integer_format_rate']:.4f}"
        )
    common_answers = report.get("most_common_extracted_answers", [])
    if common_answers:
        preview = ", ".join(
            f"{item['answer']}:{item['count']} ({item['rate']:.3f})"
            for item in common_answers[:5]
        )
        lines.append(f"- top_extracted_answers: {preview}")
    return "\n".join(lines)


def print_grbench_hopqa_eval_report(report: dict[str, Any]) -> None:
    """Print a formatted Core-HopQA diagnostic report."""
    print(format_grbench_hopqa_eval_report(report))


def dump_grbench_hopqa_eval_report(report: dict[str, Any], output_path: str | Path) -> None:
    """Persist a diagnostic report as UTF-8 JSON."""
    Path(output_path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

import json
import re
import string

import pandas as pd

from s2ge.data.tasks.postprocess import extract_answer_text
from s2ge.data.tasks.prompts import infer_grbench_task_family
from s2ge.data.tasks.specs import ENTITY_FAMILIES


_YES_WITNESS_RE = re.compile(r"^\s*yes(?:[\t, ]+(.+?))?\s*$", re.IGNORECASE)


def normalize(s: str) -> str:
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\b(<pad>)\b", " ", s)
    return " ".join(s.split())


def _parse_pred_json(pred):
    if isinstance(pred, dict):
        return pred.get("answer", ""), pred.get("path", None)
    if not isinstance(pred, str):
        pred = str(pred)
    try:
        match_obj = re.search(r"\{.*\}", pred.strip(), flags=re.S)
        if match_obj:
            obj = json.loads(match_obj.group(0))
            return obj.get("answer", pred), obj.get("path", None)
    except Exception:
        pass
    return pred, None


def _hopwise_accuracy(pred_path, gt_path):
    if gt_path is None or pred_path is None or not isinstance(gt_path, list) or not isinstance(pred_path, list) or len(gt_path) < 2:
        return None
    k = len(gt_path) - 1
    correct = 0
    for i in range(1, len(gt_path)):
        if i < len(pred_path) and pred_path[i] == gt_path[i]:
            correct += 1
    return correct / k


def _hallucination_rate(pred_path, node_id_set, edge_set):
    if pred_path is None or not isinstance(pred_path, list):
        return None
    if len(node_id_set) == 0 or len(edge_set) == 0:
        return None
    if len(pred_path) < 2:
        return 0.0
    invalid = 0
    total_steps = len(pred_path) - 1
    for i in range(1, len(pred_path)):
        src = str(pred_path[i - 1])
        dst = str(pred_path[i])
        if dst not in node_id_set or (src, dst) not in edge_set:
            invalid += 1
    return invalid / total_steps if total_steps > 0 else 0.0


def _token_f1(pred, label):
    pred_tokens = normalize(pred).split()
    label_tokens = normalize(label).split()
    if len(pred_tokens) == 0 or len(label_tokens) == 0:
        return 0.0
    common = sum(1 for token in pred_tokens if token in label_tokens)
    precision = common / len(pred_tokens)
    recall = common / len(label_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def infer_grbench_metric_task(question=None, label=None, task_type=None):
    inferred = infer_grbench_task_family(question)
    if inferred in {"topic", "hop", "counting", "path_witness"}:
        return inferred
    label_text = str(label or "").strip()
    if label_text == "no_path" or label_text.lower().startswith("yes\t") or label_text.lower().startswith("yes "):
        return "path_witness"
    if task_type in ENTITY_FAMILIES:
        return "entity"
    if "," in label_text and not label_text.replace(",", "").replace(" ", "").isdigit():
        return "topic"
    try:
        int(label_text.strip())
        return "counting"
    except ValueError:
        return "entity"


def _path_witness_metrics(pred_answer, label, gt_path, meta):
    pred_text = str(pred_answer or "").strip()
    label_text = str(label or "").strip()
    if label_text == "no_path":
        answer_ok = 1.0 if pred_text == "no_path" else 0.0
        return answer_ok, answer_ok

    valid_witnesses = set(str(x).strip() for x in (gt_path or [])[1:-1] if str(x).strip())
    if isinstance(meta, dict):
        fallback_witness = str(meta.get("witness", "")).strip()
        if fallback_witness:
            valid_witnesses.add(fallback_witness)

    match_obj = _YES_WITNESS_RE.match(pred_text)
    if not match_obj:
        return 0.0, 0.0
    pred_witness = str(match_obj.group(1) or "").strip()
    answer_ok = 1.0
    witness_ok = 1.0 if pred_witness and pred_witness in valid_witnesses else 0.0
    return answer_ok, witness_ok


def _extract_integer_answer(pred_answer):
    if isinstance(pred_answer, (int, float)):
        return str(int(pred_answer))
    text = str(pred_answer).strip()
    match_obj = re.search(r"-?\d+", text)
    return match_obj.group(0) if match_obj else text


def _split_topic_keywords(text):
    text = str(text or "").strip()
    text = re.sub(r"(?i)^(answer|keywords?)\s*:\s*", "", text)
    text = text.strip("[]{}() ")
    pieces = re.split(r"[,;\n|]+", text)
    keywords = []
    for piece in pieces:
        piece = re.sub(r"^\s*\d+[\)\.\-:]\s*", "", piece).strip(" \"'")
        if piece:
            keywords.append(piece)
    return keywords


def _topic_keyword_match(pred_keyword, gold_keyword):
    pred_norm = normalize(pred_keyword)
    gold_norm = normalize(gold_keyword)
    return bool(pred_norm) and bool(gold_norm) and (pred_norm == gold_norm or pred_norm in gold_norm or gold_norm in pred_norm)


def _topic_keyword_metrics(pred_answer, label):
    pred_keywords = _split_topic_keywords(pred_answer)
    gold_keywords = _split_topic_keywords(label)
    if not gold_keywords:
        return 0.0, 0.0, 0.0
    matched_gold = sum(any(_topic_keyword_match(pred, gold) for pred in pred_keywords) for gold in gold_keywords)
    matched_pred = sum(any(_topic_keyword_match(pred, gold) for gold in gold_keywords) for pred in pred_keywords)
    precision = matched_pred / len(pred_keywords) if pred_keywords else 0.0
    recall = matched_gold / len(gold_keywords) if gold_keywords else 0.0
    exact = 1.0 if matched_gold == len(gold_keywords) else 0.0
    return exact, _token_f1(pred_answer, label), recall


def compute_grbench_metrics(df, dataset=None):
    em_list, f1_list, hop_list, hall_list = [], [], [], []
    task_groups = {
        "hop": {"count": 0, "exact_match": [], "f1": []},
        "counting": {"count": 0, "exact_match": [], "f1": []},
        "entity": {"count": 0, "exact_match": [], "f1": []},
        "topic": {"count": 0, "exact_match": [], "f1": [], "keyword_recall": []},
        "path_witness": {"count": 0, "exact_match": [], "f1": [], "answer_accuracy": [], "witness_validity": [], "joint_accuracy": []},
    }
    node_id_set, edge_set = (set(), set())
    if dataset is not None and hasattr(dataset, "get_graph_meta"):
        node_id_set, edge_set = dataset.get_graph_meta()
    questions = df["question"].tolist() if "question" in df.columns else [None] * len(df)
    task_types = df["task_type"].tolist() if "task_type" in df.columns else [None] * len(df)
    metas = df["meta"].tolist() if "meta" in df.columns else [None] * len(df)
    for pred, label, gt_path, question, task_type, meta in zip(
        df["pred"].tolist(),
        df["label"].tolist(),
        df["gt_path"].tolist() if "gt_path" in df.columns else [None] * len(df),
        questions,
        task_types,
        metas,
    ):
        pred_answer, pred_path = _parse_pred_json(pred)
        pred_answer = extract_answer_text(pred_answer)
        label_str = str(label)
        resolved_task = infer_grbench_metric_task(question=question, label=label_str, task_type=task_type)
        if resolved_task == "topic":
            em, f1, keyword_recall = _topic_keyword_metrics(pred_answer, label_str)
            em_list.append(em)
            f1_list.append(f1)
            task_groups["topic"]["count"] += 1
            task_groups["topic"]["exact_match"].append(em)
            task_groups["topic"]["f1"].append(f1)
            task_groups["topic"]["keyword_recall"].append(keyword_recall)
        elif resolved_task in ("hop", "counting"):
            int_answer = _extract_integer_answer(pred_answer)
            em = 1.0 if normalize(int_answer) == normalize(label_str) else 0.0
            f1 = _token_f1(int_answer, label_str)
            em_list.append(em)
            f1_list.append(f1)
            task_groups[resolved_task]["count"] += 1
            task_groups[resolved_task]["exact_match"].append(em)
            task_groups[resolved_task]["f1"].append(f1)
        elif resolved_task == "path_witness":
            answer_acc, witness_valid = _path_witness_metrics(pred_answer, label_str, gt_path, meta)
            joint_acc = 1.0 if answer_acc and witness_valid else 0.0
            em_list.append(joint_acc)
            f1_list.append(joint_acc)
            task_groups["path_witness"]["count"] += 1
            task_groups["path_witness"]["exact_match"].append(joint_acc)
            task_groups["path_witness"]["f1"].append(joint_acc)
            task_groups["path_witness"]["answer_accuracy"].append(answer_acc)
            task_groups["path_witness"]["witness_validity"].append(witness_valid)
            task_groups["path_witness"]["joint_accuracy"].append(joint_acc)
        else:
            em = 1.0 if normalize(pred_answer) == normalize(label_str) else 0.0
            f1 = _token_f1(pred_answer, label_str)
            em_list.append(em)
            f1_list.append(f1)
            task_groups["entity"]["count"] += 1
            task_groups["entity"]["exact_match"].append(em)
            task_groups["entity"]["f1"].append(f1)
        hop = _hopwise_accuracy(pred_path, gt_path)
        if hop is not None:
            hop_list.append(hop)
        hall = _hallucination_rate(pred_path, node_id_set, edge_set)
        if hall is not None:
            hall_list.append(hall)
    by_task = {}
    for task_name, stats in task_groups.items():
        if stats["count"] == 0:
            continue
        by_task[task_name] = {
            "count": stats["count"],
            "exact_match": sum(stats["exact_match"]) / len(stats["exact_match"]) if stats["exact_match"] else 0.0,
            "f1": sum(stats["f1"]) / len(stats["f1"]) if stats["f1"] else 0.0,
        }
        if task_name == "topic":
            by_task[task_name]["keyword_recall"] = sum(stats["keyword_recall"]) / len(stats["keyword_recall"]) if stats["keyword_recall"] else 0.0
        if task_name == "path_witness":
            by_task[task_name]["answer_accuracy"] = sum(stats["answer_accuracy"]) / len(stats["answer_accuracy"]) if stats["answer_accuracy"] else 0.0
            by_task[task_name]["witness_validity"] = sum(stats["witness_validity"]) / len(stats["witness_validity"]) if stats["witness_validity"] else 0.0
            by_task[task_name]["joint_accuracy"] = sum(stats["joint_accuracy"]) / len(stats["joint_accuracy"]) if stats["joint_accuracy"] else 0.0
    return {
        "exact_match": sum(em_list) / len(em_list) if em_list else 0.0,
        "f1": sum(f1_list) / len(f1_list) if f1_list else 0.0,
        "hop_wise_accuracy": sum(hop_list) / len(hop_list) if hop_list else float("nan"),
        "hallucination_rate": sum(hall_list) / len(hall_list) if hall_list else float("nan"),
        "by_task": by_task,
    }


def load_grbench_metrics(path, dataset=None):
    return compute_grbench_metrics(pd.read_json(path, lines=True), dataset=dataset)


def get_accuracy_grbench(path, args=None, dataset=None, **kwargs):
    metrics = load_grbench_metrics(path, dataset=dataset)
    print(f"EM: {metrics['exact_match']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Hop-wise Accuracy: {metrics['hop_wise_accuracy']}")
    print(f"Hallucination Rate: {metrics['hallucination_rate']}")
    for task_name, task_metrics in metrics.get("by_task", {}).items():
        print(f"{task_name.title()} EM/F1: {task_metrics['exact_match']:.4f} / {task_metrics['f1']:.4f} (n={task_metrics['count']})")
        if task_name == "topic":
            print(f"Topic Keyword Recall: {task_metrics['keyword_recall']:.4f}")
    return metrics["exact_match"]

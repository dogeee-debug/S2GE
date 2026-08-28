"""Loss, prediction, and GRBench diagnostic evaluation helpers.

Prediction JSONL files are the stable boundary between generation and metric
code. Rows retain identifiers and metadata needed for exact-answer, format,
and output-state diagnostics.
"""

import json
import re

import torch

from s2ge.utils.metrics import evaluate_predictions


def _resolve_legal_integer_range(args=None):
    if args is None or not bool(getattr(args, "legal_integer_decode_enabled", False)):
        return None
    lower = int(getattr(args, "legal_integer_min", 1))
    upper = int(getattr(args, "legal_integer_max", lower))
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def move_batch_to_device(batch, device):
    """Move tensor-like batch values to ``device`` and preserve metadata."""
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            try:
                moved[key] = value.to(device)
                continue
            except TypeError:
                pass
        moved[key] = value
    return moved


def evaluate_loss(model, loader, device):
    """Return mean batch loss with gradients and training mode disabled."""
    total = 0.0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            loss_model = getattr(model, "module", model)
            if hasattr(loss_model, "compute_loss"):
                loss = loss_model.compute_loss(batch, include_contrastive=False)
            else:
                loss = model(batch)
            total += float(loss.item())
    return total / max(len(loader), 1)


def predict_to_file(model, loader, device, output_path):
    """Run native generation and write one JSON object per example."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with output_path.open("w", encoding="utf-8") as f:
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            with torch.no_grad():
                output = model.inference(batch)
            batch_size = len(output.get("id", []))
            for i in range(batch_size):
                record = {}
                for key, value in output.items():
                    if isinstance(value, (list, tuple)):
                        record[key] = value[i]
                    else:
                        record[key] = value
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_prediction_diagnostics(dataset_name, prediction_path, dataset=None, args=None, digit_min_length=6):
    """Compute optional output-state diagnostics for GRBench predictions."""
    em = evaluate_dataset(dataset_name, prediction_path, dataset=dataset, args=args)
    digit_pattern = re.compile(rf"\d{{{max(int(digit_min_length), 1)},}}")
    single_digit_pattern = re.compile(r"\d")
    legal_integer_range = _resolve_legal_integer_range(args)
    total = 0
    digit_long = 0
    single_digit = 0
    legal_integer = 0
    with prediction_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pred = str(row.get("pred", "")).strip()
            total += 1
            if digit_pattern.fullmatch(pred):
                digit_long += 1
            if single_digit_pattern.fullmatch(pred):
                single_digit += 1
            if legal_integer_range is not None and re.fullmatch(r"-?\d+", pred):
                value = int(pred)
                if legal_integer_range[0] <= value <= legal_integer_range[1]:
                    legal_integer += 1
    denom = total if total > 0 else 1
    return {
        "exact_match": float(em),
        "digit8_rate": digit_long / denom,
        "single_digit_rate": single_digit / denom,
        "legal_integer_rate": legal_integer / denom if legal_integer_range is not None else 0.0,
        "count": total,
    }


def evaluate_dataset(dataset_name, prediction_path, dataset=None, args=None):
    """Dispatch the dataset-specific metric implementation."""
    if dataset_name == "grbench":
        return evaluate_predictions(dataset_name, prediction_path, dataset=dataset, args=args)
    return evaluate_predictions(dataset_name, prediction_path)

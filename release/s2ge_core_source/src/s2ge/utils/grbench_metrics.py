"""
GRBENCH Comprehensive Evaluation Metrics
Including: Exact Match, Macro-F1, Micro-F1, Format Compliance, Hallucination Rate,
Token-level Perplexity, Dirichlet Energy, Target Node MAD, MoCo NCE Loss
"""
import json
import re

import pandas as pd
from s2ge.utils.metrics import load_grbench_metrics


def _check_format_compliance(pred):
    pred_str = str(pred)
    if pred_str.strip().startswith("{"):
        try:
            obj = json.loads(pred_str)
            return "answer" in obj or "path" in obj
        except Exception:
            pass
    return bool(re.search(r'["\']?(answer|path)["\']?\s*:', pred_str, re.IGNORECASE))


def evaluate_grbench_comprehensive(prediction_path, dataset=None, args=None):
    """
    Comprehensive GRBENCH evaluation with all metrics:
    - Exact Match
    - Macro-F1 / Micro-F1
    - Format Compliance Rate
    - Hallucination Rate
    - Token-level Perplexity (approximated via F1)
    - Dirichlet Energy
    - Target Node MAD
    - MoCo NCE Loss
    """
    metrics = load_grbench_metrics(prediction_path, dataset=dataset)
    em = metrics["exact_match"]
    f1 = metrics["f1"]
    hop_acc = metrics["hop_wise_accuracy"]
    hall_rate = metrics["hallucination_rate"]
    df = pd.read_json(prediction_path, lines=True)
    format_rate = (
        sum(1 for pred in df["pred"].tolist() if _check_format_compliance(pred)) / len(df)
        if len(df)
        else 0.0
    )
    
    # This evaluation currently exposes EM/F1/hop/hallucination only.
    # Avoid returning proxy values that look authoritative.
    macro_f1 = float("nan")
    micro_f1 = float("nan")
    
    # Print all metrics
    print("=" * 50)
    print("GRBENCH Comprehensive Evaluation Results")
    print("=" * 50)
    print(f"Exact Match (EM):           {em:.4f}")
    print(f"Token-level F1:             {f1:.4f}")
    print(f"Macro-F1:                   {macro_f1}")
    print(f"Micro-F1:                   {micro_f1}")
    print(f"Format Compliance Rate:      {format_rate:.4f}")
    print(f"Hallucination Rate:         {hall_rate:.4f}")
    print(f"Hop-wise Accuracy:          {hop_acc:.4f}")
    print("=" * 50)
    
    # Return dictionary with all metrics
    return {
        "exact_match": em,
        "f1": f1,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "format_compliance": format_rate,
        "hallucination_rate": hall_rate,
        "hop_wise_accuracy": hop_acc,
        "dirichlet_energy": float("nan"),
        "target_mad": float("nan"),
        "moco_nce_loss": float("nan"),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        results = evaluate_grbench_comprehensive(sys.argv[1])
        print(json.dumps(results, indent=2))

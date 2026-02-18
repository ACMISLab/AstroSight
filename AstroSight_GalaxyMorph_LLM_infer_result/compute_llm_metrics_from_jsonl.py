#!/usr/bin/env python3
"""Compute classification metrics (accuracy, macro-F1, weighted-F1)
from a jsonl file produced by LLM galaxy classification runs.

Each line in the jsonl is expected to be a JSON object with at least:
  - "response": model output text containing "option X is selected"
  - "labels"  : ground-truth text containing "option X is selected"

We parse the option letter A–H from both fields and then compute metrics
using sklearn.metrics.
"""

import argparse
import json
import re
from typing import List, Tuple, Dict, Any
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix


OPTION_PATTERN = re.compile(r"option\s*([A-H])\s*is\s*selected", re.IGNORECASE)


def extract_label(text: str) -> str:
    """Extract option letter A–H from a text like
    "Answer: ... option E is selected.".

    Returns the uppercase letter, or None if not found.
    """
    if not isinstance(text, str):
        return None
    match = OPTION_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).upper()


def load_labels_and_predictions(path: str) -> Tuple[List[str], List[str]]:
    """Load y_true and y_pred from a jsonl file.

    Lines where either ground-truth or prediction cannot be parsed
    are skipped.
    """
    y_true: List[str] = []
    y_pred: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            gt_text = data.get("labels")
            pred_text = data.get("response")

            true_label = extract_label(gt_text)
            pred_label = extract_label(pred_text)

            if true_label is None or pred_label is None:
                # Skip samples we cannot parse
                continue

            y_true.append(true_label)
            y_pred.append(pred_label)

    return y_true, y_pred


def compute_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    if not y_true or not y_pred:
        print("No valid samples found.")
        return {}

    labels = list("ABCDEFGH")

    accuracy = accuracy_score(y_true, y_pred)

    # macro / weighted precision, recall, F1 follow sklearn definitions
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", labels=labels, zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", labels=labels, zero_division=0
    )

    # per-class metrics and confusion matrix
    report_dict = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    print("=== Overall Metrics ===")
    print(f"Total valid samples: {len(y_true)}")
    print(f"Accuracy (%)           : {accuracy * 100:.2f}")

    print("\nMacro-averaged (%)")
    print(f"  Precision (%)        : {prec_macro * 100:.2f}")
    print(f"  Recall (%)           : {rec_macro * 100:.2f}")
    print(f"  F1-Score (%)         : {f1_macro * 100:.2f}")

    print("\nWeighted-averaged (%)")
    print(f"  Precision (%)        : {prec_weighted * 100:.2f}")
    print(f"  Recall (%)           : {rec_weighted * 100:.2f}")
    print(f"  F1-Score (%)         : {f1_weighted * 100:.2f}")

    print("\n=== Per-class report (for reference) ===")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    metrics: Dict[str, Any] = {
        "num_samples": len(y_true),
        "labels": labels,
        "overall": {
            "accuracy": accuracy,
        },
        "macro": {
            "precision": prec_macro,
            "recall": rec_macro,
            "f1": f1_macro,
        },
        "weighted": {
            "precision": prec_weighted,
            "recall": rec_weighted,
            "f1": f1_weighted,
        },
        "per_class": report_dict,
        "confusion_matrix": cm.tolist(),
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute macro-F1 and weighted-F1 from LLM jsonl results.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="20250512-081045.jsonl",
        help="Path to jsonl file (default: 20250512-081045.jsonl in current directory)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    y_true, y_pred = load_labels_and_predictions(str(input_path))
    metrics = compute_metrics(y_true, y_pred)

    if metrics:
        output_path = input_path.with_suffix(".metrics.json")
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to: {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

import json
import argparse
import re
from typing import List, Dict

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


LABEL_MAP: Dict[str, str] = {
    "A": "round elliptical",
    "B": "in-between elliptical",
    "C": "cigar-shaped elliptical",
    "D": "edge-on",
    "E": "Barred spirals",
    "F": "Unbarred spirals",
    "G": "Irregular",
    "H": "merger",
}

LABELS: List[str] = list(LABEL_MAP.keys())


def load_results(result_path: str) -> tuple[List[str], List[str]]:
    """从结果 JSON 中提取真实标签和预测标签"""
    with open(result_path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)

        detailed = data.get("detailed_results", [])
        y_true: List[str] = []
        y_pred: List[str] = []

        for item in detailed:
            true_label = item.get("true_label")
            pred_label = item.get("predicted_label")
            if true_label is None or pred_label is None:
                continue
            y_true.append(str(true_label).strip())
            y_pred.append(str(pred_label).strip())

        return y_true, y_pred
    except json.JSONDecodeError:
        true_labels = re.findall(r'"true_label"\s*:\s*"([A-H])"', content)
        pred_labels = re.findall(r'"predicted_label"\s*:\s*"([A-H])"', content)

        n = min(len(true_labels), len(pred_labels))
        y_true = [lbl.strip() for lbl in true_labels[:n]]
        y_pred = [lbl.strip() for lbl in pred_labels[:n]]

        return y_true, y_pred


def calculate_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    """计算总体、Macro、Weighted 以及每类的指标"""
    if not y_true or not y_pred:
        raise ValueError("Empty y_true or y_pred; check result file.")

    # 总体准确率
    accuracy = accuracy_score(y_true, y_pred)

    # Weighted 平均
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    # Macro 平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # 每一类指标（按固定标签顺序 A-H）
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)

    metrics: Dict = {
        "overall": {
            "total_samples": len(y_true),
            "accuracy": round(accuracy * 100, 4),
        },
        "macro": {
            "precision": round(precision_macro * 100, 4),
            "recall": round(recall_macro * 100, 4),
            "f1_score": round(f1_macro * 100, 4),
        },
        "weighted": {
            "precision": round(precision_weighted * 100, 4),
            "recall": round(recall_weighted * 100, 4),
            "f1_score": round(f1_weighted * 100, 4),
        },
        "per_class": {},
        "confusion_matrix": cm.tolist(),
        "labels": LABELS,
    }

    for i, label in enumerate(LABELS):
        metrics["per_class"][label] = {
            "category": LABEL_MAP[label],
            "precision": round(precision_per[i] * 100, 4),
            "recall": round(recall_per[i] * 100, 4),
            "f1_score": round(f1_per[i] * 100, 4),
            "support": int(support_per[i]),
        }

    return metrics


def print_metrics(metrics: Dict) -> None:
    """在终端打印总体、Macro / Weighted 以及每类指标"""
    overall = metrics["overall"]
    macro = metrics["macro"]
    weighted = metrics["weighted"]

    print("=" * 60)
    print("Re-evaluated Galaxy Classification Metrics")
    print("=" * 60)
    print(f"Total samples : {overall['total_samples']}")
    print(f"Accuracy      : {overall['accuracy']:.2f}%")
    print()
    print("Macro average:")
    print(f"  Precision   : {macro['precision']:.2f}%")
    print(f"  Recall      : {macro['recall']:.2f}%")
    print(f"  F1-score    : {macro['f1_score']:.2f}%")
    print()
    print("Weighted average:")
    print(f"  Precision   : {weighted['precision']:.2f}%")
    print(f"  Recall      : {weighted['recall']:.2f}%")
    print(f"  F1-score    : {weighted['f1_score']:.2f}%")
    print()
    print("Per-class metrics:")
    print(f"{'Label':<5} {'Category':<24} {'Prec(%)':>9} {'Rec(%)':>9} {'F1(%)':>9} {'Support':>9}")
    print("-" * 70)
    for label in metrics["labels"]:
        data = metrics["per_class"][label]
        print(
            f"{label:<5} {data['category']:<24} "
            f"{data['precision']:>9.2f} {data['recall']:>9.2f} {data['f1_score']:>9.2f} {data['support']:>9}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate galaxy classification metrics from result JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="gpt5_galaxy_classification_results.json",
        help="Path to result JSON file.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional output path to save computed metrics as JSON.",
    )
    args = parser.parse_args()

    y_true, y_pred = load_results(args.input)
    metrics = calculate_metrics(y_true, y_pred)
    print_metrics(metrics)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to {args.save_json}")


if __name__ == "__main__":
    main()

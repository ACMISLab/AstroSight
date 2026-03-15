#!/usr/bin/env python3
"""
解析大模型推理结果并生成详细的分类报告
支持多个模型结果的对比分析
"""

import os
import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path

# 类别映射
OPTION_TO_LABEL = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3,
    'E': 4, 'F': 5, 'G': 6, 'H': 7
}

LABEL_TO_OPTION = {v: k for k, v in OPTION_TO_LABEL.items()}

CLASS_NAMES = {
    0: "Round Elliptical (A)",
    1: "In-between Elliptical (B)",
    2: "Cigar-shaped Elliptical (C)",
    3: "Edge-on Spiral (D)",
    4: "Barred Spiral (E)",
    5: "Unbarred Spiral (F)",
    6: "Irregular (G)",
    7: "Merger (H)"
}


def extract_option(text):
    """
    从文本中提取选项字母
    支持多种格式：
    - "A:round elliptical"
    - "option A"
    - "A"
    """
    text = text.strip()
    
    # 格式1: "A:round elliptical"
    if ':' in text:
        option = text.split(':')[0].strip()
        if option in OPTION_TO_LABEL:
            return option
    
    # 格式2: "option A is selected"
    if 'option' in text.lower():
        parts = text.split('option')
        if len(parts) > 1:
            after_option = parts[1].strip()
            for char in after_option:
                if char in OPTION_TO_LABEL:
                    return char
    
    # 格式3: 直接是字母
    for char in text:
        if char in OPTION_TO_LABEL:
            return char
    
    return None


def parse_inference_results(jsonl_path):
    """
    解析推理结果JSONL文件
    
    Returns:
        predictions: list of predicted labels (0-7)
        ground_truths: list of ground truth labels (0-7)
        details: list of dicts with detailed info
    """
    predictions = []
    ground_truths = []
    details = []
    
    print(f"Parsing: {jsonl_path}")
    
    with open(jsonl_path, 'r') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines):
        try:
            entry = json.loads(line)
            
            # 提取预测结果
            response = entry.get('response', '')
            pred_option = extract_option(response)
            
            # 提取真实标签
            labels = entry.get('labels', '')
            gt_option = extract_option(labels)
            
            # 转换为数字标签
            pred_label = OPTION_TO_LABEL.get(pred_option, -1) if pred_option else -1
            gt_label = OPTION_TO_LABEL.get(gt_option, -1) if gt_option else -1
            
            predictions.append(pred_label)
            ground_truths.append(gt_label)
            
            # 保存详细信息
            image_path = entry.get('images', [{}])[0].get('path', '') if entry.get('images') else ''
            details.append({
                'index': idx,
                'image': os.path.basename(image_path) if image_path else f'sample_{idx}',
                'predicted': pred_option,
                'predicted_label': pred_label,
                'ground_truth': gt_option,
                'ground_truth_label': gt_label,
                'correct': pred_label == gt_label,
                'response': response,
                'labels': labels
            })
            
        except Exception as e:
            print(f"Warning: Error parsing line {idx}: {e}")
            predictions.append(-1)
            ground_truths.append(-1)
            details.append({
                'index': idx,
                'error': str(e)
            })
    
    return predictions, ground_truths, details


def compute_metrics(predictions, ground_truths):
    """
    计算分类指标
    
    Returns:
        dict with metrics
    """
    # 过滤掉无效预测
    valid_indices = [i for i in range(len(predictions)) 
                     if predictions[i] >= 0 and ground_truths[i] >= 0]
    
    preds = [predictions[i] for i in valid_indices]
    gts = [ground_truths[i] for i in valid_indices]
    
    if len(preds) == 0:
        return {'error': 'No valid predictions'}
    
    # 总体准确率
    correct = sum(1 for p, g in zip(preds, gts) if p == g)
    accuracy = correct / len(preds) * 100
    
    # 每类别指标
    per_class_metrics = {}
    
    for label in range(8):
        # TP, FP, FN, TN
        tp = sum(1 for p, g in zip(preds, gts) if p == label and g == label)
        fp = sum(1 for p, g in zip(preds, gts) if p == label and g != label)
        fn = sum(1 for p, g in zip(preds, gts) if p != label and g == label)
        tn = sum(1 for p, g in zip(preds, gts) if p != label and g != label)
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Support
        support = sum(1 for g in gts if g == label)
        
        per_class_metrics[label] = {
            'precision': precision * 100,
            'recall': recall * 100,
            'f1': f1 * 100,
            'support': support,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    # Macro平均
    macro_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
    macro_recall = np.mean([m['recall'] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m['f1'] for m in per_class_metrics.values()])
    
    # 混淆矩阵
    confusion_matrix = np.zeros((8, 8), dtype=int)
    for p, g in zip(preds, gts):
        confusion_matrix[g][p] += 1
    
    return {
        'total_samples': len(predictions),
        'valid_samples': len(preds),
        'invalid_samples': len(predictions) - len(preds),
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': per_class_metrics,
        'confusion_matrix': confusion_matrix.tolist()
    }


def print_detailed_report(jsonl_path, output_dir=None):
    """
    生成并打印详细报告
    """
    # 解析结果
    predictions, ground_truths, details = parse_inference_results(jsonl_path)
    
    # 计算指标
    metrics = compute_metrics(predictions, ground_truths)
    
    # 模型名称
    model_name = os.path.basename(jsonl_path).replace('.jsonl', '')
    
    print("\n" + "="*80)
    print(f"Inference Results Analysis: {model_name}")
    print("="*80)
    
    # 基本信息
    print(f"\n[1] Basic Information")
    print("-"*80)
    print(f"Total samples:   {metrics['total_samples']}")
    print(f"Valid samples:   {metrics['valid_samples']}")
    print(f"Invalid samples: {metrics['invalid_samples']}")
    
    # 整体性能
    print(f"\n[2] Overall Performance")
    print("-"*80)
    print(f"Accuracy:        {metrics['accuracy']:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']:.2f}%")
    print(f"Macro Recall:    {metrics['macro_recall']:.2f}%")
    print(f"Macro F1-Score:  {metrics['macro_f1']:.2f}%")
    
    # 每类别性能
    print(f"\n[3] Per-Class Performance")
    print("-"*80)
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    for label in range(8):
        m = metrics['per_class'][label]
        class_name = CLASS_NAMES[label]
        print(f"{class_name:<30} {m['precision']:>10.2f}%  {m['recall']:>10.2f}%  "
              f"{m['f1']:>10.2f}%  {m['support']:>8d}")
    
    # 混淆矩阵
    print(f"\n[4] Confusion Matrix")
    print("-"*80)
    print("Rows: Ground Truth, Columns: Predicted")
    print()
    
    # 表头
    header = "GT\\Pred |"
    for i in range(8):
        header += f"  {LABEL_TO_OPTION[i]}  |"
    print(header)
    print("-" * len(header))
    
    # 矩阵内容
    cm = np.array(metrics['confusion_matrix'])
    for i in range(8):
        row = f"   {LABEL_TO_OPTION[i]}    |"
        for j in range(8):
            row += f" {cm[i][j]:4d} |"
        print(row)
    
    # 预测分布
    print(f"\n[5] Prediction Distribution")
    print("-"*80)
    pred_counts = Counter([p for p in predictions if p >= 0])
    gt_counts = Counter([g for g in ground_truths if g >= 0])
    
    print(f"{'Class':<30} {'Predicted':<12} {'Ground Truth':<12} {'Difference':<12}")
    print("-"*80)
    for label in range(8):
        pred_count = pred_counts.get(label, 0)
        gt_count = gt_counts.get(label, 0)
        diff = pred_count - gt_count
        class_name = CLASS_NAMES[label]
        print(f"{class_name:<30} {pred_count:>10d}  {gt_count:>12d}  {diff:>+11d}")
    
    # 错误分析
    print(f"\n[6] Error Analysis")
    print("-"*80)
    errors = [d for d in details if not d.get('correct', False) and 'error' not in d]
    print(f"Total errors: {len(errors)}")
    
    if len(errors) > 0:
        # 最常见的错误类型
        error_types = Counter()
        for err in errors:
            gt = err['ground_truth']
            pred = err['predicted']
            if gt and pred:
                error_types[(gt, pred)] += 1
        
        print(f"\nTop 10 Most Common Errors:")
        for (gt, pred), count in error_types.most_common(10):
            gt_name = CLASS_NAMES[OPTION_TO_LABEL[gt]]
            pred_name = CLASS_NAMES[OPTION_TO_LABEL[pred]]
            print(f"  {gt} → {pred}: {count:4d} times  ({gt_name} misclassified as {pred_name})")
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存JSON格式的完整结果
        output_file = os.path.join(output_dir, f"{model_name}_analysis.json")
        with open(output_file, 'w') as f:
            json.dump({
                'model': model_name,
                'metrics': metrics,
                'errors': errors[:100]  # 保存前100个错误样本
            }, f, indent=2)
        
        print(f"\n[7] Results saved to: {output_file}")
    
    print("\n" + "="*80)
    
    return metrics, details


def compare_multiple_models(jsonl_paths, output_dir=None):
    """
    对比多个模型的结果
    """
    all_results = {}
    
    for path in jsonl_paths:
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping...")
            continue
        
        model_name = os.path.basename(path).replace('.jsonl', '')
        predictions, ground_truths, details = parse_inference_results(path)
        metrics = compute_metrics(predictions, ground_truths)
        
        all_results[model_name] = {
            'metrics': metrics,
            'predictions': predictions,
            'ground_truths': ground_truths
        }
    
    # 打印对比表
    print("\n" + "="*100)
    print("Model Comparison")
    print("="*100)
    
    print(f"\n{'Model':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*100)
    
    for model_name, result in all_results.items():
        m = result['metrics']
        print(f"{model_name:<30} {m['accuracy']:>10.2f}%  {m['macro_precision']:>10.2f}%  "
              f"{m['macro_recall']:>10.2f}%  {m['macro_f1']:>10.2f}%")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        comparison_file = os.path.join(output_dir, "model_comparison.json")
        
        # 只保存metrics，不保存predictions
        comparison_data = {
            model: {'metrics': result['metrics']}
            for model, result in all_results.items()
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\nComparison saved to: {comparison_file}")
    
    return all_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze LLM inference results')
    parser.add_argument('jsonl_file', type=str, help='Path to inference results JSONL file')
    parser.add_argument('--output-dir', type=str, default='./analysis_results',
                        help='Output directory for analysis results')
    parser.add_argument('--compare', nargs='+', help='Additional JSONL files to compare')
    
    args = parser.parse_args()
    
    # 分析主文件
    metrics, details = print_detailed_report(args.jsonl_file, args.output_dir)
    
    # 如果有多个文件，进行对比
    if args.compare:
        all_files = [args.jsonl_file] + args.compare
        compare_multiple_models(all_files, args.output_dir)


if __name__ == "__main__":
    main()

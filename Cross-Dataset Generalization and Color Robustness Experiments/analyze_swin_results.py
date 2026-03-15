#!/usr/bin/env python3
"""
分析和对比Swin Transformer在彩色和灰度DECaLS数据集上的结果
"""

import json
import argparse
import numpy as np

CLASS_NAMES = [
    'Barred Spirals',
    'Cigar-shaped Elliptical',
    'Edge-on',
    'In-between Elliptical',
    'Irregular',
    'Merger',
    'Round Elliptical',
    'Unbarred Spirals'
]


def load_results(json_path):
    """加载结果JSON文件"""
    with open(json_path, 'r') as f:
        return json.load(f)


def print_comparison(color_results, gray_results):
    """打印彩色和灰度结果对比"""
    
    print("\n" + "="*90)
    print("Swin Transformer: Color vs Grayscale Comparison on DECaLS")
    print("="*90)
    
    # 整体准确率对比
    print("\n[1] Overall Accuracy")
    print("-"*90)
    color_acc = color_results['accuracy']
    gray_acc = gray_results['accuracy']
    diff = color_acc - gray_acc
    
    print(f"Color Images:     {color_acc:6.2f}%")
    print(f"Grayscale Images: {gray_acc:6.2f}%")
    print(f"Difference:       {diff:+6.2f}% {'(Color better)' if diff > 0 else '(Grayscale better)'}")
    
    # 每类别对比
    print("\n[2] Per-Class Performance Comparison")
    print("-"*90)
    print(f"{'Class':<30} {'Color Acc':<12} {'Gray Acc':<12} {'Difference':<12}")
    print("-"*90)
    
    color_report = color_results['classification_report']
    gray_report = gray_results['classification_report']
    
    for i, class_name in enumerate(CLASS_NAMES):
        # 使用原始类别名称（小写+下划线）
        original_name = class_name.lower().replace(' ', '_').replace('-', '_')
        
        if original_name in color_report and original_name in gray_report:
            color_f1 = color_report[original_name]['f1-score'] * 100
            gray_f1 = gray_report[original_name]['f1-score'] * 100
            diff = color_f1 - gray_f1
            
            print(f"{class_name:<30} {color_f1:>10.2f}%  {gray_f1:>10.2f}%  {diff:>+10.2f}%")
    
    # Macro平均对比
    print("-"*90)
    color_macro = color_report['macro avg']
    gray_macro = gray_report['macro avg']
    
    print(f"\n[3] Macro Average Metrics")
    print("-"*90)
    print(f"{'Metric':<20} {'Color':<15} {'Grayscale':<15} {'Difference':<15}")
    print("-"*90)
    
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        color_val = color_macro[metric] * 100
        gray_val = gray_macro[metric] * 100
        diff = color_val - gray_val
        print(f"{metric.capitalize():<20} {color_val:>13.2f}%  {gray_val:>13.2f}%  {diff:>+13.2f}%")
    
    # 分析
    print("\n[4] Analysis")
    print("-"*90)
    
    if diff > 0:
        print(f"✓ Color images perform better by {diff:.2f}%")
        print(f"  This indicates that color information is beneficial for galaxy classification.")
    elif diff < 0:
        print(f"✗ Grayscale images perform better by {-diff:.2f}%")
        print(f"  This is unexpected and may indicate an issue.")
    else:
        print(f"= Color and grayscale perform equally")
    
    # 颜色贡献度
    color_contribution = (diff / color_acc) * 100 if color_acc > 0 else 0
    print(f"\nColor contribution: {color_contribution:.2f}% of total accuracy")
    
    print("\n" + "="*90)


def save_comparison(color_results, gray_results, output_path):
    """保存对比结果"""
    comparison = {
        'color': {
            'accuracy': color_results['accuracy'],
            'macro_avg': color_results['classification_report']['macro avg']
        },
        'grayscale': {
            'accuracy': gray_results['accuracy'],
            'macro_avg': gray_results['classification_report']['macro avg']
        },
        'difference': {
            'accuracy': color_results['accuracy'] - gray_results['accuracy'],
            'color_contribution_percent': (
                (color_results['accuracy'] - gray_results['accuracy']) / 
                color_results['accuracy'] * 100
            ) if color_results['accuracy'] > 0 else 0
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Swin Transformer results')
    parser.add_argument('--color', type=str, required=True,
                        help='Path to color results JSON')
    parser.add_argument('--grayscale', type=str, required=True,
                        help='Path to grayscale results JSON')
    parser.add_argument('--output', type=str, 
                        default='swin_decals_results/comparison.json',
                        help='Output comparison JSON file')
    
    args = parser.parse_args()
    
    # 加载结果
    print("Loading results...")
    color_results = load_results(args.color)
    gray_results = load_results(args.grayscale)
    
    # 打印对比
    print_comparison(color_results, gray_results)
    
    # 保存对比
    save_comparison(color_results, gray_results, args.output)
    
    print("\n✓ Analysis completed!")


if __name__ == "__main__":
    main()

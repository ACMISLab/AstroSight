#!/usr/bin/env python3
"""
AstroSight Galaxy Morphological Attribute Regression Evaluation Script

This script evaluates the attribute regression performance of AstroSight model
by comparing predicted attributes with ground truth labels.

Input: JSONL file with 'response' (predictions) and 'labels' (ground truth)
Output: Complete evaluation metrics (MAE, MSE, R²) for overall and per-attribute

Usage:
python evaluate_astrosight_attributes.py --input InternVL2.5_38B_arttibute_regression_lora.jsonl
"""

import argparse
import json
import re
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def parse_attributes_from_text(text):
    """
    从文本中解析17个属性值
    
    Args:
        text: 包含属性的文本字符串
        
    Returns:
        dict: 属性名称到值的映射
    """
    # 定义17个属性的正则表达式模式
    patterns = {
        'f_smooth': r'f_smooth=([0-9.]+)',
        'f_features/disk': r'f_features/disk=([0-9.]+)', 
        'f_edge-on/yes': r'f_edge-on/yes=([0-9.]+)',
        'f_edge-on/no': r'f_edge-on/no=([0-9.]+)',
        'f_bar/yes': r'f_bar/yes=([0-9.]+)',
        'f_bar/no': r'f_bar/no=([0-9.]+)',
        'f_spiral/yes': r'f_spiral/yes=([0-9.]+)', 
        'f_odd/yes': r'f_odd/yes=([0-9.]+)',
        'f_odd/no': r'f_odd/no=([0-9.]+)',
        'f_completelyround': r'f_completelyround=([0-9.]+)',
        'f_in-between': r'f_in-between=([0-9.]+)',
        'f_cigar-shaped': r'f_cigar-shaped=([0-9.]+)',
        'f_disturbed': r'f_disturbed=([0-9.]+)',
        'f_irregular': r'f_irregular=([0-9.]+)',
        'f_other': r'f_other=([0-9.]+)',
        'f_merger': r'f_merger=([0-9.]+)',
        'f_dustlane': r'f_dustlane=([0-9.]+)'
    }
    
    attributes = {}
    for attr_name, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            attributes[attr_name] = float(match.group(1))
        else:
            attributes[attr_name] = 0.0  # 默认值
    
    return attributes


def load_predictions_and_labels(jsonl_file):
    """
    从JSONL文件加载预测值和真实标签
    
    Args:
        jsonl_file: JSONL文件路径
        
    Returns:
        tuple: (predictions_dict, labels_dict, attribute_names)
    """
    print(f"📁 加载文件: {jsonl_file}")
    
    # 17个属性名称（按字母顺序）
    attribute_names = [
        'f_bar/no', 'f_bar/yes', 'f_cigar-shaped', 'f_completelyround',
        'f_disturbed', 'f_dustlane', 'f_edge-on/no', 'f_edge-on/yes',
        'f_features/disk', 'f_in-between', 'f_irregular', 'f_merger',
        'f_odd/no', 'f_odd/yes', 'f_other', 'f_smooth', 'f_spiral/yes'
    ]
    
    predictions_list = []
    labels_list = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
                
            try:
                data = json.loads(line.strip())
                
                # 解析预测值
                response_text = data.get('response', '')
                pred_attrs = parse_attributes_from_text(response_text)
                
                # 解析真实标签
                label_text = data.get('labels', '')
                label_attrs = parse_attributes_from_text(label_text)
                
                # 按照固定顺序提取属性值
                pred_values = [pred_attrs.get(attr, 0.0) for attr in attribute_names]
                label_values = [label_attrs.get(attr, 0.0) for attr in attribute_names]
                
                predictions_list.append(pred_values)
                labels_list.append(label_values)
                
            except json.JSONDecodeError as e:
                print(f"⚠️ 警告 (行{line_num}): JSON解析失败 - {e}")
                continue
            except Exception as e:
                print(f"⚠️ 警告 (行{line_num}): 处理失败 - {e}")
                continue
    
    predictions = np.array(predictions_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.float32)
    
    print(f"✅ 成功加载 {len(predictions)} 个样本")
    print(f"   预测值形状: {predictions.shape}")
    print(f"   标签形状: {labels.shape}")
    print(f"   属性数量: {len(attribute_names)}")
    
    return predictions, labels, attribute_names


def calculate_overall_metrics(predictions, labels):
    """
    计算整体评估指标（多种计算方式）
    
    Args:
        predictions: 预测值数组 (N, 17)
        labels: 真实标签数组 (N, 17)
        
    Returns:
        dict: 整体指标（包含多种R²计算方式）
    """
    # 展平所有数据点计算（样本×属性）
    pred_flat = predictions.flatten()
    label_flat = labels.flatten()
    
    # 基础指标
    mae = mean_absolute_error(label_flat, pred_flat)
    mse = mean_squared_error(label_flat, pred_flat)
    rmse = np.sqrt(mse)
    
    # 方式1: 整体R² (Overall R²) - 展平后计算
    r2_overall = r2_score(label_flat, pred_flat)
    
    # 方式2: 属性平均R² (Attribute-Average R²) - 每个属性单独计算后平均
    r2_per_attr = []
    for i in range(predictions.shape[1]):
        r2_attr = r2_score(labels[:, i], predictions[:, i])
        r2_per_attr.append(r2_attr)
    r2_attribute_average = np.mean(r2_per_attr)
    
    # 方式3: 样本平均R² (Sample-Average R²) - 每个样本单独计算后平均
    r2_per_sample = []
    for i in range(predictions.shape[0]):
        # 避免单样本方差为0的情况
        if np.std(labels[i, :]) > 1e-10:
            r2_sample = r2_score(labels[i, :], predictions[i, :])
            r2_per_sample.append(r2_sample)
    r2_sample_average = np.mean(r2_per_sample) if r2_per_sample else 0.0
    
    # 方式4: 加权R² (Variance-Weighted R²) - 按属性方差加权
    attr_variances = np.var(labels, axis=0)
    total_variance = np.sum(attr_variances)
    r2_weighted = 0.0
    for i in range(predictions.shape[1]):
        if total_variance > 0:
            weight = attr_variances[i] / total_variance
            r2_attr = r2_score(labels[:, i], predictions[:, i])
            r2_weighted += weight * r2_attr
    
    # 手动计算R² (验证sklearn结果)
    ss_res = np.sum((label_flat - pred_flat) ** 2)
    ss_tot = np.sum((label_flat - np.mean(label_flat)) ** 2)
    r2_manual = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # 相关系数
    correlation = np.corrcoef(label_flat, pred_flat)[0, 1]
    
    # 误差统计
    errors = pred_flat - label_flat
    abs_errors = np.abs(errors)
    
    # 百分位数误差
    percentiles = [50, 75, 90, 95, 99]
    percentile_errors = {f'P{p}': float(np.percentile(abs_errors, p)) for p in percentiles}
    
    # 最大误差
    max_error = float(np.max(abs_errors))
    
    # 误差在不同阈值内的比例
    error_thresholds = [0.05, 0.1, 0.15, 0.2]
    error_within_threshold = {
        f'within_{t}': float(np.mean(abs_errors <= t) * 100)
        for t in error_thresholds
    }
    
    return {
        # 基础指标
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        
        # 多种R²计算方式
        'R2_overall': float(r2_overall),  # 主要指标：整体R²
        'R2_attribute_average': float(r2_attribute_average),  # 属性平均R²
        'R2_sample_average': float(r2_sample_average),  # 样本平均R²
        'R2_variance_weighted': float(r2_weighted),  # 方差加权R²
        'R2_manual_verification': float(r2_manual),  # 手动计算验证
        
        # 相关性
        'pearson_correlation': float(correlation),
        
        # 误差分布统计
        'max_absolute_error': max_error,
        'median_absolute_error': float(np.median(abs_errors)),
        'std_absolute_error': float(np.std(abs_errors)),
        
        # 百分位数误差
        'percentile_errors': percentile_errors,
        
        # 误差阈值统计（百分比）
        'error_within_threshold_percent': error_within_threshold,
        
        # 数据统计
        'total_predictions': int(len(pred_flat)),
        'num_samples': int(predictions.shape[0]),
        'num_attributes': int(predictions.shape[1]),
        
        # 预测值统计
        'prediction_mean': float(np.mean(pred_flat)),
        'prediction_std': float(np.std(pred_flat)),
        'prediction_min': float(np.min(pred_flat)),
        'prediction_max': float(np.max(pred_flat)),
        
        # 真实值统计
        'label_mean': float(np.mean(label_flat)),
        'label_std': float(np.std(label_flat)),
        'label_min': float(np.min(label_flat)),
        'label_max': float(np.max(label_flat))
    }


def calculate_per_attribute_metrics(predictions, labels, attribute_names):
    """
    计算每个属性的评估指标
    
    Args:
        predictions: 预测值数组 (N, 17)
        labels: 真实标签数组 (N, 17)
        attribute_names: 属性名称列表
        
    Returns:
        dict: 每个属性的指标
    """
    per_attr_metrics = {}
    
    for i, attr_name in enumerate(attribute_names):
        pred_attr = predictions[:, i]
        label_attr = labels[:, i]
        
        mae = mean_absolute_error(label_attr, pred_attr)
        mse = mean_squared_error(label_attr, pred_attr)
        r2 = r2_score(label_attr, pred_attr)
        rmse = np.sqrt(mse)
        
        # 计算统计信息
        mean_pred = np.mean(pred_attr)
        mean_label = np.mean(label_attr)
        std_pred = np.std(pred_attr)
        std_label = np.std(label_attr)
        
        per_attr_metrics[attr_name] = {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'R2': float(r2),
            'mean_prediction': float(mean_pred),
            'mean_label': float(mean_label),
            'std_prediction': float(std_pred),
            'std_label': float(std_label)
        }
    
    return per_attr_metrics


def print_evaluation_results(overall_metrics, per_attr_metrics, attribute_names):
    """打印评估结果"""
    print("\n" + "="*80)
    print("🎯 AstroSight 星系属性回归评估结果")
    print("="*80)
    
    # 数据统计
    print("\n📈 数据统计:")
    print(f"   样本数量: {overall_metrics['num_samples']:,}")
    print(f"   属性数量: {overall_metrics['num_attributes']}")
    print(f"   总数据点: {overall_metrics['total_predictions']:,}")
    
    # 基础指标
    print("\n📊 基础性能指标:")
    print(f"   MAE (Mean Absolute Error):      {overall_metrics['MAE']:.6f}")
    print(f"   MSE (Mean Squared Error):       {overall_metrics['MSE']:.6f}")
    print(f"   RMSE (Root Mean Squared Error): {overall_metrics['RMSE']:.6f}")
    print(f"   中位数绝对误差 (Median AE):      {overall_metrics['median_absolute_error']:.6f}")
    print(f"   最大绝对误差 (Max AE):          {overall_metrics['max_absolute_error']:.6f}")
    print(f"   误差标准差 (Std AE):            {overall_metrics['std_absolute_error']:.6f}")
    
    # 多种R²计算方式
    print("\n🎯 R² 指标 (多种计算方式):")
    print("   " + "-"*76)
    print(f"   1️⃣  整体R² (Overall R²):              {overall_metrics['R2_overall']:.6f}")
    print("       └─ 计算方式: 将所有样本×属性展平后计算")
    print(f"       └─ 公式: R² = 1 - SSE/SST (N×K={overall_metrics['total_predictions']:,}个数据点)")
    print("")
    print(f"   2️⃣  属性平均R² (Attribute-Avg R²):     {overall_metrics['R2_attribute_average']:.6f}")
    print("       └─ 计算方式: 每个属性单独计算R²后取平均")
    print(f"       └─ 公式: R² = mean(R²₁, R²₂, ..., R²₁₇)")
    print("")
    print(f"   3️⃣  样本平均R² (Sample-Avg R²):       {overall_metrics['R2_sample_average']:.6f}")
    print("       └─ 计算方式: 每个样本单独计算R²后取平均")
    print(f"       └─ 公式: R² = mean(R²_sample₁, R²_sample₂, ...)")
    print("")
    print(f"   4️⃣  方差加权R² (Variance-Weighted R²): {overall_metrics['R2_variance_weighted']:.6f}")
    print("       └─ 计算方式: 按每个属性的方差加权平均")
    print("       └─ 公式: R² = Σ(wᵢ × R²ᵢ), wᵢ = Var(attrᵢ) / Σ Var")
    print("")
    print(f"   ✅  手动验证R² (Manual Verification):  {overall_metrics['R2_manual_verification']:.6f}")
    print("       └─ 与整体R²一致，验证计算正确性")
    print("   " + "-"*76)
    
    # 相关性
    print("\n🔗 相关性分析:")
    print(f"   Pearson相关系数: {overall_metrics['pearson_correlation']:.6f}")
    print(f"   R² vs 相关系数²: {overall_metrics['R2_overall']:.6f} vs {overall_metrics['pearson_correlation']**2:.6f}")
    
    # 误差百分位数
    print("\n📊 误差分布 (百分位数):")
    for key, value in overall_metrics['percentile_errors'].items():
        percentile = key[1:]  # 去掉'P'
        print(f"   {percentile}th percentile: {value:.6f}")
    
    # 误差阈值统计
    print("\n✅ 误差在阈值内的样本比例:")
    for key, value in overall_metrics['error_within_threshold_percent'].items():
        threshold = key.replace('within_', '')
        print(f"   误差 ≤ {threshold}: {value:.2f}%")
    
    # 预测值和真实值统计
    print("\n📉 预测值统计:")
    print(f"   均值: {overall_metrics['prediction_mean']:.6f}")
    print(f"   标准差: {overall_metrics['prediction_std']:.6f}")
    print(f"   范围: [{overall_metrics['prediction_min']:.6f}, {overall_metrics['prediction_max']:.6f}]")
    
    print("\n� 真实值统计:")
    print(f"   均值: {overall_metrics['label_mean']:.6f}")
    print(f"   标准差: {overall_metrics['label_std']:.6f}")
    print(f"   范围: [{overall_metrics['label_min']:.6f}, {overall_metrics['label_max']:.6f}]")
    
    print("\n�📋 各属性详细指标:")
    print("-"*80)
    print(f"{'Attribute':<25} {'MAE':<10} {'MSE':<10} {'RMSE':<10} {'R²':<10}")
    print("-"*80)
    
    for attr_name in attribute_names:
        metrics = per_attr_metrics[attr_name]
        print(f"{attr_name:<25} {metrics['MAE']:<10.6f} {metrics['MSE']:<10.6f} "
              f"{metrics['RMSE']:<10.6f} {metrics['R2']:<10.6f}")
    
    print("-"*80)
    
    # 找出表现最好和最差的属性
    sorted_by_mae = sorted(per_attr_metrics.items(), key=lambda x: x[1]['MAE'])
    best_attrs = sorted_by_mae[:3]
    worst_attrs = sorted_by_mae[-3:]
    
    print("\n🏆 表现最好的3个属性 (按MAE):")
    for attr_name, metrics in best_attrs:
        print(f"   {attr_name:<25} MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f}")
    
    print("\n⚠️  表现最差的3个属性 (按MAE):")
    for attr_name, metrics in worst_attrs:
        print(f"   {attr_name:<25} MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f}")


def save_results(overall_metrics, per_attr_metrics, output_file):
    """保存评估结果到JSON文件"""
    results = {
        'model_info': {
            'name': 'AstroSight (InternVL2.5-38B)',
            'task': 'Galaxy Morphological Attribute Regression',
            'num_attributes': 17,
            'model_type': 'Vision-Language Model (VLM)',
            'training_method': 'LoRA Fine-tuning'
        },
        'overall_metrics': overall_metrics,
        'per_attribute_metrics': per_attr_metrics,
        'r2_calculation_methods': {
            'description': 'Multiple R² calculation methods for comprehensive evaluation',
            'methods': {
                'overall_r2': {
                    'value': overall_metrics['R2_overall'],
                    'method': 'Flatten all samples×attributes and compute R²',
                    'formula': 'R² = 1 - SSE/SST where all N×K points are used',
                    'use_case': 'Standard ML evaluation, reflects overall model performance'
                },
                'attribute_average_r2': {
                    'value': overall_metrics['R2_attribute_average'],
                    'method': 'Compute R² for each attribute separately, then average',
                    'formula': 'R² = mean(R²₁, R²₂, ..., R²₁₇)',
                    'use_case': 'Equal weight per attribute, fair comparison across attributes'
                },
                'sample_average_r2': {
                    'value': overall_metrics['R2_sample_average'],
                    'method': 'Compute R² for each sample separately, then average',
                    'formula': 'R² = mean(R²_sample₁, R²_sample₂, ...)',
                    'use_case': 'Per-sample performance evaluation'
                },
                'variance_weighted_r2': {
                    'value': overall_metrics['R2_variance_weighted'],
                    'method': 'Weight each attribute R² by its variance',
                    'formula': 'R² = Σ(wᵢ × R²ᵢ), where wᵢ = Var(attrᵢ) / Σ Var',
                    'use_case': 'Emphasize attributes with higher variance'
                }
            },
            'recommendation': 'Use overall_r2 as primary metric for paper reporting'
        },
        'summary': {
            'best_attributes': [],
            'worst_attributes': []
        }
    }
    
    # 找出最好和最差的属性
    sorted_by_mae = sorted(per_attr_metrics.items(), key=lambda x: x[1]['MAE'])
    results['summary']['best_attributes'] = [
        {'name': name, 'MAE': metrics['MAE'], 'R2': metrics['R2']}
        for name, metrics in sorted_by_mae[:5]
    ]
    results['summary']['worst_attributes'] = [
        {'name': name, 'MAE': metrics['MAE'], 'R2': metrics['R2']}
        for name, metrics in sorted_by_mae[-5:]
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_types(results), f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 结果已保存到: {output_file}")


def generate_latex_table(overall_metrics, per_attr_metrics):
    """生成LaTeX表格格式"""
    print("\n📝 LaTeX表格格式 (用于论文):")
    print("-"*80)
    print("% 整体指标 (使用整体R²)")
    print(f"AstroSight (InternVL2.5-38B) & {overall_metrics['MAE']:.4f} & "
          f"{overall_metrics['MSE']:.4f} & {overall_metrics['R2_overall']:.4f} \\\\")
    
    print("\n% 整体指标 (使用属性平均R²)")
    print(f"AstroSight (InternVL2.5-38B) & {overall_metrics['MAE']:.4f} & "
          f"{overall_metrics['MSE']:.4f} & {overall_metrics['R2_attribute_average']:.4f} \\\\")
    
    print("\n% R²对比表格")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Method & Overall R² & Attr-Avg R² & Variance-Weighted R² & Correlation \\\\")
    print("\\hline")
    print(f"AstroSight & {overall_metrics['R2_overall']:.4f} & "
          f"{overall_metrics['R2_attribute_average']:.4f} & "
          f"{overall_metrics['R2_variance_weighted']:.4f} & "
          f"{overall_metrics['pearson_correlation']:.4f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    
    print("\n% 各属性指标 (前5个)")
    attribute_names = list(per_attr_metrics.keys())[:5]
    for attr_name in attribute_names:
        metrics = per_attr_metrics[attr_name]
        print(f"{attr_name} & {metrics['MAE']:.4f} & {metrics['MSE']:.4f} & "
              f"{metrics['R2']:.4f} \\\\")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate AstroSight Galaxy Attribute Regression Performance'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='InternVL2.5_38B_arttibute_regression_lora.jsonl',
        help='Input JSONL file with predictions and labels'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results (default: input_name_evaluation.json)'
    )
    
    args = parser.parse_args()
    
    # 设置输入输出路径
    input_file = Path(args.input)
    if not input_file.is_absolute():
        input_file = Path('/mnt/acmis_hby/Paper_experiment_one/AstroSight_GalaxyMorph_LLM_infer_result') / input_file
    
    if args.output is None:
        output_file = input_file.parent / f"{input_file.stem}_evaluation.json"
    else:
        output_file = Path(args.output)
    
    print("="*80)
    print("🚀 AstroSight 星系属性回归评估")
    print("="*80)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print("="*80)
    
    # 加载数据
    predictions, labels, attribute_names = load_predictions_and_labels(input_file)
    
    # 计算整体指标
    print("\n🔄 计算整体指标...")
    overall_metrics = calculate_overall_metrics(predictions, labels)
    
    # 计算每个属性的指标
    print("🔄 计算各属性指标...")
    per_attr_metrics = calculate_per_attribute_metrics(predictions, labels, attribute_names)
    
    # 打印结果
    print_evaluation_results(overall_metrics, per_attr_metrics, attribute_names)
    
    # 保存结果
    save_results(overall_metrics, per_attr_metrics, output_file)
    
    # 生成LaTeX表格
    generate_latex_table(overall_metrics, per_attr_metrics)
    
    print("\n" + "="*80)
    print("✅ 评估完成!")
    print("="*80)


if __name__ == '__main__':
    main()

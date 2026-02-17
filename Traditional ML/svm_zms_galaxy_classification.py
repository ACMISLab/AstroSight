#!/usr/bin/env python3
"""
SVM + Zernike Moments Galaxy Classification
使用Zernike Moments特征和SVM进行星系形态分类
传统机器学习基线模型
"""

import argparse
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


def load_features(feature_path, split):
    """加载预提取的Zernike特征"""
    feature_file = os.path.join(feature_path, f'{split}_zernike_features.npz')
    
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"特征文件不存在: {feature_file}")
    
    data = np.load(feature_file, allow_pickle=True)
    features = data['features']
    labels = data['labels']
    class_names = data['class_names'].tolist()
    
    return features, labels, class_names


def train_svm(X_train, y_train, class_weights=None):
    """训练SVM分类器"""
    print("\n" + "="*60)
    print("🚀 开始训练 SVM")
    print("="*60)
    print(f"训练样本: {X_train.shape[0]:,}")
    print(f"特征维度: {X_train.shape[1]}")
    print(f"Kernel: RBF")
    print(f"C: 1.5")
    print(f"Gamma: scale")
    print(f"Class Weight: {'balanced' if class_weights else 'None'}")
    print("="*60)
    
    # 特征标准化
    print("\n标准化特征...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    print("✓ 特征标准化完成")
    
    # 训练SVM
    print("\n训练SVM模型...")
    start_time = time.time()
    
    svm_model = SVC(
        kernel='rbf',
        C=1.5,
        gamma='scale',
        probability=True,
        class_weight='balanced' if class_weights else None,
        random_state=42,
        verbose=False
    )
    
    svm_model.fit(X_train_scaled, y_train)
    
    training_time = time.time() - start_time
    
    print(f"✓ SVM训练完成")
    print(f"  训练时间: {training_time:.2f} 秒 ({training_time/60:.2f} 分钟)")
    
    return svm_model, scaler, training_time


def evaluate_model(model, scaler, X_test, y_test, class_names):
    """评估模型"""
    print("\n" + "="*60)
    print("📊 测试集评估")
    print("="*60)
    
    # 标准化测试特征
    X_test_scaled = scaler.transform(X_test)
    
    # 预测
    print("正在预测...")
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=list(range(len(class_names)))
    )
    
    # 宏平均和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # 组织结果
    results = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted)
        },
        'per_class_metrics': {}
    }
    
    for i, class_name in enumerate(class_names):
        results['per_class_metrics'][class_name] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return results, y_pred, y_proba


def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # 归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm, cm_normalized


def main():
    parser = argparse.ArgumentParser('SVM + Zernike Moments Galaxy Classification')
    
    # Data parameters
    parser.add_argument('--feature_path', default='./features', type=str,
                       help='预提取特征路径')
    parser.add_argument('--output_dir', default='./result', type=str,
                       help='输出目录')
    
    # Model parameters
    parser.add_argument('--use_class_weight', action='store_true', default=True,
                       help='使用类别权重处理不平衡')
    
    # Testing
    parser.add_argument('--test_only', action='store_true',
                       help='仅测试模式')
    parser.add_argument('--model_path', type=str, default='',
                       help='预训练模型路径')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 SVM + Zernike Moments Galaxy Classification")
    print("="*60)
    print(f"特征路径: {args.feature_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载特征
    print("\n" + "="*60)
    print("📂 加载特征")
    print("="*60)
    
    print("加载训练集特征...")
    X_train, y_train, class_names = load_features(args.feature_path, 'train')
    print(f"✓ 训练集: {X_train.shape[0]:,} 样本, {X_train.shape[1]} 维特征")
    
    print("\n加载验证集特征...")
    X_val, y_val, _ = load_features(args.feature_path, 'val')
    print(f"✓ 验证集: {X_val.shape[0]:,} 样本")
    
    print("\n加载测试集特征...")
    X_test, y_test, _ = load_features(args.feature_path, 'test')
    print(f"✓ 测试集: {X_test.shape[0]:,} 样本")
    
    # 合并训练集和验证集
    X_train_full = np.concatenate([X_train, X_val], axis=0)
    y_train_full = np.concatenate([y_train, y_val], axis=0)
    
    print(f"\n合并后训练集: {X_train_full.shape[0]:,} 样本")
    print(f"类别数: {len(class_names)}")
    print(f"类别名称: {class_names}")
    
    model_name = f"svm_zms_galaxy_{len(class_names)}_classes"
    
    if args.test_only:
        # 仅测试模式
        print("\n" + "="*60)
        print("🔍 仅测试模式")
        print("="*60)
        
        if not args.model_path or not os.path.exists(args.model_path):
            print(f"❌ 模型文件不存在: {args.model_path}")
            return
        
        print(f"📁 加载模型: {args.model_path}")
        with open(args.model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        model = checkpoint['model']
        scaler = checkpoint['scaler']
        class_names = checkpoint['class_names']
        
    else:
        # 训练模式
        model, scaler, training_time = train_svm(
            X_train_full, y_train_full,
            class_weights=args.use_class_weight
        )
        
        # 保存模型
        model_path = output_dir / f'{model_name}_model.pkl'
        checkpoint = {
            'model': model,
            'scaler': scaler,
            'class_names': class_names,
            'training_time': training_time
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        print(f"\n💾 模型已保存: {model_path}")
    
    # 测试评估
    results, y_pred, y_proba = evaluate_model(
        model, scaler, X_test, y_test, class_names
    )
    
    # 打印结果
    print("\n" + "="*60)
    print("🎯 测试结果")
    print("="*60)
    print(f"准确率 (Accuracy):        {results['overall_metrics']['accuracy']:.4f}")
    print(f"精确率 (Precision-Macro): {results['overall_metrics']['precision_macro']:.4f}")
    print(f"召回率 (Recall-Macro):    {results['overall_metrics']['recall_macro']:.4f}")
    print(f"F1分数 (F1-Macro):        {results['overall_metrics']['f1_macro']:.4f}")
    print(f"\n加权平均:")
    print(f"精确率 (Precision-Weighted): {results['overall_metrics']['precision_weighted']:.4f}")
    print(f"召回率 (Recall-Weighted):    {results['overall_metrics']['recall_weighted']:.4f}")
    print(f"F1分数 (F1-Weighted):        {results['overall_metrics']['f1_weighted']:.4f}")
    
    print("\n各类别详细指标:")
    print("-"*60)
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"{class_name:25s} | P:{metrics['precision']:.3f} R:{metrics['recall']:.3f} "
              f"F1:{metrics['f1']:.3f} (n={metrics['support']})")
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_test],
        'predicted_label': [class_names[i] for i in y_pred],
        'correct': [t == p for t, p in zip(y_test, y_pred)]
    })
    predictions_df.to_csv(output_dir / f'{model_name}_predictions.csv', index=False)
    
    # 绘制混淆矩阵
    cm_raw, cm_normalized = plot_confusion_matrix(
        y_test, y_pred, class_names,
        output_dir / f'{model_name}_confusion_matrix.png'
    )
    
    # 保存结果文件
    print("\n" + "="*60)
    print("💾 保存结果文件")
    print("="*60)
    
    # 组织最终结果
    final_results = {
        'model_info': {
            'name': 'SVM + Zernike Moments',
            'kernel': 'rbf',
            'C': 1.5,
            'gamma': 'scale',
            'class_weight': 'balanced' if args.use_class_weight else None,
            'zernike_order': 45,
            'feature_dimension': int(X_train_full.shape[1])
        },
        'training_config': {
            'use_class_weight': args.use_class_weight,
            'test_only_mode': args.test_only
        },
        'results': results,
        'confusion_matrix_raw': cm_raw.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names
    }
    
    # 保存完整结果
    output_filename = f'{model_name}_test_results.json' if args.test_only else f'{model_name}_results.json'
    with open(output_dir / output_filename, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✅ 预测结果: {model_name}_predictions.csv")
    print(f"✅ 完整结果: {output_filename}")
    print(f"✅ 混淆矩阵图: {model_name}_confusion_matrix.png")
    if not args.test_only:
        print(f"✅ 训练模型: {model_name}_model.pkl")
    
    print("\n" + "="*60)
    print("✨ SVM + Zernike Moments 星系分类任务完成!")
    print("="*60)


if __name__ == '__main__':
    main()

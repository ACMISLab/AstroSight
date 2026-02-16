#!/usr/bin/env python3
"""
ECA-CNN 模型评估脚本 - 仅评估模式
用于加载已训练的模型并完成测试集评估
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 使用GPU 1

import numpy as np
import pandas as pd
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def convert_numpy_types(obj):
    """递归转换NumPy类型为Python原生类型"""
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


def create_test_generator(data_path, batch_size=32, image_size=(224, 224)):
    """创建测试数据生成器"""
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator


def evaluate_model(model, test_generator):
    """评估模型"""
    # 重置生成器
    test_generator.reset()
    
    # 预测
    print("🔄 开始预测...")
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # 获取类别名称
    class_names = list(test_generator.class_indices.keys())
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # 各类别详细指标
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(class_names)), zero_division=0
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
    
    return results, y_true, y_pred, class_names


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 原始混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=12)
    
    # 归一化混淆矩阵
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Proportion'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 混淆矩阵已保存: {save_path}")
    
    return cm, cm_normalized


def main():
    print("="*60)
    print("🔍 ECA-CNN 模型评估（仅评估模式）")
    print("="*60)
    
    # 配置
    data_path = '/mnt/acmis_hby/Paper_experiment_one/baselines_dataset'
    model_path = 'eca_cnn_galaxy_8_classes_best.h5'
    model_name = 'eca_cnn_galaxy_8_classes'
    batch_size = 32
    image_size = (224, 224)
    
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        return
    
    # 创建测试数据生成器
    print("\n📂 加载测试数据...")
    test_generator = create_test_generator(data_path, batch_size, image_size)
    class_names = list(test_generator.class_indices.keys())
    
    print(f"✅ 测试集: {test_generator.samples:,} 样本")
    print(f"✅ 类别数: {len(class_names)}")
    print(f"✅ 类别名称: {class_names}")
    
    # 加载模型（使用自定义对象处理Lambda层）
    print(f"\n📥 加载模型: {model_path}")
    
    # 定义自定义Lambda函数来处理output_shape
    def transpose_lambda(x):
        return tf.transpose(x, [0, 2, 1])
    
    # 创建自定义对象字典
    custom_objects = {
        'transpose_lambda': transpose_lambda
    }
    
    try:
        # 使用custom_objects加载模型
        model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"⚠️ 加载失败，错误: {e}")
        print("🔧 尝试重新编译模型...")
        
        # 如果加载失败，尝试导入修复后的模型定义
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from eca_cnn_galaxy_classification import create_eca_cnn_model
        
        # 重新创建模型结构
        print("🔧 重新创建模型结构...")
        model = create_eca_cnn_model(input_shape=(224, 224, 3), num_classes=len(class_names))
        
        # 尝试加载权重
        try:
            model.load_weights(model_path)
            print("✅ 使用权重加载成功")
        except Exception as e2:
            print(f"❌ 权重加载也失败: {e2}")
            print("💡 建议：需要重新训练模型或使用修复后的代码保存模型")
            return
    
    # 获取模型参数
    total_params = model.count_params()
    print(f"📊 模型参数量: {total_params:,}")
    
    # 测试评估
    print("\n" + "="*60)
    print("📊 测试集评估")
    print("="*60)
    results, y_true, y_pred, class_names = evaluate_model(model, test_generator)
    
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
        print(f"{class_name:25s} | P:{metrics['precision']:.3f} R:{metrics['recall']:.3f} F1:{metrics['f1']:.3f} (n={metrics['support']})")
    
    # 保存预测结果
    print("\n" + "="*60)
    print("💾 保存结果文件")
    print("="*60)
    
    predictions_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'predicted_label': [class_names[i] for i in y_pred],
        'correct': [t == p for t, p in zip(y_true, y_pred)]
    })
    predictions_df.to_csv(f'{model_name}_predictions.csv', index=False)
    print(f"✅ 预测结果: {model_name}_predictions.csv")
    
    # 绘制混淆矩阵
    cm_raw, cm_normalized = plot_confusion_matrix(
        y_true, y_pred, class_names, f'{model_name}_confusion_matrix.png'
    )
    
    # 保存完整结果
    final_results = {
        'model_info': {
            'name': 'ECA-CNN (DIAT-DSCNN-ECA-Net)',
            'total_parameters': int(total_params),
            'trainable_parameters': int(total_params)
        },
        'training_config': {
            'image_size': image_size[0],
            'batch_size': batch_size,
            'test_only_mode': True,
            'model_path': model_path
        },
        'results': results,
        'confusion_matrix_raw': cm_raw.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names
    }
    
    output_filename = f'{model_name}_results.json'
    with open(output_filename, 'w') as f:
        json.dump(convert_numpy_types(final_results), f, indent=2)
    print(f"✅ 完整结果: {output_filename}")
    
    # 单独保存混淆矩阵数据
    confusion_matrix_data = {
        'class_names': class_names,
        'confusion_matrix_raw': cm_raw.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_mapping': {name: idx for idx, name in enumerate(class_names)},
        'matrix_description': {
            'raw': '原始混淆矩阵 - 每个元素表示实际预测的样本数量',
            'normalized': '归一化混淆矩阵 - 每行归一化为1，表示每个真实类别的预测准确率'
        }
    }
    
    cm_filename = f'{model_name}_confusion_matrix_data.json'
    with open(cm_filename, 'w') as f:
        json.dump(convert_numpy_types(confusion_matrix_data), f, indent=2)
    print(f"✅ 混淆矩阵数据: {cm_filename}")
    
    print("\n" + "="*60)
    print("✅ 评估完成！")
    print("="*60)
    print(f"📝 论文对比表格格式:")
    print(f"ECA-CNN & {results['overall_metrics']['accuracy']:.4f} & {results['overall_metrics']['precision_macro']:.4f} & {results['overall_metrics']['recall_macro']:.4f} & {results['overall_metrics']['f1_macro']:.4f} \\\\")


if __name__ == "__main__":
    main()

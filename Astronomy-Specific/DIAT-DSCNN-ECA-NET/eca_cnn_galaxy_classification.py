#!/usr/bin/env python3
"""
ECA-CNN Galaxy Classification Script
基于DIAT-DSCNN-ECA-Net的星系形态分类实现
使用分离卷积和高效通道注意力机制
"""

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Input, SeparableConv2D, BatchNormalization, Activation, 
    MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout,
    Conv1D, Add, Lambda, Reshape, Multiply

    
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 设置GPU内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def eca_module(x, k_size=5):
    """ECA注意力模块 - Keras 3兼容版本"""
    # 全局平均池化
    squeeze = GlobalAveragePooling2D()(x)
    
    # 扩展维度用于Conv1D
    squeeze_expanded = Reshape((1, -1))(squeeze)
    
    # 1D卷积生成注意力权重
    attn = Conv1D(filters=1,
                  kernel_size=k_size,
                  padding='same',
                  kernel_initializer='he_normal',
                  use_bias=False)(squeeze_expanded)
    
    # 转置和扩展维度
    attn = Lambda(lambda x: tf.transpose(x, [0, 2, 1]), output_shape=lambda input_shape: (input_shape[0], input_shape[2], input_shape[1]))(attn)
    attn = Reshape((1, 1, -1))(attn)
    
    # Sigmoid激活
    attn = Activation('sigmoid')(attn)
    
    # 通道注意力加权
    scale = Multiply()([x, attn])
    
    return scale


def create_eca_cnn_model(input_shape=(224, 224, 3), num_classes=8):
    """
    创建ECA-CNN模型
    """
    img_input = Input(input_shape)
    
    # 初始卷积层
    x = SeparableConv2D(32, (3, 3), strides=(1, 1), padding='same',
                       depthwise_initializer='he_normal',
                       pointwise_initializer="he_normal",
                       use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x_offset = Activation('relu')(x)
    
    # Block 1: 多尺度分离卷积 + ECA注意力
    residual1 = SeparableConv2D(32, (3, 3), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual1 = BatchNormalization()(residual1)
    residual1 = Activation('relu')(residual1)
    
    residual2 = SeparableConv2D(32, (5, 5), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual2 = BatchNormalization()(residual2)
    residual2 = Activation('relu')(residual2)
    
    residual3 = SeparableConv2D(32, (7, 7), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual3 = BatchNormalization()(residual3)
    residual3 = Activation('relu')(residual3)
    
    # ECA注意力
    channel_attention_map = eca_module(x_offset)
    
    # 融合多尺度特征
    x11 = Add()([residual1, residual2, residual3, channel_attention_map])
    
    # 下采样
    x = SeparableConv2D(64, (3, 3), strides=(1, 1), padding='same',
                       depthwise_initializer='he_normal',
                       pointwise_initializer="he_normal",
                       use_bias=False)(x11)
    x = BatchNormalization()(x)
    x_offset = Activation('relu')(x)
    x_offset = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x_offset)
    
    # Block 2: 多尺度分离卷积 + ECA注意力
    residual1 = SeparableConv2D(64, (3, 3), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual1 = BatchNormalization()(residual1)
    residual1 = Activation('relu')(residual1)
    
    residual2 = SeparableConv2D(64, (5, 5), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual2 = BatchNormalization()(residual2)
    residual2 = Activation('relu')(residual2)
    
    residual3 = SeparableConv2D(64, (7, 7), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual3 = BatchNormalization()(residual3)
    residual3 = Activation('relu')(residual3)
    
    # ECA注意力
    channel_attention_map = eca_module(x_offset)
    
    # 融合多尺度特征
    x11 = Add()([residual1, residual2, residual3, channel_attention_map])
    
    # 下采样
    x = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same',
                       depthwise_initializer='he_normal',
                       pointwise_initializer="he_normal",
                       use_bias=False)(x11)
    x = BatchNormalization()(x)
    x_offset = Activation('relu')(x)
    x_offset = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x_offset)
    
    # Block 3: 多尺度分离卷积 + ECA注意力
    residual1 = SeparableConv2D(128, (3, 3), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual1 = BatchNormalization()(residual1)
    residual1 = Activation('relu')(residual1)
    
    residual2 = SeparableConv2D(128, (5, 5), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual2 = BatchNormalization()(residual2)
    residual2 = Activation('relu')(residual2)
    
    residual3 = SeparableConv2D(128, (5, 5), strides=(1, 1), padding='same',
                               depthwise_initializer='he_normal',
                               pointwise_initializer="he_normal",
                               use_bias=False)(x_offset)
    residual3 = BatchNormalization()(residual3)
    residual3 = Activation('relu')(residual3)
    
    # ECA注意力
    channel_attention_map = eca_module(x_offset)
    
    # 融合多尺度特征
    x11 = Add()([residual1, residual2, residual3, channel_attention_map])
    
    # 最终卷积层
    x = SeparableConv2D(256, (3, 3), strides=(1, 1), padding='same',
                       depthwise_initializer='he_normal',
                       pointwise_initializer="he_normal",
                       use_bias=False)(x11)
    x = BatchNormalization()(x)
    x_offset = Activation('relu')(x)
    
    # 全局平均池化和分类器
    x15 = GlobalAveragePooling2D()(x_offset)
    x = Dropout(0.3)(x15)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(img_input, output)
    return model


def lr_schedule(epoch):
    """学习率调度"""
    learning_rate = 0.01
    if epoch > 10:
        learning_rate = 0.002
    if epoch > 25:
        learning_rate = 0.001
    if epoch > 35:
        learning_rate = 0.0001
    return learning_rate


def create_data_generators(data_path, batch_size=32, image_size=(224, 224)):
    """创建数据生成器"""
    # 训练数据增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=180,
        horizontal_flip=True,
        vertical_flip=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    # 验证和测试数据标准化
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_path, 'val'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(data_path, 'test'),
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator


def evaluate_model(model, test_generator):
    """评估模型"""
    # 重置生成器
    test_generator.reset()
    
    # 预测
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # 获取类别名称
    class_names = list(test_generator.class_indices.keys())
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names)))
    )
    
    # 宏平均和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
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
            'accuracy': float(precision[i]),
            'support': int(support[i])
        }
    
    return results, y_true, y_pred, class_names


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    
    display_names = [name.replace('_', ' ').title() for name in class_names]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=display_names,
        yticklabels=display_names,
        cbar_kws={'label': 'Normalized Accuracy'}
    )
    plt.title('Confusion Matrix - ECA-CNN Galaxy Classification', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm_raw, cm_normalized


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
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


def main():
    parser = argparse.ArgumentParser(description='ECA-CNN Galaxy Classification')
    parser.add_argument('--data_path', type=str,
                       default='/mnt/acmis_hby/Paper_experiment_one/baselines_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--test_only', action='store_true', help='Only run testing, skip training')
    parser.add_argument('--model_path', type=str, default='eca_cnn_galaxy_best.h5', 
                       help='Path to pre-trained model for testing')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    print("="*60)
    print("🚀 ECA-CNN Galaxy Classification")
    print("="*60)
    print(f"使用设备: GPU {args.gpu}")
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"Keras版本: {keras.__version__}")
    
    # 数据生成器
    train_generator, val_generator, test_generator = create_data_generators(
        args.data_path, args.batch_size, (args.image_size, args.image_size)
    )
    
    # 获取类别信息
    class_names = list(train_generator.class_indices.keys())
    num_classes = len(class_names)
    
    print("\n" + "="*60)
    print("📊 数据集统计")
    print("="*60)
    print(f"训练集: {train_generator.samples:,} 样本")
    print(f"验证集: {val_generator.samples:,} 样本")
    print(f"测试集: {test_generator.samples:,} 样本")
    print(f"类别数: {num_classes}")
    print(f"类别名称: {class_names}")
    
    # 创建模型
    model = create_eca_cnn_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes
    )
    
    # 计算模型参数
    total_params = model.count_params()
    
    print("\n" + "="*60)
    print("🏗️ 模型信息")
    print("="*60)
    print(f"模型名称: ECA-CNN (DIAT-DSCNN-ECA-Net)")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {total_params:,}")
    print(f"输入尺寸: {args.image_size}×{args.image_size}×3")
    
    model_name = f"eca_cnn_galaxy_{num_classes}_classes"
    
    if args.test_only:
        # 仅测试模式
        print("🔍 仅测试模式...")
        if not os.path.exists(args.model_path):
            print(f"❌ 模型文件不存在: {args.model_path}")
            return
        
        print(f"📁 加载预训练模型: {args.model_path}")
        model = keras.models.load_model(args.model_path)
        
    else:
        # 训练模式
        print("\n" + "="*60)
        print("🚀 开始训练 ECA-CNN")
        print("="*60)
        print(f"训练轮数: {args.epochs}")
        print(f"批次大小: {args.batch_size}")
        print(f"初始学习率: {args.learning_rate}")
        print(f"学习率调度: Epoch 10→0.002, Epoch 25→0.001, Epoch 35→0.0001")
        print(f"早停耐心: 15 epochs")
        print("="*60)
        
        # 编译模型
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=args.learning_rate),
            metrics=['accuracy']
        )
        
        # 回调函数
        lr_callback = LearningRateScheduler(lr_schedule)
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        model_checkpoint = ModelCheckpoint(
            f'{model_name}_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        callbacks = [lr_callback, early_stopping, model_checkpoint]
        
        # 训练模型
        start_time = time.time()
        history = model.fit(
            train_generator,
            epochs=args.epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ 训练完成")
        print("="*60)
        print(f"总用时: {training_time/60:.1f} 分钟 ({training_time:.0f} 秒)")
        print(f"最佳验证准确率: {max(history.history['val_accuracy']):.4f}")
        print(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}")
        print(f"最终验证准确率: {history.history['val_accuracy'][-1]:.4f}")
        
        # 保存训练历史
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f'{model_name}_history.csv', index=False)
        print(f"💾 训练历史保存: {model_name}_history.csv")
        
        # 加载最佳模型
        model = keras.models.load_model(f'{model_name}_best.h5')
    
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
    predictions_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'predicted_label': [class_names[i] for i in y_pred],
        'correct': [t == p for t, p in zip(y_true, y_pred)]
    })
    predictions_df.to_csv(f'{model_name}_predictions.csv', index=False)
    
    # 绘制混淆矩阵
    cm_raw, cm_normalized = plot_confusion_matrix(
        y_true, y_pred, class_names, f'{model_name}_confusion_matrix.png'
    )
    
    # 保存结果文件
    print("\n" + "="*60)
    print("💾 保存结果文件")
    print("="*60)
    
    # 组织最终结果
    final_results = {
        'model_info': {
            'name': 'ECA-CNN (DIAT-DSCNN-ECA-Net)',
            'total_parameters': int(total_params),
            'trainable_parameters': int(total_params)
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'image_size': args.image_size,
            'test_only_mode': args.test_only,
            'model_path': args.model_path if args.test_only else None
        },
        'results': results,
        'confusion_matrix_raw': cm_raw.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': class_names
    }
    
    # 保存完整结果
    output_filename = f'{model_name}_test_results.json' if args.test_only else f'{model_name}_results.json'
    with open(output_filename, 'w') as f:
        json.dump(convert_numpy_types(final_results), f, indent=2)
    
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
    
    print(f"✅ 预测结果: {model_name}_predictions.csv")
    print(f"✅ 完整结果: {output_filename}")
    print(f"✅ 混淆矩阵图: {model_name}_confusion_matrix.png")
    print(f"✅ 混淆矩阵数据: {cm_filename}")
    if not args.test_only:
        print(f"✅ 训练历史: {model_name}_history.csv")
        print(f"✅ 最佳模型: {model_name}_best.h5")
    
    print("\n" + "="*60)
    print("✨ ECA-CNN 星系分类任务完成!")
    print("="*60)


if __name__ == '__main__':
    main()

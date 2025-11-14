#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import shutil
import re
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Scikit-learn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn import metrics

# TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, MaxPooling1D, 
    Dropout, Flatten, Dense
)
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

# Zernike Moments
from ZEMO import zemo

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

class GalaxyZMExperiment:
    """星系ZM对比实验类"""
    
    def __init__(self, config):
        self.config = config
        self.class_mapping = {
            'A': 'round_elliptical',
            'B': 'in_between_elliptical', 
            'C': 'cigar_shaped_elliptical',
            'D': 'edge_on',
            'E': 'barred_spirals',
            'F': 'unbarred_spirals',
            'G': 'irregular',
            'H': 'merger'
        }
        
        self.class_names = [
            'Round Elliptical',      # A: 0
            'In-between Elliptical', # B: 1
            'Cigar Elliptical',      # C: 2
            'Edge-on Disk',          # D: 3
            'Barred Spiral',         # E: 4
            'Unbarred Spiral',       # F: 5
            'Irregular',             # G: 6
            'Merger'                 # H: 7
        ]
        
        print("="*80)
        print("8-Class Galaxy Classification Experiment Initialized")
        print("="*80)
        
    def build_dataset(self, jsonl_paths, output_root):
        """构建统一格式的数据集"""
        print(f"Building dataset from JSONL files to: {output_root}")
        
        # 创建根目录及所有类别子目录
        os.makedirs(output_root, exist_ok=True)
        for class_dir in self.class_mapping.values():
            os.makedirs(os.path.join(output_root, class_dir), exist_ok=True)

        total_processed = 0
        # 处理每个JSONL文件
        for jsonl_file in jsonl_paths:
            print(f"Processing: {jsonl_file}")
            with open(jsonl_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # 解析分类标签
                        option = None
                        for msg in data['messages']:
                            if msg['role'] == 'assistant':
                                match = re.search(r'option ([A-H])', msg['content'], re.I)
                                if match:
                                    option = match.group(1).upper()
                                    break
                        
                        if option not in self.class_mapping:
                            continue
                        
                        # 处理关联图片
                        for src_path in data['images']:
                            if not os.path.exists(src_path):
                                continue
                            
                            # 构建目标路径
                            class_dir = self.class_mapping[option]
                            filename = os.path.basename(src_path)
                            dest_path = os.path.join(output_root, class_dir, filename)
                            
                            # 处理文件名冲突
                            counter = 1
                            while os.path.exists(dest_path):
                                base, ext = os.path.splitext(filename)
                                dest_path = os.path.join(output_root, class_dir, 
                                                       f"{base}_v{counter}{ext}")
                                counter += 1
                            
                            shutil.copy2(src_path, dest_path)
                            total_processed += 1
                            
                    except Exception as e:
                        print(f"Error processing line {line_num} in {jsonl_file}: {str(e)}")
        
        print(f"Dataset building completed! Total images: {total_processed}")
        return total_processed

    def calculate_zernike_moments(self, data_dir, image_size, zernike_order):
        """计算Zernike矩"""
        print(f"Calculating ZMs for: {data_dir}")
        
        ZBFSTR = zemo.zernike_bf(image_size, zernike_order, 1)
        
        image_files = [os.path.join(data_dir, filename) 
                      for filename in os.listdir(data_dir) 
                      if filename.endswith('.jpg')]
        
        zernike_moments = []
        total_files = len(image_files)
        
        for i, img_path in enumerate(image_files):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{total_files}")
                
            image = cv2.imread(img_path)
            resized_image = cv2.resize(image, (image_size, image_size))
            im = resized_image[:, :, 0]  # 使用R通道
            Z = np.abs(zemo.zernike_mom(np.array(im), ZBFSTR))
            zernike_moments.append(Z)
        
        df = pd.DataFrame(zernike_moments)
        print(f"  Completed! Shape: {df.shape}")
        return df

    def compute_all_zms(self):
        """计算所有类别的ZM特征"""
        print("Computing Zernike Moments for all classes...")
        
        image_size = self.config['image_size']
        zernike_order = self.config['zernike_order']
        
        # 各类别路径
        paths = {
            'barred_spirals': os.path.join(self.config['dataset_path'], 'barred_spirals'),
            'unbarred_spirals': os.path.join(self.config['dataset_path'], 'unbarred_spirals'),
            'round_elliptical': os.path.join(self.config['dataset_path'], 'round_elliptical'),
            'cigar_shaped_elliptical': os.path.join(self.config['dataset_path'], 'cigar_shaped_elliptical'),
            'in_between_elliptical': os.path.join(self.config['dataset_path'], 'in_between_elliptical'),
            'edge_on': os.path.join(self.config['dataset_path'], 'edge_on'),
            'irregular': os.path.join(self.config['dataset_path'], 'irregular'),
            'merger': os.path.join(self.config['dataset_path'], 'merger')
        }
        
        # 计算并保存ZM
        for class_name, path in paths.items():
            if os.path.exists(path):
                zm_df = self.calculate_zernike_moments(path, image_size, zernike_order)
                output_path = os.path.join(self.config['zm_output_path'], f"{class_name}_zms.csv")
                zm_df.to_csv(output_path, index=False)
                print(f"Saved ZMs for {class_name}: {output_path}")
            else:
                print(f"Warning: Path not found: {path}")

    def load_and_organize_data(self):
        """加载和组织数据 - 按照A-H顺序"""
        print("Loading and organizing ZM data...")
        
        zm_path = self.config['zm_output_path']
        
        # 按照A-H顺序加载数据
        data_files = [
            ('round_elliptical_zms.csv', 'A'),      # A: round elliptical
            ('in_between_elliptical_zms.csv', 'B'), # B: in-between elliptical
            ('cigar_shaped_elliptical_zms.csv', 'C'), # C: cigar-shaped elliptical
            ('edge_on_zms.csv', 'D'),               # D: edge-on
            ('barred_spirals_zms.csv', 'E'),        # E: barred spirals
            ('unbarred_spirals_zms.csv', 'F'),      # F: unbarred spirals
            ('irregular_zms.csv', 'G'),             # G: irregular
            ('merger_zms.csv', 'H')                 # H: merger
        ]
        
        all_data = []
        all_labels = []
        
        for i, (filename, class_label) in enumerate(data_files):
            filepath = os.path.join(zm_path, filename)
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                if "Unnamed: 0" in df.columns:
                    df.drop("Unnamed: 0", axis=1, inplace=True)
                
                all_data.append(df.values)
                all_labels.extend([i] * len(df))  # 标签0-7对应A-H
                
                print(f"Loaded {class_label}({i}): {len(df)} samples")
            else:
                print(f"Warning: File not found: {filepath}")
        
        # 合并数据
        self.all_zm_data = np.concatenate(all_data, axis=0)
        self.all_labels = np.array(all_labels)
        
        print(f"Total data shape: {self.all_zm_data.shape}")
        print(f"Total labels: {len(self.all_labels)}")
        print(f"Label distribution: {np.bincount(self.all_labels)}")
        print("Class mapping: 0=A(round_elliptical), 1=B(in_between_elliptical), 2=C(cigar_shaped_elliptical), 3=D(edge_on), 4=E(barred_spirals), 5=F(unbarred_spirals), 6=G(irregular), 7=H(merger)")

    def calculate_tss_multiclass(self, y_true, y_pred, num_classes=8):
        """计算多分类TSS (True Skill Statistic)"""
        cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        tss_scores = []
        
        for i in range(num_classes):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            
            if (tp + fn) > 0 and (fp + tn) > 0:
                sensitivity = tp / (tp + fn)
                specificity = tn / (tn + fp)
                tss = sensitivity + specificity - 1
            else:
                tss = 0
            tss_scores.append(tss)
        
        return tss_scores, np.mean(tss_scores)

    def calculate_class_weights(self, y_train):
        """计算类别权重"""
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        num_classes = len(class_counts)

        class_weights = {}
        for i in range(num_classes):
            if class_counts[i] > 0:
                class_weights[i] = total_samples / (num_classes * class_counts[i])
            else:
                class_weights[i] = 1.0
        
        return class_weights

    def train_svm_model(self):
        """训练SVM模型"""
        print("\n" + "="*80)
        print("Training SVM Model")
        print("="*80)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.all_zm_data, self.all_labels, 
            test_size=self.config['test_size'], 
            shuffle=True, 
            random_state=42,
            stratify=self.all_labels
        )
        
        print(f"Train samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # 计算类别权重
        class_weights = self.calculate_class_weights(y_train)
        print(f"Class weights: {class_weights}")
        
        # 训练SVM
        model = SVC(
            kernel='rbf', 
            probability=True, 
            C=1.5, 
            gamma='scale', 
            class_weight=class_weights
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        tss_per_class, tss_avg = self.calculate_tss_multiclass(y_test, y_pred)
        
        # 输出结果
        print(f"Training time: {training_time:.2f} seconds")
        print("\nSVM Performance Metrics:")
        print("-" * 50)
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall:    {recall:.4f}")
        print(f"Weighted F1-Score:  {f1:.4f}")
        print(f"TSS (average):      {tss_avg:.4f}")
        
        print("\nPer-class TSS scores:")
        for i, (name, tss) in enumerate(zip(self.class_names, tss_per_class)):
            print(f"  {name:20s}: {tss:.4f}")
        
        # 详细分类报告
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names, digits=4))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tss_avg': tss_avg,
            'tss_per_class': tss_per_class,
            'confusion_matrix': cm,
            'training_time': training_time
        }

    def build_cnn_model(self, input_shape, num_classes=8):
        """构建1D-CNN模型"""
        x = Input(shape=input_shape)

        # 隐藏层
        c0 = Conv1D(256, kernel_size=3, strides=2, padding="same")(x)
        b0 = BatchNormalization()(c0)
        m0 = MaxPooling1D(pool_size=2)(b0)
        d0 = Dropout(0.1)(m0)

        c1 = Conv1D(128, kernel_size=3, strides=2, padding="same")(d0)
        b1 = BatchNormalization()(c1)
        m1 = MaxPooling1D(pool_size=2)(b1)
        d1 = Dropout(0.1)(m1)

        c2 = Conv1D(64, kernel_size=3, strides=2, padding="same")(d1)
        b2 = BatchNormalization()(c2)
        m2 = MaxPooling1D(pool_size=2)(b2)
        d2 = Dropout(0.1)(m2)

        f = Flatten()(d2)

        # 输出层
        de0 = Dense(64, activation='relu')(f)
        de1 = Dense(32, activation='relu')(de0)
        de2 = Dense(num_classes, activation='softmax')(de1)

        model = Model(inputs=x, outputs=de2, name="cnn_zm_8class_galaxy")
        return model

    def train_cnn_model(self):
        """训练CNN模型"""
        print("\n" + "="*80)
        print("Training 1D-CNN Model")
        print("="*80)
        
        # 数据分割
        X_train, X_test, y_train, y_test = train_test_split(
            self.all_zm_data, self.all_labels, 
            test_size=self.config['test_size'], 
            shuffle=True, 
            random_state=42,
            stratify=self.all_labels
        )
        
        # 数据预处理 - CNN需要3D输入 (samples, features, channels)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # 标签编码
        y_train_encoded = to_categorical(y_train, num_classes=8)
        
        print(f"Train shape: {X_train.shape}")
        print(f"Test shape: {X_test.shape}")
        
        # 计算类别权重
        class_weights = self.calculate_class_weights(y_train)
        print(f"Class weights: {class_weights}")
        
        # 构建模型
        model = self.build_cnn_model(input_shape=(X_train.shape[1], 1))
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        # 回调函数
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        
        # 训练
        start_time = time.time()
        history = model.fit(
            X_train, y_train_encoded,
            batch_size=self.config['batch_size'],
            epochs=self.config['epochs'],
            class_weight=class_weights,
            verbose=1,
            callbacks=[es],
            validation_split=0.1
        )
        training_time = time.time() - start_time
        
        # 预测
        y_pred_prob = model.predict(X_test)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        tss_per_class, tss_avg = self.calculate_tss_multiclass(y_test, y_pred)
        
        # 输出结果
        print(f"\nTraining time: {training_time:.2f} seconds")
        print("\n1D-CNN Performance Metrics:")
        print("-" * 50)
        print(f"Accuracy:           {accuracy:.4f}")
        print(f"Weighted Precision: {precision:.4f}")
        print(f"Weighted Recall:    {recall:.4f}")
        print(f"Weighted F1-Score:  {f1:.4f}")
        print(f"TSS (average):      {tss_avg:.4f}")
        
        print("\nPer-class TSS scores:")
        for i, (name, tss) in enumerate(zip(self.class_names, tss_per_class)):
            print(f"  {name:20s}: {tss:.4f}")
        
        # 详细分类报告
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names, digits=4))
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tss_avg': tss_avg,
            'tss_per_class': tss_per_class,
            'confusion_matrix': cm,
            'training_time': training_time,
            'history': history
        }

    def save_results(self, svm_results, cnn_results):
        """保存实验结果"""
        print("\n" + "="*80)
        print("Experiment Results Summary")
        print("="*80)
        
        results_summary = {
            'SVM': {
                'accuracy': svm_results['accuracy'],
                'precision': svm_results['precision'],
                'recall': svm_results['recall'],
                'f1_score': svm_results['f1_score'],
                'tss_avg': svm_results['tss_avg'],
                'training_time': svm_results['training_time']
            },
            'CNN': {
                'accuracy': cnn_results['accuracy'],
                'precision': cnn_results['precision'],
                'recall': cnn_results['recall'],
                'f1_score': cnn_results['f1_score'],
                'tss_avg': cnn_results['tss_avg'],
                'training_time': cnn_results['training_time']
            }
        }
        
        print("Method Comparison (8-class classification):")
        print("-"*80)
        print(f"{'Method':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TSS':<10} {'Time(s)':<10}")
        print("-"*80)
        
        for method, metrics in results_summary.items():
            print(f"{method:<15} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1_score']:<10.4f} {metrics['tss_avg']:<10.4f} {metrics['training_time']:<10.2f}")
        
        # 保存到文件
        output_file = os.path.join(self.config['output_path'], 'experiment_results.json')
        with open(output_file, 'w') as f:
            # 转换numpy数组为列表以便JSON序列化
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump({
                'svm_results': {k: convert_numpy(v) for k, v in svm_results.items() if k != 'model'},
                'cnn_results': {k: convert_numpy(v) for k, v in cnn_results.items() if k not in ['model', 'history']},
                'config': self.config,
                'class_names': self.class_names
            }, f, indent=2, default=convert_numpy)
        
        print(f"\nResults saved to: {output_file}")

def main():
    """主函数"""
    # 配置参数
    config = {
        # 路径配置 - 根据服务器实际路径调整
        'dataset_path': '/mnt/acmis_hby/Galaxy-Zoo-Classification/Contrast_experiment/gz2_images_dataset07',
        'zm_output_path': '/mnt/acmis_hby/Galaxy-Zoo-Classification/Contrast_experiment/machine_learning_for_morphological_galaxy_classification/ZMs',
        'output_path': '/mnt/acmis_hby/Galaxy-Zoo-Classification/Contrast_experiment/machine_learning_for_morphological_galaxy_classification/results',
        
        # 数据配置
        'image_size': 224,
        'zernike_order': 45,
        'test_size': 0.2,
        
        # 训练配置
        'batch_size': 64,
        'epochs': 30,
        
        # GPU配置
        'gpu_device': "0"  # 根据您的GPU情况调整
    }
    
    # 创建输出目录
    os.makedirs(config['output_path'], exist_ok=True)
    os.makedirs(config['zm_output_path'], exist_ok=True)
    
    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_device']
    
    # 创建实验对象
    experiment = GalaxyZMExperiment(config)
    
    print("Starting 8-Class Galaxy Classification Experiment...")
    print(f"Configuration: {config}")
    
    try:
        # 步骤1: 计算ZM特征 (如果尚未计算)
        if not all(os.path.exists(os.path.join(config['zm_output_path'], f"{cls}_zms.csv")) 
                  for cls in experiment.class_mapping.values()):
            print("\nStep 1: Computing Zernike Moments...")
            experiment.compute_all_zms()
        else:
            print("\nStep 1: Zernike Moments already computed, skipping...")
        
        # 步骤2: 加载和组织数据
        print("\nStep 2: Loading and organizing data...")
        experiment.load_and_organize_data()
        
        # 步骤3: 训练SVM模型
        print("\nStep 3: Training SVM model...")
        svm_results = experiment.train_svm_model()
        
        # 步骤4: 训练CNN模型
        print("\nStep 4: Training CNN model...")
        cnn_results = experiment.train_cnn_model()
        
        # 步骤5: 保存结果
        print("\nStep 5: Saving results...")
        experiment.save_results(svm_results, cnn_results)
        
        print("\n" + "="*80)
        print("Experiment completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

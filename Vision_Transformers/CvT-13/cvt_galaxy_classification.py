#!/usr/bin/env python3
"""
CVT Galaxy Classification Script
适配CVT模型用于星系分类任务，包含完整的训练和评估功能
"""

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import sys

# 添加lib路径
sys.path.append('lib')
from models import build_model
from config import config, update_config


class GalaxyDataset(Dataset):
    """星系数据集类"""
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 类别名称
        self.class_names = [
            'barred_spirals',
            'cigar_shaped_elliptical',
            'edge_on', 
            'in_between_elliptical',
            'irregular',
            'merger',
            'round_elliptical',
            'unbarred_spirals'
        ]
        
        # 加载所有图像路径和标签
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.exists(class_dir):
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.images.append(os.path.join(class_dir, img_file))
                        self.labels.append(class_idx)
        
        print(f"📁 {data_dir.split('/')[-1]} 数据加载:")
        print(f"   总样本数: {len(self.images)}")
        for i, class_name in enumerate(self.class_names):
            count = sum(1 for label in self.labels if label == i)
            print(f"   {class_name}: {count}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        # 提取galaxy_id（从文件名）
        galaxy_id = os.path.splitext(os.path.basename(img_path))[0]
        
        return image, label, galaxy_id


def create_data_transforms():
    """创建数据变换"""
    # 训练数据变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(180),  # 天文图像可任意旋转
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证/测试数据变换
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform, val_test_transform


def train_model(model, num_epochs, criterion, optimizer, scheduler, 
                train_loader, val_loader, device, model_name,
                print_every=1, early_stop_epochs=15):
    """训练模型"""
    best_model_weights = deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_epoch = -1
    
    # 训练历史记录
    history_dic = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # 训练阶段
        model.train()
        epoch_train_cum_loss = 0.0
        epoch_train_cum_corrects = 0
        
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, pred_classes = torch.max(outputs.detach(), dim=1)
            pred_classes = pred_classes.long()
            
            epoch_train_cum_loss += loss.item() * images.size(0)
            epoch_train_cum_corrects += torch.sum(pred_classes == labels.data).detach().to('cpu').item()
            
            loss.backward()
            optimizer.step()
        
        # 验证阶段
        model.eval()
        epoch_val_cum_loss = 0.0
        epoch_val_cum_corrects = 0
        
        for images, labels, _ in val_loader:
            images = images.to(device)
            labels = labels.long().to(device)
            
            with torch.no_grad():
                outputs = model(images)
                _, pred_classes = torch.max(outputs.detach(), dim=1)
                loss = criterion(outputs, labels)
                
                epoch_val_cum_loss += loss.item() * images.size(0)
                epoch_val_cum_corrects += torch.sum(pred_classes == labels.data).detach().to('cpu').item()
        
        # 计算指标
        train_loss = epoch_train_cum_loss / len(train_loader.dataset)
        train_acc = epoch_train_cum_corrects / len(train_loader.dataset)
        val_loss = epoch_val_cum_loss / len(val_loader.dataset)
        val_acc = epoch_val_cum_corrects / len(val_loader.dataset)
        
        # 更新历史记录
        history_dic['train_loss'].append(train_loss)
        history_dic['train_acc'].append(train_acc)
        history_dic['val_loss'].append(val_loss)
        history_dic['val_acc'].append(val_acc)
        history_dic['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 检查是否是最佳准确率
        if val_acc > best_val_acc:
            best_train_acc = train_acc
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_weights = deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"{model_name}_best.pth")
        
        epoch_end_time = time.time()
        epoch_time_used = epoch_end_time - epoch_start_time
        mm = epoch_time_used // 60
        ss = epoch_time_used % 60
        
        # 打印指标
        if (epoch+1) % print_every == 0:
            if epoch == (best_epoch - 1):
                print(f"Epoch {epoch+1}/{num_epochs}\tTrain loss: {train_loss:.4f}\tTrain acc: {train_acc:.4f}\tVal loss: {val_loss:.4f}\tVal acc: {val_acc:.4f}\tTime: {mm:.0f}m {ss:.0f}s\t<--")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}\tTrain loss: {train_loss:.4f}\tTrain acc: {train_acc:.4f}\tVal loss: {val_loss:.4f}\tVal acc: {val_acc:.4f}\tTime: {mm:.0f}m {ss:.0f}s")
        
        # 早停
        if (epoch+1) - best_epoch >= early_stop_epochs:
            print(f"Early stopping... (Model did not improve after {early_stop_epochs} epochs)")
            break
        
        scheduler.step()
    
    # 加载最佳权重
    model.load_state_dict(best_model_weights)
    print(f"Best epoch = {best_epoch}, with training accuracy = {best_train_acc:.4f} and validation accuracy = {best_val_acc:.4f}")
    
    return model, history_dic


def evaluate_model(model, test_loader, device, class_names):
    """评估模型"""
    model.eval()
    y_true = []
    y_pred = []
    galaxy_ids = []
    
    with torch.no_grad():
        for images, labels, ids in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            galaxy_ids.extend(ids)
    
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
    
    # 详细分类报告
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
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
    
    return results, y_true, y_pred, galaxy_ids


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """绘制混淆矩阵并返回原始数据"""
    # 原始混淆矩阵（未归一化）
    cm_raw = confusion_matrix(y_true, y_pred)
    # 归一化混淆矩阵
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')
    
    # 简化类别名称用于显示
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
    plt.title('Confusion Matrix - CVT Galaxy Classification', fontsize=16)
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
    parser = argparse.ArgumentParser(description='CVT Galaxy Classification')
    parser.add_argument('--cfg', type=str, 
                       default='experiments/galaxy/cvt/cvt-13-galaxy-224x224.yaml',
                       help='Config file path')
    parser.add_argument('--data_path', type=str,
                       default='/mnt/acmis_hby/Paper_experiment_one/baselines_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--test_only', action='store_true', help='Only run testing, skip training')
    parser.add_argument('--model_path', type=str, default='cvt_galaxy_best.pth', help='Path to pre-trained model')
    parser.add_argument('opts', help="Modify config options using the command-line", 
                       default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # 更新配置
    try:
        update_config(config, args)
    except Exception as e:
        print(f"配置更新失败，使用默认配置: {e}")
        # 手动设置关键配置
        config.defrost()
        config.DATASET.ROOT = args.data_path
        config.TRAIN.BATCH_SIZE_PER_GPU = args.batch_size
        config.TRAIN.LR = args.learning_rate
        config.TRAIN.END_EPOCH = args.epochs
        config.freeze()
    
    # 强制设置为8分类
    config.defrost()
    config.MODEL.NUM_CLASSES = 8
    config.freeze()
    
    # 设备设置
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU型号: {torch.cuda.get_device_name(args.gpu)}")
    
    # 数据变换
    train_transform, val_transform, test_transform = create_data_transforms()
    
    # 数据集
    train_data_dir = os.path.join(args.data_path, 'train')
    val_data_dir = os.path.join(args.data_path, 'val') 
    test_data_dir = os.path.join(args.data_path, 'test')
    
    train_dataset = GalaxyDataset(train_data_dir, transform=train_transform)
    val_dataset = GalaxyDataset(val_data_dir, transform=val_transform)
    test_dataset = GalaxyDataset(test_data_dir, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"\n📊 数据集统计:")
    print(f"   训练集: {len(train_dataset)} 样本")
    print(f"   验证集: {len(val_dataset)} 样本") 
    print(f"   测试集: {len(test_dataset)} 样本")
    
    # 建立模型
    model = build_model(config)
    model = model.to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🏗️ 模型信息:")
    print(f"   模型: CVT-13")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    class_names = train_dataset.class_names
    model_name = f"cvt_galaxy_{len(class_names)}_classes"
    
    if args.test_only:
        # 仅测试模式
        print("🔍 仅测试模式...")
        if not os.path.exists(args.model_path):
            print(f"❌ 模型文件不存在: {args.model_path}")
            return
        
        print(f"📁 加载预训练模型: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
    else:
        # 训练模式
        print("🚀 开始训练CVT...")
        
        # 损失函数和优化器（类别权重平衡）
        class_counts = [6320, 5857, 6320, 6320, 4680, 1659, 6320, 6320]  # 训练集各类别样本数
        total_samples = sum(class_counts)
        class_weights = torch.FloatTensor([total_samples/(8*count) for count in class_counts]).to(device)
        print(f"   类别权重: {class_weights.cpu().numpy()}")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.05)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # 训练模型
        model, history_dic = train_model(
            model, args.epochs, criterion, optimizer, scheduler,
            train_loader, val_loader, device, model_name,
            print_every=1, early_stop_epochs=15
        )
        
        # 保存训练历史
        history_df = pd.DataFrame(history_dic)
        history_df.to_csv(f'{model_name}_history.csv', index=False)
        print(f"💾 训练历史保存: {model_name}_history.csv")
    
    # 测试评估
    print("\n📊 开始测试评估...")
    results, y_true, y_pred, galaxy_ids = evaluate_model(model, test_loader, device, class_names)
    
    print("\n🎯 测试结果:")
    print(f"   准确率: {results['overall_metrics']['accuracy']:.4f}")
    print(f"   宏平均 F1: {results['overall_metrics']['f1_macro']:.4f}")
    print(f"   加权平均 F1: {results['overall_metrics']['f1_weighted']:.4f}")
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'galaxy_id': galaxy_ids,
        'true_label': [class_names[i] for i in y_true],
        'predicted_label': [class_names[i] for i in y_pred],
        'correct': [t == p for t, p in zip(y_true, y_pred)]
    })
    predictions_df.to_csv(f'{model_name}_predictions.csv', index=False)
    
    # 绘制混淆矩阵
    cm_raw, cm_normalized = plot_confusion_matrix(y_true, y_pred, class_names, f'{model_name}_confusion_matrix.png')
    
    # 组织最终结果
    final_results = {
        'model_info': {
            'name': 'CVT (Convolutional Vision Transformer)',
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params)
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
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
    
    # 单独保存混淆矩阵原始数据
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
    
    print(f"\n💾 结果已保存:")
    print(f"   预测结果: {model_name}_predictions.csv")
    print(f"   完整结果: {output_filename}")
    print(f"   混淆矩阵图: {model_name}_confusion_matrix.png")
    print(f"   混淆矩阵原始数据: {cm_filename}")
    if not args.test_only:
        print(f"   训练历史: {model_name}_history.csv")
        print(f"   最佳模型: {model_name}_best.pth")
    
    print("\n✨ CVT星系分类任务完成!")


if __name__ == '__main__':
    main()

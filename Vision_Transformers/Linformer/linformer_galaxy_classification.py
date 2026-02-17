#!/usr/bin/env python3
"""
Galaxy Classification with Vision Transformer (Linformer) - Adapted for Your Dataset

**Model**: Vision Transformer with Linformer attention mechanism
**Paper**: Linformer: Self-Attention with Linear Complexity (2020)
**Task**: Galaxy Morphological Classification (8 classes)
**Dataset**: Your galaxy classification dataset

Usage:
python linformer_galaxy_classification.py --epochs 200 --batch_size 64 --gpu 1
python linformer_galaxy_classification.py --test_only --model_path linformer_galaxy_best.pth --gpu 1
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import time
from copy import deepcopy
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import lr_scheduler

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, classification_report
)

# Vision Transformer with Linformer
try:
    from linformer import Linformer
    from vit_pytorch.efficient import ViT
except ImportError:
    print("Please install required packages:")
    print("pip install vit_pytorch linformer")
    exit(1)

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


class GalaxyDataset(Dataset):
    """Galaxy Dataset for folder-based structure"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # 类别名称（与您的数据集匹配）
        self.class_names = [
            'barred_spirals',           # 0
            'cigar_shaped_elliptical',  # 1
            'edge_on',                  # 2
            'in_between_elliptical',    # 3
            'irregular',                # 4
            'merger',                   # 5
            'round_elliptical',         # 6
            'unbarred_spirals'          # 7
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
        
        # 统计各类别样本数
        for class_idx, class_name in enumerate(self.class_names):
            count = self.labels.count(class_idx)
            print(f"   {class_name}: {count}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"❌ 无法加载图像 {img_path}: {e}")
            # 创建一个默认的黑色图像
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # 提取galaxy_id（从文件名）
        galaxy_id = os.path.splitext(os.path.basename(img_path))[0]
        
        return image, label, galaxy_id


def create_data_transforms():
    """
    创建数据变换
    """
    # 训练数据变换（增强的数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(180),  # 天文图像可任意旋转
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),  # 增强颜色变换
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 添加随机裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet标准化
    ])

    # 验证/测试数据变换
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform


def train_model(model, num_epochs, criterion, optimizer, scheduler, train_loader, val_loader, 
                device, model_name, print_every=1, early_stop_epochs=10):
    """
    训练模型
    """
    # 缓存最佳模型
    best_model_weights = deepcopy(model.state_dict())
    best_train_acc = 0.0
    best_val_acc = 0.0
    best_epoch = -1    

    # 记录训练历史
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
            
            pred_logits = model(images)
            loss = criterion(pred_logits, labels)

            _, pred_classes = torch.max(pred_logits.detach(), dim=1)
            pred_classes = pred_classes.long()

            epoch_train_cum_loss += loss.item() * images.size(0)
            epoch_train_cum_corrects += torch.sum(pred_classes==labels.data).detach().to('cpu').item()

            loss.backward()
            optimizer.step()
            
        # 验证阶段
        model.eval()
        epoch_val_cum_loss = 0.0
        epoch_val_cum_corrects = 0

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.long().to(device)

                pred_logits = model(images)
                _, pred_classes = torch.max(pred_logits.detach(), dim=1)
                loss = criterion(pred_logits, labels)

                epoch_val_cum_loss += loss.item() * images.size(0)
                epoch_val_cum_corrects += torch.sum(pred_classes==labels.data).detach().to('cpu').item()

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
        history_dic['lr'].append(scheduler.get_last_lr()[0])

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


def predict_model(model, test_loader, device):
    """
    预测测试数据
    """
    model.eval()
    y_true = []
    y_pred = []
    y_label = []

    with torch.no_grad():
        for images, labels, galaxy_id in test_loader:
            images = images.to(device)
            labels = labels.long().to(device)

            pred_logits = model(images)
            _, pred_classes = torch.max(pred_logits.detach(), dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(pred_classes.cpu().tolist())
            y_label.extend(galaxy_id)
    
    predict_df = pd.DataFrame({
        'GalaxyID': y_label, 
        'class': y_true, 
        'pred': y_pred
    })

    return y_true, y_pred, predict_df


def evaluate_model(y_true, y_pred, class_names):
    """
    评估模型性能
    """
    # 计算各种指标
    acc = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro')
    recall_macro = recall_score(y_true, y_pred, average='macro')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    precision_weighted = precision_score(y_true, y_pred, average='weighted')
    recall_weighted = recall_score(y_true, y_pred, average='weighted')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 各类别准确率
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    results = {
        'overall_metrics': {
            'accuracy': float(acc),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted)
        },
        'per_class_metrics': {
            class_names[i]: {
                'accuracy': float(class_accuracies[i]),
                'support': int(cm.sum(axis=1)[i])
            } for i in range(len(class_names))
        },
        'confusion_matrix': cm.tolist()
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵并返回原始数据
    """
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
    plt.title('Confusion Matrix - Linformer Galaxy Classification', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return cm_raw, cm_normalized


def main():
    parser = argparse.ArgumentParser(description='Linformer Galaxy Classification')
    parser.add_argument('--data_path', type=str, 
                       default='/mnt/acmis_hby/Paper_experiment_one/baselines_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate (优化: 3e-4->1e-4)')
    parser.add_argument('--step_size', type=int, default=10, help='Scheduler step size (优化: 5->10)')
    parser.add_argument('--gamma', type=float, default=0.7, help='Scheduler gamma (优化: 0.9->0.7)')
    parser.add_argument('--gpu', type=int, default=1, help='GPU ID to use')
    parser.add_argument('--test_only', action='store_true', help='Only run testing, skip training')
    parser.add_argument('--model_path', type=str, default='linformer_galaxy_best.pth', help='Path to pre-trained model for testing')
    
    # Linformer/ViT 参数 (保持原架构)
    parser.add_argument('--patch_size', type=int, default=28, help='Patch size')
    parser.add_argument('--depth', type=int, default=12, help='Transformer depth')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--k_dim', type=int, default=64, help='Linformer k dimension')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU型号: {torch.cuda.get_device_name(args.gpu)}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    print(f"🎯 实验配置:")
    print(f"   数据路径: {args.data_path}")
    print(f"   图像大小: 224x224")
    print(f"   批次大小: {args.batch_size}")
    print(f"   类别数: 8")
    if args.test_only:
        print(f"   模式: 仅测试")
        print(f"   模型路径: {args.model_path}")
    else:
        print(f"   学习率: {args.learning_rate}")
        print(f"   最大轮数: {args.epochs}")
    
    # 创建数据变换
    train_transform, val_test_transform = create_data_transforms()
    
    print("🔄 创建数据集...")
    
    if args.test_only:
        # 仅测试模式
        test_dataset = GalaxyDataset(
            os.path.join(args.data_path, 'test'),
            transform=val_test_transform
        )
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"✅ 数据加载器创建完成 (仅测试模式):")
        print(f"   测试样本: {len(test_dataset)} ({len(test_loader)} 批次)")
        
        train_loader = None
        val_loader = None
        
    else:
        # 完整训练模式
        train_dataset = GalaxyDataset(
            os.path.join(args.data_path, 'train'),
            transform=train_transform
        )
        val_dataset = GalaxyDataset(
            os.path.join(args.data_path, 'val'),
            transform=val_test_transform
        )
        test_dataset = GalaxyDataset(
            os.path.join(args.data_path, 'test'),
            transform=val_test_transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"✅ 数据加载器创建完成:")
        print(f"   训练样本: {len(train_dataset)} ({len(train_loader)} 批次)")
        print(f"   验证样本: {len(val_dataset)} ({len(val_loader)} 批次)")
        print(f"   测试样本: {len(test_dataset)} ({len(test_loader)} 批次)")
    
    # 创建模型
    model_name = f"linformer_galaxy_{args.patch_size}_{args.hidden_dim}_{args.depth}"
    
    # 计算序列长度
    seq_len = int((224/args.patch_size)**2) + 1
    
    # Linformer
    lin = Linformer(
        dim=args.hidden_dim, 
        seq_len=seq_len, 
        depth=args.depth, 
        k=args.k_dim, 
        heads=args.num_heads,
        dim_head=None, 
        one_kv_head=False, 
        share_kv=False, 
        reversible=False, 
        dropout=args.dropout
    )
    
    # Vision Transformer
    model = ViT(
        image_size=224, 
        patch_size=args.patch_size, 
        num_classes=8, 
        dim=args.hidden_dim, 
        transformer=lin, 
        pool='cls', 
        channels=3
    ).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  模型架构:")
    print(f"   模型: Linformer + Vision Transformer")
    print(f"   Patch大小: {args.patch_size}")
    print(f"   深度: {args.depth}")
    print(f"   隐藏维度: {args.hidden_dim}")
    print(f"   K维度: {args.k_dim}")
    print(f"   注意力头数: {args.num_heads}")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    if args.test_only:
        # 仅测试模式
        print("🧪 仅测试模式：跳过训练，直接进行测试集评估...")
        
        if not os.path.exists(args.model_path):
            print(f"❌ 模型文件不存在: {args.model_path}")
            return
        
        print(f"📁 加载预训练模型: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
    else:
        # 训练模式
        print("🚀 开始训练Linformer...")
        
        # 损失函数和优化器（添加类别权重平衡）
        # 根据样本数量计算类别权重 (样本少的类别权重高)
        class_counts = [6320, 5857, 6320, 6320, 4680, 1659, 6320, 6320]  # 训练集各类别样本数
        total_samples = sum(class_counts)
        class_weights = torch.FloatTensor([total_samples/(8*count) for count in class_counts]).to(device)
        print(f"   类别权重: {class_weights.cpu().numpy()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)  # 使用AdamW优化器
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # 使用余弦退火调度器
        
        # 训练模型
        model, history_dic = train_model(
            model, args.epochs, criterion, optimizer, scheduler, 
            train_loader, val_loader, device, model_name,
            print_every=1, early_stop_epochs=10
        )
        
        # 保存最终模型
        torch.save(model.state_dict(), f'{model_name}.pth')
        
        # 保存训练历史
        history_df = pd.DataFrame(history_dic)
        history_df.to_csv(f'{model_name}_history.csv', index=False)
    
    # 测试集评估
    print("🧪 开始测试集评估...")
    y_true, y_pred, predict_df = predict_model(model, test_loader, device)
    
    # 保存预测结果
    predict_df.to_csv(f'{model_name}_predictions.csv', index=False)
    
    # 评估结果
    class_names = test_dataset.class_names
    results = evaluate_model(y_true, y_pred, class_names)
    
    print(f"\n🎯 测试结果:")
    print(f"   准确率: {results['overall_metrics']['accuracy']:.4f}")
    print(f"   Macro F1-Score: {results['overall_metrics']['f1_macro']:.4f}")
    print(f"   Weighted F1-Score: {results['overall_metrics']['f1_weighted']:.4f}")
    
    print(f"\n📋 各类别准确率:")
    for class_name, metrics in results['per_class_metrics'].items():
        print(f"   {class_name}: {metrics['accuracy']:.4f} (支持样本: {metrics['support']})")
    
    # 详细分类报告
    print(f"\n📋 详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))
    
    # 绘制混淆矩阵并获取原始数据
    cm_raw, cm_normalized = plot_confusion_matrix(y_true, y_pred, class_names, f'{model_name}_confusion_matrix.png')
    
    # 保存完整结果
    final_results = {
        'model_info': {
            'name': 'Linformer + Vision Transformer',
            'patch_size': args.patch_size,
            'depth': args.depth,
            'hidden_dim': args.hidden_dim,
            'k_dim': args.k_dim,
            'num_heads': args.num_heads,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params
        },
        'training_config': {
            'epochs': args.epochs if not args.test_only else 0,
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
    
    # 保存到JSON
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
    
    print(f"\n✅ Linformer星系分类实验完成!")


if __name__ == "__main__":
    main()

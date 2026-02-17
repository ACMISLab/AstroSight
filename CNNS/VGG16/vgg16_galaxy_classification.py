#!/usr/bin/env python3
"""
VGG16 Galaxy Classification Script
使用预训练VGG16模型进行星系形态分类
基于ImageNet预训练权重，微调用于8类星系分类任务
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class VGG16Classifier(nn.Module):
    """VGG16分类器"""
    def __init__(self, num_classes=8, pretrained=True, freeze_features=False):
        super(VGG16Classifier, self).__init__()
        # 加载预训练VGG16
        self.vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        
        # 可选：冻结特征提取层
        if freeze_features:
            for param in self.vgg16.features.parameters():
                param.requires_grad = False
        
        # 替换分类器
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.vgg16(x)


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    for batch_idx, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += images.size(0)
        
        if (batch_idx + 1) % 50 == 0:
            batch_acc = torch.sum(preds == labels.data).item() / images.size(0)
            print(f"  Batch [{batch_idx + 1}/{len(data_loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {batch_acc:.4f}")
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def validate(model, criterion, data_loader, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()


def test_and_evaluate(model, data_loader, device, class_names):
    """测试并评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n正在进行测试集预测...")
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(class_names)))
    )
    
    # 宏平均和加权平均
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
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
    
    return results, all_labels, all_preds, all_probs


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


def plot_training_history(history, output_path):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # 损失曲线
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_args_parser():
    parser = argparse.ArgumentParser('VGG16 Galaxy Classification', add_help=False)
    
    # Data parameters
    parser.add_argument('--data_path', default='/mnt/acmis_hby/Paper_experiment_one/baselines_dataset',
                        type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./result', type=str, help='output directory')
    
    # Model parameters
    parser.add_argument('--num_classes', default=8, type=int, help='number of classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='use pretrained weights from ImageNet')
    parser.add_argument('--freeze_features', action='store_true', default=False,
                        help='freeze feature extraction layers')
    
    # Training parameters
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--patience', type=int, default=10,
                        help='early stopping patience')
    
    # Other parameters
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--test_only', action='store_true', help='only run testing')
    parser.add_argument('--model_path', type=str, default='', help='path to model for testing')
    
    return parser


def main(args):
    """主函数"""
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*60)
    print("🚀 VGG16 Galaxy Classification")
    print("="*60)
    print(f"使用设备: GPU {args.gpu}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    
    # 设置随机种子
    setup_seed(args.seed)
    cudnn.benchmark = True
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'),
        transform=transform_train
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, 'val'),
        transform=transform_val
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, 'test'),
        transform=transform_val
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print("\n" + "="*60)
    print("📊 数据集统计")
    print("="*60)
    print(f"训练集: {len(train_dataset):,} 样本")
    print(f"验证集: {len(val_dataset):,} 样本")
    print(f"测试集: {len(test_dataset):,} 样本")
    print(f"类别数: {num_classes}")
    print(f"类别名称: {class_names}")
    
    # 创建模型
    model = VGG16Classifier(
        num_classes=num_classes,
        pretrained=args.pretrained,
        freeze_features=args.freeze_features
    )
    model = model.to(device)
    
    # 计算模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("🏗️ 模型信息")
    print("="*60)
    print(f"模型名称: VGG16 (ImageNet Pretrained)")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {total_params - trainable_params:,}")
    print(f"输入尺寸: 224×224×3")
    print(f"预训练权重: {'ImageNet' if args.pretrained else 'None'}")
    print(f"特征层冻结: {'是' if args.freeze_features else '否'}")
    
    model_name = f"vgg16_galaxy_{num_classes}_classes"
    
    if args.test_only:
        # 仅测试模式
        print("\n" + "="*60)
        print("🔍 仅测试模式")
        print("="*60)
        if not args.model_path or not os.path.exists(args.model_path):
            print(f"❌ 模型文件不存在: {args.model_path}")
            return
        
        print(f"📁 加载模型: {args.model_path}")
        checkpoint = torch.load(args.model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # 训练模式
        print("\n" + "="*60)
        print("🚀 开始训练 VGG16")
        print("="*60)
        print(f"训练轮数: {args.epochs}")
        print(f"批次大小: {args.batch_size}")
        print(f"学习率: {args.lr}")
        print(f"权重衰减: {args.weight_decay}")
        print(f"早停耐心: {args.patience} epochs")
        print("="*60)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # 训练循环
        best_val_acc = 0.0
        patience_counter = 0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        start_time = time.time()
        
        for epoch in range(args.epochs):
            print(f"\nEpoch [{epoch + 1}/{args.epochs}]")
            print("-" * 60)
            
            # 训练
            train_loss, train_acc = train_one_epoch(
                model, criterion, optimizer, train_loader, device, epoch
            )
            
            # 验证
            val_loss, val_acc = validate(model, criterion, val_loader, device)
            
            # 更新学习率
            scheduler.step(val_acc)
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"\n训练 - Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"验证 - Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }
                torch.save(checkpoint, output_dir / f'{model_name}_best.pth')
                print(f"✅ 保存最佳模型 (Val Acc: {val_acc:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ 验证准确率未提升 ({patience_counter}/{args.patience})")
                
                if patience_counter >= args.patience:
                    print(f"\n⚠️ 早停触发！最佳验证准确率: {best_val_acc:.4f}")
                    break
        
        training_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ 训练完成")
        print("="*60)
        print(f"总用时: {training_time/60:.1f} 分钟 ({training_time:.0f} 秒)")
        print(f"最佳验证准确率: {best_val_acc:.4f}")
        print(f"最终训练准确率: {history['train_acc'][-1]:.4f}")
        print(f"最终验证准确率: {history['val_acc'][-1]:.4f}")
        
        # 保存训练历史
        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / f'{model_name}_history.csv', index=False)
        print(f"💾 训练历史保存: {model_name}_history.csv")
        
        # 绘制训练曲线
        plot_training_history(history, output_dir / f'{model_name}_training_curves.png')
        print(f"📈 训练曲线保存: {model_name}_training_curves.png")
        
        # 加载最佳模型
        checkpoint = torch.load(output_dir / f'{model_name}_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试评估
    print("\n" + "="*60)
    print("📊 测试集评估")
    print("="*60)
    
    results, y_true, y_pred, y_probs = test_and_evaluate(
        model, test_loader, device, class_names
    )
    
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
        'true_label': [class_names[i] for i in y_true],
        'predicted_label': [class_names[i] for i in y_pred],
        'correct': [t == p for t, p in zip(y_true, y_pred)]
    })
    predictions_df.to_csv(output_dir / f'{model_name}_predictions.csv', index=False)
    
    # 绘制混淆矩阵
    cm_raw, cm_normalized = plot_confusion_matrix(
        y_true, y_pred, class_names,
        output_dir / f'{model_name}_confusion_matrix.png'
    )
    
    # 保存结果文件
    print("\n" + "="*60)
    print("💾 保存结果文件")
    print("="*60)
    
    # 组织最终结果
    final_results = {
        'model_info': {
            'name': 'VGG16 (ImageNet Pretrained)',
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'pretrained': args.pretrained,
            'freeze_features': args.freeze_features
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
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
        print(f"✅ 训练历史: {model_name}_history.csv")
        print(f"✅ 训练曲线: {model_name}_training_curves.png")
        print(f"✅ 最佳模型: {model_name}_best.pth")
    
    print("\n" + "="*60)
    print("✨ VGG16 星系分类任务完成!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('VGG16 Galaxy Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

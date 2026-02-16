#!/usr/bin/env python3
"""
HAN-DCN Galaxy Classification Script
基于HAN-DCN (Hierarchical Attention Network with Deformable CNN) 的星系形态分类
论文: Galaxy Morphological Classification of the Legacy Surveys with Deformable Convolutional Neural Networks
"""

import argparse
import datetime
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, accuracy

import models.han_dcn as han_dcn


def setup_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_args_parser():
    parser = argparse.ArgumentParser('HAN-DCN Galaxy Classification', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    
    # Model parameters
    parser.add_argument('--model', default='resnet_HAN_DCN', type=str,
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--num_classes', default=8, type=int, help='number of classes')
    
    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adam")')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay')
    
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate')
    
    # Dataset parameters
    parser.add_argument('--data_path', 
                        default='/mnt/acmis_hby/Paper_experiment_one/baselines_dataset',
                        type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./result',
                        help='path where to save')
    
    # Device parameters
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID to use')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # Training mode
    parser.add_argument('--test_only', action='store_true', help='Only run testing')
    parser.add_argument('--model_path', type=str, default='han_dcn_galaxy_best.pth',
                        help='Path to pre-trained model for testing')
    parser.add_argument('--early_stop', type=int, default=20,
                        help='Early stopping patience')
    
    return parser


def create_model(num_classes=8):
    """创建HAN-DCN模型"""
    model = han_dcn.resnet_HAN_DCN(num_classes=num_classes)
    return model


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, loss_scaler):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        loss_value = loss.item()
        
        optimizer.zero_grad()
        loss_scaler(loss, optimizer, parameters=model.parameters())
        
        torch.cuda.synchronize()
        
        _, predicted = torch.max(outputs.data, 1)
        total_samples += targets.size(0)
        total_correct += (predicted == targets).sum().item()
        total_loss += loss_value * targets.size(0)
        
        if batch_idx % 50 == 0:
            print(f'  Batch [{batch_idx}/{len(data_loader)}] Loss: {loss_value:.4f}')
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return {'loss': avg_loss, 'accuracy': accuracy}


@torch.no_grad()
def evaluate(data_loader, model, device):
    """评估模型"""
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    all_predictions = []
    all_targets = []
    
    for images, target in data_loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
        
        _, predicted = torch.max(output.data, 1)
        
        total_samples += target.size(0)
        total_correct += (predicted == target).sum().item()
        total_loss += loss.item() * target.size(0)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(target.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }


def compute_metrics(y_true, y_pred, class_names):
    """计算完整评估指标"""
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(class_names)))
    )
    
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
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
    
    return results


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
    plt.title('Confusion Matrix - HAN-DCN Galaxy Classification', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm_raw, cm_normalized


def main(args):
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    print(f"🚀 HAN-DCN Galaxy Classification")
    print(f"   使用设备: GPU {args.gpu}")
    print(f"   PyTorch版本: {torch.__version__}")
    
    # 设置随机种子
    setup_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.CenterCrop(180),
        transforms.Resize(args.input_size),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.CenterCrop(180),
        transforms.Resize(args.input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), 
        transform=transform_train
    )
    dataset_val = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), 
        transform=transform_val
    )
    dataset_test = datasets.ImageFolder(
        os.path.join(args.data_path, "test"), 
        transform=transform_val
    )
    
    # 获取类别信息
    class_names = dataset_train.classes
    num_classes = len(class_names)
    args.num_classes = num_classes
    
    print(f"\n📊 数据集统计:")
    print(f"   训练集: {len(dataset_train)} 样本")
    print(f"   验证集: {len(dataset_val)} 样本")
    print(f"   测试集: {len(dataset_test)} 样本")
    print(f"   类别数: {num_classes}")
    print(f"   类别名称: {class_names}")
    
    # 创建数据加载器
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(1.5 * args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=int(1.5 * args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # 创建模型
    model = create_model(num_classes=num_classes)
    model.to(device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🏗️ 模型信息:")
    print(f"   模型: HAN-DCN (Hierarchical Attention Network + Deformable CNN)")
    print(f"   总参数: {n_parameters:,}")
    print(f"   输入尺寸: {args.input_size}×{args.input_size}")
    
    model_name = f"han_dcn_galaxy_{num_classes}_classes"
    
    if args.test_only:
        # 仅测试模式
        print("\n🔍 仅测试模式...")
        if not os.path.exists(args.model_path):
            print(f"❌ 模型文件不存在: {args.model_path}")
            return
        
        print(f"📁 加载预训练模型: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # 训练模式
        print("\n🚀 开始训练HAN-DCN...")
        
        # 优化器和调度器
        optimizer = create_optimizer(args, model)
        loss_scaler = NativeScaler()
        lr_scheduler, _ = create_scheduler(args, optimizer)
        
        criterion = torch.nn.CrossEntropyLoss()
        
        # 训练历史
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'lr': []
        }
        
        best_accuracy = 0.0
        best_epoch = 0
        no_improve_count = 0
        
        start_time = time.time()
        
        for epoch in range(args.epochs):
            print(f"\nEpoch [{epoch+1}/{args.epochs}]")
            
            # 训练
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler
            )
            
            lr_scheduler.step(epoch)
            current_lr = optimizer.param_groups[0]['lr']
            
            # 验证
            val_stats = evaluate(data_loader_val, model, device)
            
            # 记录历史
            history['train_loss'].append(train_stats['loss'])
            history['train_acc'].append(train_stats['accuracy'])
            history['val_loss'].append(val_stats['loss'])
            history['val_acc'].append(val_stats['accuracy'])
            history['lr'].append(current_lr)
            
            print(f"  Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['accuracy']:.4f}")
            print(f"  Val Loss: {val_stats['loss']:.4f}, Val Acc: {val_stats['accuracy']:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # 保存最佳模型
            if val_stats['accuracy'] > best_accuracy:
                best_accuracy = val_stats['accuracy']
                best_epoch = epoch + 1
                no_improve_count = 0
                
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'best_accuracy': best_accuracy,
                    'args': args,
                }, output_dir / f'{model_name}_best.pth')
                print(f"  ✨ 新的最佳模型! Acc: {best_accuracy:.4f}")
            else:
                no_improve_count += 1
            
            # 早停
            if no_improve_count >= args.early_stop:
                print(f"\n⏹️ 早停: {args.early_stop}轮未改善")
                break
        
        training_time = time.time() - start_time
        print(f"\n⏱️ 训练完成，用时: {training_time/60:.1f}分钟")
        print(f"   最佳Epoch: {best_epoch}, 最佳验证准确率: {best_accuracy:.4f}")
        
        # 保存训练历史
        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / f'{model_name}_history.csv', index=False)
        
        # 加载最佳模型
        checkpoint = torch.load(output_dir / f'{model_name}_best.pth', map_location=device)
        model.load_state_dict(checkpoint['model'])
    
    # 测试评估
    print("\n📊 开始测试评估...")
    test_stats = evaluate(data_loader_test, model, device)
    
    y_true = test_stats['targets']
    y_pred = test_stats['predictions']
    
    # 计算完整指标
    results = compute_metrics(y_true, y_pred, class_names)
    
    print("\n🎯 测试结果:")
    print(f"   准确率: {results['overall_metrics']['accuracy']:.4f}")
    print(f"   宏平均 F1: {results['overall_metrics']['f1_macro']:.4f}")
    print(f"   加权平均 F1: {results['overall_metrics']['f1_weighted']:.4f}")
    
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
    
    # 组织最终结果
    final_results = {
        'model_info': {
            'name': 'HAN-DCN (Hierarchical Attention Network + Deformable CNN)',
            'total_parameters': int(n_parameters),
            'trainable_parameters': int(n_parameters)
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'input_size': args.input_size,
            'optimizer': args.opt,
            'scheduler': args.sched,
            'weight_decay': args.weight_decay
        },
        'results': results,
        'confusion_matrix_raw': cm_raw.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist(),
        'class_names': list(class_names)
    }
    
    # 保存完整结果
    output_filename = f'{model_name}_test_results.json' if args.test_only else f'{model_name}_results.json'
    with open(output_dir / output_filename, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # 保存混淆矩阵数据
    cm_data = {
        'class_names': list(class_names),
        'confusion_matrix_raw': cm_raw.tolist(),
        'confusion_matrix_normalized': cm_normalized.tolist()
    }
    with open(output_dir / f'{model_name}_confusion_matrix_data.json', 'w') as f:
        json.dump(cm_data, f, indent=2)
    
    print(f"\n💾 结果已保存到 {output_dir}/:")
    print(f"   预测结果: {model_name}_predictions.csv")
    print(f"   完整结果: {output_filename}")
    print(f"   混淆矩阵图: {model_name}_confusion_matrix.png")
    print(f"   混淆矩阵数据: {model_name}_confusion_matrix_data.json")
    if not args.test_only:
        print(f"   训练历史: {model_name}_history.csv")
        print(f"   最佳模型: {model_name}_best.pth")
    
    print("\n✨ HAN-DCN星系分类任务完成!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HAN-DCN Galaxy Classification', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)

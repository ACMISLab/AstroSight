#!/usr/bin/env python3
"""
Swin Transformer Galaxy Morphological Attribute Prediction - Fixed Baseline (17 attributes)

**Model**: Microsoft Swin Transformer (Swin-Base-Patch4-Window7-224)  
**Paper**: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (ICCV 2021)
**Task**: Galaxy Morphological Attribute Regression (17 attributes) - FIXED VERSION
**Dataset**: Galaxy Zoo dataset with 17 morphological attributes

This script implements the official Swin Transformer as a baseline for galaxy morphological attribute prediction task.
Fixed from 16 to 17 attributes, matching the complete feature set.

Usage:
python swin_transformer_attribute_regression.py --epochs 50 --batch_size 16 --gpu 0
python swin_transformer_attribute_regression.py --test_only --model_path swin_transformer_attribute_regression_best.pth --gpu 0
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from transformers import SwinModel

import numpy as np
import json
import re
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
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


class GalaxyAttributeDataset(Dataset):
    """星系属性数据集 - 修复为17个特征"""
    
    def __init__(self, jsonl_file, transform=None):
        self.transform = transform
        self.data = []
        
        # ✅ 修复：17个星系形态属性名称
        self.attribute_names = [
            'f_bar/no', 'f_bar/yes', 'f_cigar-shaped', 'f_completelyround',
            'f_disturbed', 'f_dustlane', 'f_edge-on/no', 'f_edge-on/yes',
            'f_features/disk', 'f_in-between', 'f_irregular', 'f_merger',
            'f_odd/no', 'f_odd/yes', 'f_other', 'f_smooth', 'f_spiral/yes'
        ]
        
        # 读取JSONL文件
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_item = json.loads(line.strip())
                    self.data.append(data_item)
        
        print(f"📁 加载数据: {len(self.data)} 个样本")
        print(f"✅ 使用 {len(self.attribute_names)} 个属性 (修复版：16→17)")
        
    def parse_attributes(self, content):
        """从assistant的内容中解析17个属性值"""
        # ✅ 修复：定义17个属性的模式
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
        
        attributes = []
        for attr_name in self.attribute_names:
            match = re.search(patterns[attr_name], content)
            if match:
                value = float(match.group(1))
            else:
                value = 0.0  # 默认值
            attributes.append(value)
        
        return np.array(attributes, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 获取图像路径
        image_path = item['images'][0]  # 取第一个图像路径
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ 无法加载图像 {image_path}: {e}")
            # 创建一个默认的黑色图像
            image = Image.new('RGB', (224, 224), color='black')
        
        # 解析属性
        assistant_content = item['messages'][1]['content']  # assistant的回复
        attributes = self.parse_attributes(assistant_content)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(attributes, dtype=torch.float32)


class SubsetDataset(Dataset):
    """子集数据集，用于训练/验证分割"""
    
    def __init__(self, full_dataset, indices, transform=None):
        self.full_dataset = full_dataset
        self.indices = list(indices)
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # 获取真实的索引
        real_idx = self.indices[idx]
        item = self.full_dataset.data[real_idx]
        
        # 获取图像路径
        image_path = item['images'][0]  # 取第一个图像路径
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ 无法加载图像 {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        # 解析属性
        assistant_content = item['messages'][1]['content']
        attributes = self.full_dataset.parse_attributes(assistant_content)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(attributes, dtype=torch.float32)


class SwinTransformerRegressor(nn.Module):
    """基于Swin Transformer的回归器 - 修复为17个特征"""
    
    def __init__(self, num_attributes=17):  # ✅ 修复：改为17个特征
        super(SwinTransformerRegressor, self).__init__()
        
        # 加载预训练的Swin Transformer backbone (保持原始架构)
        self.swin = SwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window7-224"
        )
        
        # 回归头 (只有这部分是新增的)
        self.regressor = nn.Sequential(
            nn.LayerNorm(1024),  # Swin-Base的hidden size
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_attributes),  # ✅ 修复：输出17个特征
            nn.Sigmoid()  # 输出0-1之间的值
        )
    
    def forward(self, x):
        # 通过Swin Transformer获取特征 (完全使用原始架构)
        outputs = self.swin(x)
        # 使用pooler_output作为全局特征
        features = outputs.pooler_output  # [batch_size, 1024]
        
        # 回归预测
        attributes = self.regressor(features)
        return attributes


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader, desc="训练")):
        images, targets = images.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_predictions.append(outputs.detach().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
    
    # 计算训练MAE
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    train_mae = mean_absolute_error(all_targets, all_predictions)
    
    return total_loss / len(train_loader), train_mae


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc="验证"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 计算指标
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    val_mae = mean_absolute_error(all_targets, all_predictions)
    val_mse = mean_squared_error(all_targets, all_predictions)
    val_r2 = r2_score(all_targets, all_predictions)
    
    return total_loss / len(val_loader), val_mae, val_mse, val_r2


def evaluate_test_set(model, test_loader, device, attribute_names):
    """评估测试集 - 增强版：输出详细的总体指标"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="测试"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # 合并所有预测和目标
    all_predictions = np.concatenate(all_predictions, axis=0)  # (N, 17)
    all_targets = np.concatenate(all_targets, axis=0)          # (N, 17)
    
    # 展平用于详细计算
    pred_flat = all_predictions.flatten()
    label_flat = all_targets.flatten()
    
    # ========== 基础指标 ==========
    mae = mean_absolute_error(label_flat, pred_flat)
    mse = mean_squared_error(label_flat, pred_flat)
    rmse = np.sqrt(mse)
    
    # ========== 多种R²计算方式 ==========
    # 方式1: 整体R² (展平后计算)
    r2_overall = r2_score(label_flat, pred_flat)
    
    # 方式2: 属性平均R² (sklearn默认方式)
    r2_attribute_average = r2_score(all_targets, all_predictions, multioutput='uniform_average')
    
    # 方式3: 样本平均R²
    r2_per_sample = []
    for i in range(all_predictions.shape[0]):
        if np.std(all_targets[i, :]) > 1e-10:
            r2_sample = r2_score(all_targets[i, :], all_predictions[i, :])
            r2_per_sample.append(r2_sample)
    r2_sample_average = np.mean(r2_per_sample) if r2_per_sample else 0.0
    
    # 方式4: 方差加权R²
    attr_variances = np.var(all_targets, axis=0)
    total_variance = np.sum(attr_variances)
    r2_weighted = 0.0
    for i in range(all_predictions.shape[1]):
        if total_variance > 0:
            weight = attr_variances[i] / total_variance
            r2_attr = r2_score(all_targets[:, i], all_predictions[:, i])
            r2_weighted += weight * r2_attr
    
    # 手动验证R²
    ss_res = np.sum((label_flat - pred_flat) ** 2)
    ss_tot = np.sum((label_flat - np.mean(label_flat)) ** 2)
    r2_manual = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # ========== 相关性分析 ==========
    correlation = np.corrcoef(label_flat, pred_flat)[0, 1]
    
    # ========== 误差统计 ==========
    errors = pred_flat - label_flat
    abs_errors = np.abs(errors)
    
    # 百分位数误差
    percentiles = [50, 75, 90, 95, 99]
    percentile_errors = {f'P{p}': float(np.percentile(abs_errors, p)) for p in percentiles}
    
    # 最大误差
    max_error = float(np.max(abs_errors))
    
    # 误差阈值统计
    error_thresholds = [0.05, 0.1, 0.15, 0.2]
    error_within_threshold = {
        f'within_{t}': float(np.mean(abs_errors <= t) * 100)
        for t in error_thresholds
    }
    
    # ========== 整合总体指标 ==========
    overall_metrics = {
        # 基础指标
        'MAE': float(mae),
        'MSE': float(mse),
        'RMSE': float(rmse),
        
        # 多种R²计算方式
        'R2_overall': float(r2_overall),
        'R2_attribute_average': float(r2_attribute_average),
        'R2_sample_average': float(r2_sample_average),
        'R2_variance_weighted': float(r2_weighted),
        'R2_manual_verification': float(r2_manual),
        
        # 相关性
        'pearson_correlation': float(correlation),
        
        # 误差分布
        'max_absolute_error': max_error,
        'median_absolute_error': float(np.median(abs_errors)),
        'std_absolute_error': float(np.std(abs_errors)),
        'percentile_errors': percentile_errors,
        'error_within_threshold_percent': error_within_threshold,
        
        # 数据统计
        'total_predictions': int(len(pred_flat)),
        'num_samples': int(all_predictions.shape[0]),
        'num_attributes': int(all_predictions.shape[1]),
        
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
    
    # ========== 计算每个属性的指标 ==========
    attribute_results = {}
    for i, attr_name in enumerate(attribute_names):
        attr_true = all_targets[:, i]
        attr_pred = all_predictions[:, i]
        
        mae_attr = mean_absolute_error(attr_true, attr_pred)
        mse_attr = mean_squared_error(attr_true, attr_pred)
        r2_attr = r2_score(attr_true, attr_pred)
        rmse_attr = np.sqrt(mse_attr)
        
        attribute_results[attr_name] = {
            'MAE': float(mae_attr),
            'MSE': float(mse_attr),
            'RMSE': float(rmse_attr),
            'R2': float(r2_attr),
            'mean_prediction': float(np.mean(attr_pred)),
            'mean_label': float(np.mean(attr_true)),
            'std_prediction': float(np.std(attr_pred)),
            'std_label': float(np.std(attr_true))
        }
    
    # 为了向后兼容，返回旧的格式 + 新的详细指标
    return overall_metrics, attribute_results


def main():
    parser = argparse.ArgumentParser(description='Swin Transformer Galaxy Attribute Prediction (Fixed 17 attributes)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.01, help='Validation split ratio')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (e.g., 0, 1). If None, auto-detect')
    parser.add_argument('--test_only', action='store_true', help='Only run testing, skip training')
    parser.add_argument('--model_path', type=str, default='swin_transformer_attribute_regression_best.pth', help='Path to pre-trained model for testing')
    
    args = parser.parse_args()
    
    # GPU设备选择
    if args.gpu is not None:
        if torch.cuda.is_available() and args.gpu < torch.cuda.device_count():
            device = torch.device(f'cuda:{args.gpu}')
            print(f"🚀 Using specified GPU: {args.gpu}")
        else:
            print(f"❌ GPU {args.gpu} not available. Available GPUs: {torch.cuda.device_count()}")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"🚀 使用设备: {device}")
    if torch.cuda.is_available():
        print(f"   GPU型号: {torch.cuda.get_device_name()}")
        print(f"   GPU内存: {torch.cuda.get_device_properties(device.index if device.type == 'cuda' else 0).total_memory / 1024**3:.1f}GB")
    
    
    DATA_PATH = "/mnt/acmis_hby/Paper_experiment_one/galaxy_attributes"
    
    # 实验参数
    IMG_SIZE = 224
    NUM_ATTRIBUTES = 17  # ✅ 修复：17个形态属性
    
    print(f"🎯 实验配置:")
    print(f"   数据路径: {DATA_PATH}")
    print(f"   图像大小: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   属性数量: {NUM_ATTRIBUTES}")
    if args.test_only:
        print(f"   模式: 仅测试")
        print(f"   模型路径: {args.model_path}")
    else:
        print(f"   学习率: {args.learning_rate}")
        print(f"   最大轮数: {args.epochs}")
        print(f"   验证集比例: {args.val_split}")
    
    # ✅ 修复：使用你Jupyter notebook中的数据增强配置
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # 使用你notebook中的设置
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("🔄 创建数据集...")
    
    if args.test_only:
        # 仅测试模式：只加载测试数据集和一个小的训练集用于获取属性名称
        full_train_dataset = GalaxyAttributeDataset(
            os.path.join(DATA_PATH, 'train.jsonl'),
            transform=None  # 暂时不应用变换
        )
        
        test_dataset = GalaxyAttributeDataset(
            os.path.join(DATA_PATH, 'test.jsonl'),
            transform=val_test_transform
        )
        
        # 创建数据加载器
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"✅ 数据加载器创建完成 (仅测试模式):")
        print(f"   测试样本: {len(test_dataset)} ({len(test_loader)} 批次)")
        
        # 设置为None，避免后续代码出错
        train_loader = None
        val_loader = None
        train_dataset = None
        val_dataset = None
        
    else:
        # 完整训练模式
        # 加载完整训练数据
        full_train_dataset = GalaxyAttributeDataset(
            os.path.join(DATA_PATH, 'train.jsonl'),
            transform=None  # 暂时不应用变换
        )
        
        # 分割训练集和验证集
        train_size = int((1 - args.val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size
        
        # 使用固定种子确保可重现
        generator = torch.Generator().manual_seed(42)
        train_indices, val_indices = random_split(range(len(full_train_dataset)), 
                                                [train_size, val_size], 
                                                generator=generator)
        
        # 创建训练、验证和测试数据集
        train_dataset = SubsetDataset(full_train_dataset, train_indices, transform=train_transform)
        val_dataset = SubsetDataset(full_train_dataset, val_indices, transform=val_test_transform)
        
        test_dataset = GalaxyAttributeDataset(
            os.path.join(DATA_PATH, 'test.jsonl'),
            transform=val_test_transform
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        print(f"✅ 数据加载器创建完成:")
        print(f"   训练样本: {len(train_dataset)} ({len(train_loader)} 批次)")
        print(f"   验证样本: {len(val_dataset)} ({len(val_loader)} 批次)")
        print(f"   测试样本: {len(test_dataset)} ({len(test_loader)} 批次)")
    
    # 创建模型
    model = SwinTransformerRegressor(num_attributes=NUM_ATTRIBUTES).to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🏗️  模型架构:")
    print(f"   模型: Swin Transformer Base (官方预训练)")
    print(f"   输入尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   输出属性: {NUM_ATTRIBUTES}")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    
    if not args.test_only:
        # 优化器和调度器（仅训练模式需要）
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        criterion = nn.MSELoss()
        
        print(f"⚙️  训练配置:")
        print(f"   优化器: AdamW (lr={args.learning_rate}, wd={args.weight_decay})")
        print(f"   调度器: CosineAnnealingLR")
        print(f"   损失函数: MSELoss")
    else:
        # 仅测试模式，不需要优化器
        optimizer = None
        scheduler = None
        criterion = nn.MSELoss()  # 仍需要criterion用于测试
    
    if args.test_only:
        # 仅测试模式：跳过训练，直接加载模型
        print("🧪 仅测试模式：跳过训练，直接进行测试集评估...")
        print("="*60)
        
        # 检查模型文件是否存在
        if not os.path.exists(args.model_path):
            print(f"❌ 模型文件不存在: {args.model_path}")
            return
        
        print(f"📁 加载预训练模型: {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        
        # 设置默认值
        best_val_mae = 0.0  # 仅测试模式下无验证MAE
        epoch = 0  # 无训练轮数
        
    else:
        # 训练循环
        print("🚀 开始训练Swin Transformer...")
        print("="*60)
        
        best_val_mae = float('inf')
        patience = 10
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_maes = []
        
        for epoch in range(args.epochs):
            print(f"📅 Epoch {epoch+1}/{args.epochs}")
            
            # 训练
            train_loss, train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
            
            # 验证
            val_loss, val_mae, val_mse, val_r2 = validate_epoch(model, val_loader, criterion, device)
            
            # 更新学习率
            scheduler.step()
            
            # 记录指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_maes.append(val_mae)
            
            print(f"   训练 - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
            print(f"   验证 - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
            print(f"   学习率: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 早停和模型保存
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_counter = 0
                torch.save(model.state_dict(), '/mnt/acmis_hby/Paper_experiment_one/Transformer_experiemnt/Swin_transformer/results/swin_transformer_attribute_regression_best.pth')
                print(f"   ✅ 新的最佳模型! MAE: {best_val_mae:.4f}")
            else:
                patience_counter += 1
                print(f"   ⏳ 等待改进: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"🛑 早停触发! 最佳验证MAE: {best_val_mae:.4f}")
                break
        
        print(f"🎯 训练完成! 最佳验证MAE: {best_val_mae:.4f}")
        print(f"📁 最佳模型已保存: swin_transformer_attribute_regression_best.pth")
        
        # 加载最佳模型用于测试
        model.load_state_dict(torch.load('/mnt/acmis_hby/Paper_experiment_one/Transformer_experiemnt/Swin_transformer/results/swin_transformer_attribute_regression_best.pth'))
    
    # 测试集评估
    print("🧪 开始测试集评估...")
    print("="*50)
    
    # 评估测试集（新版返回详细指标）
    overall_metrics, attribute_results = evaluate_test_set(
        model, test_loader, device, full_train_dataset.attribute_names
    )
    
    print(f"🎯 测试集整体结果:")
    print(f"   MAE: {overall_metrics['MAE']:.4f}")
    print(f"   MSE: {overall_metrics['MSE']:.4f}")
    print(f"   RMSE: {overall_metrics['RMSE']:.4f}")
    print(f"\n📊 多种R²计算方式:")
    print(f"   整体R² (Overall):           {overall_metrics['R2_overall']:.4f}")
    print(f"   属性平均R² (Attr-Avg):      {overall_metrics['R2_attribute_average']:.4f}")
    print(f"   样本平均R² (Sample-Avg):    {overall_metrics['R2_sample_average']:.4f}")
    print(f"   方差加权R² (Var-Weighted):  {overall_metrics['R2_variance_weighted']:.4f}")
    print(f"\n🔗 相关性: {overall_metrics['pearson_correlation']:.4f}")
    print(f"� 误差统计:")
    print(f"   中位数误差: {overall_metrics['median_absolute_error']:.4f}")
    print(f"   最大误差: {overall_metrics['max_absolute_error']:.4f}")
    print(f"   误差≤0.1: {overall_metrics['error_within_threshold_percent']['within_0.1']:.2f}%")
    
    print(f"\n� 各属性详细结果:")
    for attr_name, metrics in attribute_results.items():
        print(f"   {attr_name:<20}: MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")
    
    # 保存完整结果
    results = {
        'model_info': {
            'name': 'Swin Transformer (Fixed)',
            'version': 'Swin-Base-Patch4-Window7-224',
            'paper': 'Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (ICCV 2021)',
            'pretrained': 'microsoft/swin-base-patch4-window7-224',
            'total_parameters': total_params,
            'task': 'Galaxy Morphological Attribute Regression',
            'num_attributes': NUM_ATTRIBUTES
        },
        'training_config': {
            'epochs': epoch + 1,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'image_size': IMG_SIZE,
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'loss_function': 'MSELoss',
            'validation_split': args.val_split
        },
        'overall_metrics': overall_metrics,
        'per_attribute_metrics': attribute_results,
        'training_results': {
            'best_val_mae': float(best_val_mae) if not args.test_only else None,
            'epochs_trained': epoch + 1 if not args.test_only else 0,
            'test_only_mode': args.test_only,
            'model_path': args.model_path if args.test_only else None
        },
        'r2_calculation_methods': {
            'description': 'Multiple R² calculation methods for comprehensive evaluation',
            'methods': {
                'overall_r2': {
                    'value': overall_metrics['R2_overall'],
                    'method': 'Flatten all samples×attributes and compute R²',
                    'use_case': 'Standard ML evaluation'
                },
                'attribute_average_r2': {
                    'value': overall_metrics['R2_attribute_average'],
                    'method': 'Compute R² for each attribute separately, then average',
                    'use_case': 'Equal weight per attribute (sklearn default)'
                },
                'sample_average_r2': {
                    'value': overall_metrics['R2_sample_average'],
                    'method': 'Compute R² for each sample separately, then average',
                    'use_case': 'Per-sample performance'
                },
                'variance_weighted_r2': {
                    'value': overall_metrics['R2_variance_weighted'],
                    'method': 'Weight each attribute R² by its variance',
                    'use_case': 'Emphasize high-variance attributes'
                }
            },
            'recommendation': 'Use overall_r2 for paper reporting, attribute_average_r2 was the old default'
        }
    }
    
    # 保存到JSON
    output_filename = '/mnt/acmis_hby/Paper_experiment_one/Transformer_experiemnt/Swin_transformer/results/swin_transformer_detailed_evaluation.json'
    with open(output_filename, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    
    print("=" * 80)
    print("🎯 FIXED SWIN TRANSFORMER 属性回归实验结果总结")
    print("=" * 80)
    print(f"📋 模型信息:")
    print(f"   模型: {results['model_info']['version']}")
    print(f"   论文: {results['model_info']['paper']}")
    print(f"   预训练: {results['model_info']['pretrained']}")
    print(f"   参数量: {results['model_info']['total_parameters']:,}")
    print(f"   任务: {results['model_info']['task']} ({NUM_ATTRIBUTES} attributes)")
    
    print(f"📊 整体性能指标:")
    print(f"   测试集MAE: {overall_metrics['MAE']:.4f}")
    print(f"   测试集MSE: {overall_metrics['MSE']:.4f}")
    print(f"   测试集RMSE: {overall_metrics['RMSE']:.4f}")
    print(f"   整体R² (Overall): {overall_metrics['R2_overall']:.4f}")
    print(f"   属性平均R² (旧版): {overall_metrics['R2_attribute_average']:.4f}")
    if not args.test_only:
        print(f"   最佳验证MAE: {best_val_mae:.4f}")
        print(f"   训练轮数: {epoch + 1}")
    else:
        print(f"   模式: 仅测试")
        print(f"   使用模型: {args.model_path}")
    
    print(f"📝 论文对比表格格式:")
    print(f"% 使用整体R² (推荐)")
    print(f"Swin Transformer & {overall_metrics['MAE']:.4f} & {overall_metrics['MSE']:.4f} & {overall_metrics['R2_overall']:.4f} \\\\")
    print(f"% 使用属性平均R² (旧版)")
    print(f"Swin Transformer & {overall_metrics['MAE']:.4f} & {overall_metrics['MSE']:.4f} & {overall_metrics['R2_attribute_average']:.4f} \\\\")
    print(f"\n📊 R²对比:")
    print(f"   整体R² = {overall_metrics['R2_overall']:.4f} (展平后计算)")
    print(f"   属性平均R² = {overall_metrics['R2_attribute_average']:.4f} (sklearn默认)")
    print(f"   差值 = {overall_metrics['R2_overall'] - overall_metrics['R2_attribute_average']:.4f}")
    
    print(f"📁 结果文件:")
    if args.test_only:
        print(f"   详细结果: swin_transformer_attribute_regression_result.json")
        print(f"   使用模型: {args.model_path}")
    else:
        print(f"   详细结果: swin_transformer_attribute_regression_result.json")
        print(f"   最佳模型: swin_transformer_attribute_regression_best.pth")
    
    print(f"✅ Fixed Swin Transformer基线实验完成!")
    print(f"   可直接用于与AstroSight模型对比")
    print(f"   所有17个属性的详细指标已保存")


if __name__ == "__main__":
    main()

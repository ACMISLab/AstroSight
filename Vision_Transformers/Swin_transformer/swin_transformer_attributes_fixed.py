#!/usr/bin/env python3
"""
Swin Transformer Galaxy Morphological Attribute Prediction - Baseline (17 attributes)

**Model**: Microsoft Swin Transformer (Swin-Base-Patch4-Window7-224)  
**Paper**: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows (ICCV 2021)
**Task**: Galaxy Morphological Attribute Regression (17 attributes) - FIXED VERSION
**Dataset**: Galaxy Zoo dataset with 17 morphological attributes

This script implements the official Swin Transformer as a baseline for galaxy morphological attribute prediction task.
Fixed from 16 to 17 attributes, matching the complete feature set.

Usage:
python swin_transformer_attributes_fixed.py --epochs 50 --batch_size 16 --gpu 0
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
    
    def __init__(self, jsonl_file, transform=None):
        self.transform = transform
        self.data = []
        
        self.attribute_names = [
            'f_bar/no', 'f_bar/yes', 'f_cigar-shaped', 'f_completelyround',
            'f_disturbed', 'f_dustlane', 'f_edge-on/no', 'f_edge-on/yes',
            'f_features/disk', 'f_in-between', 'f_irregular', 'f_merger',
            'f_odd/no', 'f_odd/yes', 'f_other', 'f_smooth', 'f_spiral/yes'
        ]
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_item = json.loads(line.strip())
                    self.data.append(data_item)
        
        print(f"📁 : {len(self.data)} ")
        
    def parse_attributes(self, content):
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
                value = 0.0  
            attributes.append(value)
        
        return np.array(attributes, dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        image_path = item['images'][0]  
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌  {image_path}: {e}")
            # 创建一个默认的黑色图像
            image = Image.new('RGB', (224, 224), color='black')
        
        assistant_content = item['messages'][1]['content']  
        attributes = self.parse_attributes(assistant_content)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(attributes, dtype=torch.float32)


class SubsetDataset(Dataset):
    
    def __init__(self, full_dataset, indices, transform=None):
        self.full_dataset = full_dataset
        self.indices = list(indices)
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        item = self.full_dataset.data[real_idx]
        
        image_path = item['images'][0]  
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌  {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        assistant_content = item['messages'][1]['content']
        attributes = self.full_dataset.parse_attributes(assistant_content)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(attributes, dtype=torch.float32)


class SwinTransformerRegressor(nn.Module):
    
    def __init__(self, num_attributes=17):  
        super(SwinTransformerRegressor, self).__init__()
        
        # Swin Transformer backbone 
        self.swin = SwinModel.from_pretrained(
            "microsoft/swin-base-patch4-window7-224"
        )
        
        self.regressor = nn.Sequential(
            nn.LayerNorm(1024),  
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_attributes),  
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        outputs = self.swin(x)
        features = outputs.pooler_output  
        
        attributes = self.regressor(features)
        return attributes


def train_epoch(model, train_loader, optimizer, criterion, device):
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
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    train_mae = mean_absolute_error(all_targets, all_predictions)
    
    return total_loss / len(train_loader), train_mae


def validate_epoch(model, val_loader, criterion, device):
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
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    val_mae = mean_absolute_error(all_targets, all_predictions)
    val_mse = mean_squared_error(all_targets, all_predictions)
    val_r2 = r2_score(all_targets, all_predictions)
    
    return total_loss / len(val_loader), val_mae, val_mse, val_r2


def evaluate_test_set(model, test_loader, device, attribute_names):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="测试"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    overall_mae = mean_absolute_error(all_targets, all_predictions)
    overall_mse = mean_squared_error(all_targets, all_predictions)
    overall_r2 = r2_score(all_targets, all_predictions)
    
    attribute_results = {}
    for i, attr_name in enumerate(attribute_names):
        attr_true = all_targets[:, i]
        attr_pred = all_predictions[:, i]
        
        mae = mean_absolute_error(attr_true, attr_pred)
        mse = mean_squared_error(attr_true, attr_pred)
        r2 = r2_score(attr_true, attr_pred)
        
        attribute_results[attr_name] = {
            'MAE': mae,
            'MSE': mse,
            'R2': r2
        }
    
    return overall_mae, overall_mse, overall_r2, attribute_results


def main():
    parser = argparse.ArgumentParser(description='Swin Transformer Galaxy Attribute Prediction (Fixed 17 attributes)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--val_split', type=float, default=0.01, help='Validation split ratio')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (e.g., 0, 1). If None, auto-detect')
    
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
    
    print(f"{device}")
    if torch.cuda.is_available():
        print(f"{torch.cuda.get_device_name()}")
        print(f"{torch.cuda.get_device_properties(device.index if device.type == 'cuda' else 0).total_memory / 1024**3:.1f}GB")
    
    DATA_PATH = "/mnt/acmis_hby/galaxy_contranst/galaxy_attributes"
    
    IMG_SIZE = 224
    NUM_ATTRIBUTES = 17  
    
    print(f"🎯 实验配置:")
    print(f"   数据路径: {DATA_PATH}")
    print(f"   图像大小: {IMG_SIZE}x{IMG_SIZE}")
    print(f"   批次大小: {args.batch_size}")
    print(f"   属性数量: {NUM_ATTRIBUTES}")
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
    full_train_dataset = GalaxyAttributeDataset(
        os.path.join(DATA_PATH, 'train.jsonl'),
        transform=None  
    )
    
    # 分割训练集和验证集
    train_size = int((1 - args.val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
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
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    print(f"⚙️  训练配置:")
    print(f"   优化器: AdamW (lr={args.learning_rate}, wd={args.weight_decay})")
    print(f"   调度器: CosineAnnealingLR")
    print(f"   损失函数: MSELoss")
    
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
            torch.save(model.state_dict(), '/mnt/acmis_hby/galaxy_contranst/swin_transformer_attribute_regression_best_fixed.pth')
            print(f"   ✅ 新的最佳模型! MAE: {best_val_mae:.4f}")
        else:
            patience_counter += 1
            print(f"   ⏳ 等待改进: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"🛑 早停触发! 最佳验证MAE: {best_val_mae:.4f}")
            break
    
    print(f"🎯 训练完成! 最佳验证MAE: {best_val_mae:.4f}")
    print(f"📁 最佳模型已保存: swin_transformer_attribute_regression_best_fixed.pth")
    
    # 测试集评估
    print("🧪 开始测试集评估...")
    print("="*50)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('/mnt/acmis_hby/galaxy_contranst/swin_transformer_attribute_regression_best_fixed.pth'))
    
    # 评估测试集
    overall_mae, overall_mse, overall_r2, attribute_results = evaluate_test_set(
        model, test_loader, device, full_train_dataset.attribute_names
    )
    
    print(f"🎯 测试集整体结果:")
    print(f"   整体MAE: {overall_mae:.4f}")
    print(f"   整体MSE: {overall_mse:.4f}")
    print(f"   整体R²: {overall_r2:.4f}")
    
    print(f"📋 各属性详细结果:")
    for attr_name, metrics in attribute_results.items():
        print(f"   {attr_name:<20}: MAE={metrics['MAE']:.4f}, MSE={metrics['MSE']:.4f}, R²={metrics['R2']:.4f}")
    
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
        'results': {
            'overall_mae': float(overall_mae),
            'overall_mse': float(overall_mse),
            'overall_r2': float(overall_r2),
            'best_val_mae': float(best_val_mae),
            'epochs_trained': epoch + 1
        },
        'attribute_results': {
            attr_name: {
                'MAE': float(metrics['MAE']),
                'MSE': float(metrics['MSE']),
                'R2': float(metrics['R2'])
            } for attr_name, metrics in attribute_results.items()
        }
    }
    
    # 保存到JSON
    with open('/mnt/acmis_hby/galaxy_contranst/swin_transformer_attribute_regression_results_fixed.json', 'w') as f:
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
    print(f"   测试集MAE: {overall_mae:.4f}")
    print(f"   测试集MSE: {overall_mse:.4f}")
    print(f"   测试集R²: {overall_r2:.4f}")
    print(f"   最佳验证MAE: {best_val_mae:.4f}")
    print(f"   训练轮数: {epoch + 1}")
    
    print(f"📝 论文对比表格格式:")
    print(f"Swin Transformer & {overall_mae:.4f} & {overall_mse:.4f} & {overall_r2:.4f} \\\\")
    
    print(f"📁 结果文件:")
    print(f"   详细结果: swin_transformer_attribute_regression_results_fixed.json")
    print(f"   最佳模型: swin_transformer_attribute_regression_best_fixed.pth")
    
    print(f"✅ Fixed Swin Transformer基线实验完成!")
    print(f"   可直接用于与AstroSight模型对比")
    print(f"   所有17个属性的详细指标已保存")


if __name__ == "__main__":
    main()

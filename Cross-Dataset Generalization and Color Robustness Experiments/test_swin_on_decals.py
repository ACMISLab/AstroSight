#!/usr/bin/env python3
"""
测试Swin Transformer在DECaLS数据集上的性能
支持彩色和灰度图像测试
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SwinForImageClassification
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import argparse

# 类别映射：DECaLS option (A-H) -> Swin class index (0-7)
OPTION_TO_CLASS = {
    'A': 6,  # Round Elliptical -> round_elliptical
    'B': 3,  # In-between Elliptical -> in_between_elliptical
    'C': 1,  # Cigar-shaped Elliptical -> cigar_shaped_elliptical
    'D': 2,  # Edge-on -> edge_on
    'E': 0,  # Barred Spirals -> barred_spirals
    'F': 7,  # Unbarred Spirals -> unbarred_spirals
    'G': 4,  # Irregular -> irregular
    'H': 5,  # Merger -> merger
}

CLASS_NAMES = [
    'barred_spirals',           # 0
    'cigar_shaped_elliptical',  # 1
    'edge_on',                  # 2
    'in_between_elliptical',    # 3
    'irregular',                # 4
    'merger',                   # 5
    'round_elliptical',         # 6
    'unbarred_spirals'          # 7
]


class DECaLSDataset(Dataset):
    """DECaLS JSONL格式数据集"""
    
    def __init__(self, jsonl_path, transform=None):
        self.transform = transform
        self.samples = []
        
        # 读取JSONL
        with open(jsonl_path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                
                # 提取图像路径
                image_path = entry['images'][0]
                
                # 提取标签（从assistant的回复中）
                response = entry['messages'][1]['content']
                option = response.split('option ')[1].split(' is')[0].strip()
                label = OPTION_TO_CLASS[option]
                
                self.samples.append({
                    'image_path': image_path,
                    'label': label,
                    'option': option
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']


def get_transforms():
    """获取数据预处理transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(checkpoint_path, device):
    """加载训练好的Swin Transformer模型"""
    print(f"Loading model from: {checkpoint_path}")
    
    # 创建模型
    model = SwinForImageClassification.from_pretrained(
        'microsoft/swin-base-patch4-window7-224',
        num_labels=8,
        ignore_mismatched_sizes=True
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded successfully")
    return model


def test_model(model, dataloader, device):
    """测试模型"""
    all_preds = []
    all_labels = []
    
    print("\nTesting...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(images)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels)


def compute_metrics(preds, labels, class_names):
    """计算评估指标"""
    # 准确率
    accuracy = accuracy_score(labels, preds) * 100
    
    # 分类报告
    report = classification_report(
        labels, preds,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def print_results(metrics, class_names):
    """打印结果"""
    print("\n" + "="*80)
    print("Test Results")
    print("="*80)
    
    print(f"\nOverall Accuracy: {metrics['accuracy']:.2f}%")
    
    print(f"\nPer-Class Metrics:")
    print("-"*80)
    print(f"{'Class':<30} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*80)
    
    report = metrics['classification_report']
    for class_name in class_names:
        if class_name in report:
            m = report[class_name]
            print(f"{class_name:<30} {m['precision']*100:>10.2f}%  {m['recall']*100:>10.2f}%  "
                  f"{m['f1-score']*100:>10.2f}%  {m['support']:>8.0f}")
    
    # Macro平均
    macro = report['macro avg']
    print("-"*80)
    print(f"{'Macro Average':<30} {macro['precision']*100:>10.2f}%  {macro['recall']*100:>10.2f}%  "
          f"{macro['f1-score']*100:>10.2f}%")
    
    print("\n" + "="*80)


def save_results(metrics, output_path):
    """保存结果"""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test Swin Transformer on DECaLS')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--jsonl', type=str, required=True,
                        help='Path to test JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for testing')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("Swin Transformer Testing on DECaLS")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.jsonl}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    # 加载数据
    print("\nLoading dataset...")
    transform = get_transforms()
    test_dataset = DECaLSDataset(args.jsonl, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"✓ Loaded {len(test_dataset)} samples")
    
    # 加载模型
    model = load_model(args.checkpoint, device)
    
    # 测试
    preds, labels = test_model(model, test_loader, device)
    
    # 计算指标
    metrics = compute_metrics(preds, labels, CLASS_NAMES)
    
    # 打印结果
    print_results(metrics, CLASS_NAMES)
    
    # 保存结果
    save_results(metrics, args.output)
    
    print("\n✓ Testing completed!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Zernike Moments特征提取脚本
从星系图像中提取Zernike Moments特征用于SVM分类
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ZEMO import zemo


def extract_zernike_features(image_path, image_size, zernike_order, zbfstr):
    """
    从单张图像提取Zernike Moments特征
    
    Args:
        image_path: 图像路径
        image_size: 调整后的图像尺寸
        zernike_order: Zernike阶数
        zbfstr: 预计算的Zernike基函数
    
    Returns:
        Zernike Moments特征向量
    """
    # 读取图像
    image = cv2.imread(image_path)
    
    # 检查图像是否成功读取
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 调整大小到Zernike计算标准尺寸
    resized_image = cv2.resize(image, (image_size, image_size))
    
    # 使用R通道（天文图像常用）
    red_channel = resized_image[:, :, 2]
    
    # 计算Zernike Moments
    zms = np.abs(zemo.zernike_mom(np.array(red_channel), zbfstr))
    
    return zms


def process_dataset(data_path, image_size, zernike_order, output_path):
    """
    处理整个数据集，提取所有图像的Zernike特征
    
    Args:
        data_path: 数据集根路径
        image_size: 图像尺寸
        zernike_order: Zernike阶数
        output_path: 输出路径
    """
    print("="*60)
    print("🔬 Zernike Moments 特征提取")
    print("="*60)
    print(f"数据路径: {data_path}")
    print(f"原始图像尺寸: 224×224 (自动检测)")
    print(f"Zernike计算尺寸: {image_size}×{image_size}")
    print(f"Zernike阶数: {zernike_order}")
    print(f"特征维度: {(zernike_order + 1) * (zernike_order + 2) // 2}")
    print(f"\n说明: 图像将从224×224自动调整到{image_size}×{image_size}用于ZMs计算")
    
    # 预计算Zernike基函数（提升速度）
    print("\n预计算Zernike基函数...")
    zbfstr = zemo.zernike_bf(image_size, zernike_order, 1)
    print("✓ 基函数计算完成")
    
    # 获取类别列表
    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_path = os.path.join(data_path, split)
        if not os.path.exists(split_path):
            print(f"⚠️ 跳过不存在的split: {split}")
            continue
        
        print(f"\n{'='*60}")
        print(f"处理 {split.upper()} 集")
        print(f"{'='*60}")
        
        class_names = sorted([d for d in os.listdir(split_path) 
                            if os.path.isdir(os.path.join(split_path, d))])
        
        all_features = []
        all_labels = []
        all_paths = []
        
        for class_idx, class_name in enumerate(class_names):
            class_path = os.path.join(split_path, class_name)
            image_files = [f for f in os.listdir(class_path) 
                          if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"\n处理类别: {class_name} ({len(image_files)} 张图像)")
            
            for img_file in tqdm(image_files, desc=f"  提取特征"):
                img_path = os.path.join(class_path, img_file)
                
                try:
                    # 提取Zernike特征
                    features = extract_zernike_features(
                        img_path, image_size, zernike_order, zbfstr
                    )
                    
                    all_features.append(features)
                    all_labels.append(class_idx)
                    all_paths.append(img_path)
                    
                except Exception as e:
                    print(f"  ⚠️ 处理失败: {img_file} - {e}")
                    continue
        
        # 保存特征
        features_array = np.array(all_features)
        labels_array = np.array(all_labels)
        
        output_file = os.path.join(output_path, f'{split}_zernike_features.npz')
        np.savez_compressed(
            output_file,
            features=features_array,
            labels=labels_array,
            paths=all_paths,
            class_names=class_names
        )
        
        print(f"\n✓ {split.upper()} 特征已保存:")
        print(f"  文件: {output_file}")
        print(f"  特征形状: {features_array.shape}")
        print(f"  标签形状: {labels_array.shape}")
        print(f"  类别数: {len(class_names)}")
        print(f"  类别名称: {class_names}")
    
    print("\n" + "="*60)
    print("✨ 所有特征提取完成!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='提取Zernike Moments特征')
    
    parser.add_argument('--data_path', type=str,
                       default='/mnt/acmis_hby/Paper_experiment_one/baselines_dataset',
                       help='数据集路径')
    parser.add_argument('--output_path', type=str,
                       default='./features',
                       help='特征输出路径')
    parser.add_argument('--image_size', type=int, default=200,
                       help='图像尺寸 (default: 200)')
    parser.add_argument('--zernike_order', type=int, default=45,
                       help='Zernike阶数 (default: 45)')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 提取特征
    process_dataset(
        args.data_path,
        args.image_size,
        args.zernike_order,
        args.output_path
    )


if __name__ == '__main__':
    main()

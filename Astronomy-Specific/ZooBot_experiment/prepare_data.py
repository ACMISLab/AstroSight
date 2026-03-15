#!/usr/bin/env python3
"""
准备ZooBot所需的数据格式
从baselines_dataset转换为ZooBot的CSV格式
"""

import os
import pandas as pd
import glob
from pathlib import Path

# 路径配置
BASELINES_DIR = "/mnt/acmis_hby/Paper_experiment_one/Datasets/baselines_dataset"
OUTPUT_DIR = "/mnt/acmis_hby/Paper_experiment_one/ZooBot_experiments/data"

# 类别映射
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


def create_csv_catalog(split='train'):
    """
    创建ZooBot所需的CSV目录文件
    
    ZooBot需要的格式:
    file_loc,label
    /path/to/image1.jpg,0
    /path/to/image2.jpg,1
    """
    print(f"\nProcessing {split} split...")
    
    data_list = []
    split_dir = os.path.join(BASELINES_DIR, split)
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(split_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} not found, skipping...")
            continue
        
        # 获取所有图像
        image_files = glob.glob(os.path.join(class_dir, '*.jpg'))
        
        for img_path in image_files:
            data_list.append({
                'file_loc': img_path,
                'label': class_idx
            })
        
        print(f"  {class_name}: {len(image_files)} images")
    
    # 创建DataFrame
    df = pd.DataFrame(data_list)
    
    # 保存CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f'{split}_catalog.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Saved {len(df)} samples to {csv_path}")
    
    return df


def verify_data():
    """验证数据准备是否正确"""
    print("\n" + "="*70)
    print("Data Verification")
    print("="*70)
    
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(OUTPUT_DIR, f'{split}_catalog.csv')
        
        if not os.path.exists(csv_path):
            print(f"✗ {split}_catalog.csv not found!")
            continue
        
        df = pd.read_csv(csv_path)
        
        print(f"\n{split.upper()} Split:")
        print(f"  Total samples: {len(df)}")
        print(f"  Label distribution:")
        
        label_counts = df['label'].value_counts().sort_index()
        for label, count in label_counts.items():
            print(f"    Class {label} ({CLASS_NAMES[label]}): {count}")
        
        # 检查文件是否存在
        missing_files = 0
        for idx, row in df.head(10).iterrows():
            if not os.path.exists(row['file_loc']):
                missing_files += 1
        
        if missing_files > 0:
            print(f"  ✗ Warning: {missing_files}/10 sample files not found!")
        else:
            print(f"  ✓ All sample files exist")
    
    print("\n" + "="*70)


def main():
    """主函数"""
    print("="*70)
    print("ZooBot Data Preparation")
    print("="*70)
    print(f"Source: {BASELINES_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # 处理三个数据集
    for split in ['train', 'val', 'test']:
        create_csv_catalog(split)
    
    # 验证数据
    verify_data()
    
    print("\n✓ Data preparation complete!")
    print(f"\nGenerated files:")
    print(f"  {OUTPUT_DIR}/train_catalog.csv")
    print(f"  {OUTPUT_DIR}/val_catalog.csv")
    print(f"  {OUTPUT_DIR}/test_catalog.csv")


if __name__ == "__main__":
    main()

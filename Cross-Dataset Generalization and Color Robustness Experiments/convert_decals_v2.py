#!/usr/bin/env python3
"""
将DECaLS测试集转换为GZ2格式 - 使用datasets库
"""

import os
import json
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# 路径配置
DECALS_DIR = "/mnt/acmis_hby/LLM/GalaxyZooDECaLS"
OUTPUT_DIR = "/mnt/acmis_hby/Paper_experiment_one/experiments_for_revision/decals_test_gz2_format"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "test_images")
OUTPUT_JSONL = os.path.join(OUTPUT_DIR, "test.jsonl")

# 类别映射
LABEL_TO_OPTION = {
    0: "A", 1: "B", 2: "C", 3: "D",
    4: "E", 5: "F", 6: "G", 7: "H",
}

# 提示词模板
PROMPT = "<image>[Output Constraints] Return the answer as one of the following options: 'A:round elliptical', 'B:in-between elliptical', 'C:cigar-shaped elliptical', 'D:edge-on', 'E:Barred spirals', 'F:Unbarred spirals', 'G:Irregular', 'H:merger'  Now, based on the morphological image, return the choice."

def convert_dataset(grayscale=False):
    """转换数据集"""
    # 设置输出路径
    if grayscale:
        output_dir = OUTPUT_DIR + "_grayscale"
    else:
        output_dir = OUTPUT_DIR
    
    output_images_dir = os.path.join(output_dir, "test_images")
    output_jsonl = os.path.join(output_dir, "test.jsonl")
    
    os.makedirs(output_images_dir, exist_ok=True)
    
    print(f"Loading dataset from: {DECALS_DIR}")
    dataset = load_dataset(DECALS_DIR, split='test')
    
    print(f"Total samples: {len(dataset)}")
    print(f"Grayscale mode: {grayscale}")
    
    # 统计标签
    from collections import Counter
    label_counts = Counter(dataset['label'])
    print("\nLabel distribution:")
    for label in sorted(label_counts.keys()):
        print(f"  Label {label} (Option {LABEL_TO_OPTION[label]}): {label_counts[label]} samples")
    
    # 转换
    jsonl_data = []
    print(f"\nConverting {len(dataset)} samples...")
    
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        image = sample['image']  # PIL Image
        label = sample['label']
        option = LABEL_TO_OPTION[label]
        
        # 处理图像
        if grayscale:
            gray = image.convert('L')
            image_to_save = gray.convert('RGB')
        else:
            image_to_save = image
        
        # 保存图像
        image_filename = f"decals_{idx:06d}.jpg"
        image_path = os.path.join(output_images_dir, image_filename)
        image_to_save.save(image_path, 'JPEG', quality=95)
        
        # 构建JSONL条目
        entry = {
            "messages": [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": f"Answer: Based on the image information, option {option} is selected."}
            ],
            "images": [image_path]
        }
        jsonl_data.append(entry)
    
    # 保存JSONL
    print(f"\nSaving to: {output_jsonl}")
    with open(output_jsonl, 'w') as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + '\n')
    
    # 保存统计
    stats = {
        'total_samples': len(jsonl_data),
        'grayscale': grayscale,
        'label_distribution': {LABEL_TO_OPTION[k]: v for k, v in label_counts.items()}
    }
    with open(os.path.join(output_dir, "statistics.json"), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Conversion Complete!")
    print(f"{'='*60}")
    print(f"Output: {output_dir}")
    print(f"Images: {output_images_dir}")
    print(f"JSONL: {output_jsonl}")
    print(f"Total: {len(jsonl_data)} samples")

def main():
    print("="*60)
    print("DECaLS to GZ2 Format Converter v2")
    print("="*60)
    
    # 彩色版本
    print("\n[1/2] Converting Color Version...")
    convert_dataset(grayscale=False)
    
    # 灰度版本
    print("\n[2/2] Converting Grayscale Version...")
    convert_dataset(grayscale=True)
    
    print("\n" + "="*60)
    print("All Done!")
    print("="*60)

if __name__ == "__main__":
    main()

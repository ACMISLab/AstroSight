# Swin Transformer Galaxy Baseline

Baseline implementations of Swin Transformer for galaxy morphological classification and attribute regression.

## Overview

This repository contains two Swin Transformer experiments:

### 1. Galaxy Classification (8 Classes)
Classifies galaxies into morphological categories:
- `barred_spirals`, `cigar_shaped_elliptical`, `edge_on`, `in_between_elliptical`
- `irregular`, `merger`, `round_elliptical`, `unbarred_spirals`

### 2. Attribute Regression (17 Attributes)
Predicts continuous morphological attributes:
- `f_bar/no`, `f_bar/yes`, `f_cigar-shaped`, `f_completelyround`
- `f_disturbed`, `f_dustlane`, `f_edge-on/no`, `f_edge-on/yes`
- `f_features/disk`, `f_in-between`, `f_irregular`, `f_merger`
- `f_odd/no`, `f_odd/yes`, `f_other`, `f_smooth`, `f_spiral/yes`

## Model Architecture

- **Model**: Swin Transformer (microsoft/swin-base-patch4-window7-224)
- **Parameters**: ~88M (pre-trained backbone)
- **Input Size**: 224×224
- **Tasks**: Classification (8 classes) + Regression (17 attributes)
- **Output**: Softmax (classification) / Sigmoid (regression)

## Key Files

- `swin_transformer_attribute_regression.py` - Main training and evaluation script
- `Swin_Transformer_Galaxy_Classification.ipynb` - Jupyter notebook implementation
- `swin_transformer_attribute_regression_best.pth` - Pre-trained model weights
- `results/` - Training results and evaluation metrics

## Quick Start

### Requirements
```bash
pip install torch torchvision
pip install transformers
pip install scikit-learn matplotlib seaborn pandas
```

### Dataset Structure
**For Classification**: Folder structure
```
baselines_dataset/
├── train/
│   ├── barred_spirals/
│   └── ...
├── val/
└── test/
```

**For Attribute Regression**: JSONL format
```json
{
  "messages": [...],
  "images": ["/path/to/galaxy_image.jpg"]
}
```

### Training

**Classification Task**:
```bash
# Use Jupyter notebook: Swin_Transformer_Galaxy_Classification.ipynb
```

**Attribute Regression Task**:
```bash
# Basic training
python swin_transformer_attribute_regression.py \
    --train_file /path/to/train.jsonl \
    --val_file /path/to/val.jsonl \
    --test_file /path/to/test.jsonl \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --gpu 0

# Testing only
python swin_transformer_attribute_regression.py \
    --test_only \
    --model_path swin_transformer_attribute_regression_best.pth \
    --test_file /path/to/test.jsonl
```

## Features

- **Pre-trained Backbone**: Uses microsoft/swin-base-patch4-window7-224
- **Dual Tasks**: Classification (8 classes) + Regression (17 attributes)
- **Data Augmentation**: Astronomy-specific augmentations
- **Model Checkpointing**: Saves best model weights
- **Flexible Input**: Supports both folder structure and JSONL format

## Output Files

**Classification Task**:
- Model weights and evaluation results via Jupyter notebook

**Attribute Regression Task**:
- `swin_transformer_attribute_regression_best.pth` - Best model weights
- `swin_transformer_attribute_regression_result.json` - Complete evaluation results
- `swin_transformer_attribute_regression_test_result.json` - Test-only results

## Evaluation Metrics

**Classification**: Accuracy, Precision, Recall, F1-score
**Regression**: MAE, MSE, R² score per attribute

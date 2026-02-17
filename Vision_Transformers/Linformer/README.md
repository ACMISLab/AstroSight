# Linformer Galaxy Classification Baseline

A baseline implementation of Linformer (Linear Transformer) for galaxy morphological classification.

## Overview

This repository contains the implementation of Linformer model for classifying galaxies into 8 morphological categories:
- `barred_spirals`
- `cigar_shaped_elliptical` 
- `edge_on`
- `in_between_elliptical`
- `irregular`
- `merger`
- `round_elliptical`
- `unbarred_spirals`

## Model Architecture

- **Model**: Linformer + Vision Transformer
- **Parameters**: ~2.8M
- **Input Size**: 224×224
- **Patch Size**: 28×28
- **Hidden Dimension**: 128
- **Linformer K Dimension**: 64
- **Classes**: 8
- **Original code** :https://github.com/soliao/Galaxy-Zoo-Classification

## Key Files

- `linformer_galaxy_classification.py` - Main training and evaluation script
- `linformer_galaxy_28_128_12_best.pth` - Pre-trained model weights
- `result/` - Training results and evaluation metrics

## Quick Start

### Requirements
```bash
pip install torch torchvision
pip install vit-pytorch linformer
pip install scikit-learn matplotlib seaborn pandas
```

### Dataset Structure
```
baselines_dataset/
├── train/
│   ├── barred_spirals/
│   ├── cigar_shaped_elliptical/
│   └── ...
├── val/
└── test/
```

### Training
```bash
# Basic training
python linformer_galaxy_classification.py \
    --data_path /path/to/baselines_dataset \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --gpu 0

# Background training
nohup python linformer_galaxy_classification.py \
    --data_path /path/to/baselines_dataset \
    --epochs 200 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --gpu 0 \
    > linformer_training.log 2>&1 &
```

### Testing Only
```bash
python linformer_galaxy_classification.py \
    --test_only \
    --model_path linformer_galaxy_28_128_12_best.pth \
    --data_path /path/to/baselines_dataset
```

## Features

- **Linear Attention**: Efficient O(n) complexity instead of O(n²)
- **Class Imbalance Handling**: Automatic class weight calculation
- **Data Augmentation**: Astronomy-specific augmentations (rotation, flipping)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score (macro/weighted)
- **Visualization**: Confusion matrix plots
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model weights

## Files

**Main Code**:
- `linformer_galaxy_classification.py` - Training and evaluation script

**Training Results** (in `result/` folder):
- `linformer_galaxy_28_128_12_best.pth` - Best model weights
- `linformer_galaxy_28_128_12_results.json` - Complete evaluation results
- `linformer_galaxy_28_128_12_history.csv` - Training history
- `linformer_galaxy_28_128_12_predictions.csv` - Detailed predictions
- `linformer_galaxy_28_128_12_confusion_matrix.png` - Confusion matrix plot
- `linformer_galaxy_28_128_12_confusion_matrix_data.json` - Raw confusion matrix data

**Training Log**:
- `linformer_training.log` - Complete training log

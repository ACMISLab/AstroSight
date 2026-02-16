# HAN-DCN Galaxy Classification Baseline

A baseline implementation of HAN-DCN (Hierarchical Attention Network with Deformable CNN) for galaxy morphological classification.

**Paper**: [Galaxy Morphological Classification of the Legacy Surveys with Deformable Convolutional Neural Networks](https://iopscience.iop.org/article/10.3847/1538-3881/ad10ab)

## Overview

This repository contains the implementation of HAN-DCN model for classifying galaxies into 8 morphological categories:
- `barred_spirals`
- `cigar_shaped_elliptical` 
- `edge_on`
- `in_between_elliptical`
- `irregular`
- `merger`
- `round_elliptical`
- `unbarred_spirals`

## Model Architecture

- **Model**: HAN-DCN (ResNet + Deformable Conv + Layer Attention)
- **Parameters**: ~1.2M
- **Key Features**: 
  - DCNv2 (Deformable Convolution v2)
  - LAM (Layer Attention Module)
  - Residual connections
- **Classes**: 8

## Key Files

- `han_dcn_galaxy_classification.py` - Main training and evaluation script
- `dis_main_train.py` - Original training script（https://github.com/kustcn/legacy_galaxy）
- `models/han_dcn.py` - Model architecture
- `engine.py` - Training/evaluation engine
- `utils.py` - Utility functions

## Quick Start

### Requirements
```bash
pip install torch torchvision
pip install timm
pip install scikit-learn matplotlib seaborn pandas numpy
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
cd /mnt/acmis_hby/Paper_experiment_one/legacy_galaxy-master

# Basic training
python han_dcn_galaxy_classification.py \
    --data_path /path/to/baselines_dataset \
    --epochs 100 \
    --batch_size 64 \
    --lr 5e-4 \
    --gpu 0

# Background training
nohup python han_dcn_galaxy_classification.py \
    --data_path /path/to/baselines_dataset \
    --epochs 100 \
    --batch_size 64 \
    --lr 5e-4 \
    --gpu 0 \
    > han_dcn_training.log 2>&1 &
```

### Testing Only
```bash
python han_dcn_galaxy_classification.py \
    --test_only \
    --model_path result/han_dcn_galaxy_8_classes_best.pth \
    --data_path /path/to/baselines_dataset
```

## Features

- **Deformable Convolution**: Adaptive receptive field for irregular galaxy shapes
- **Layer Attention Module**: Hierarchical feature fusion
- **Data Augmentation**: Astronomy-specific augmentations (rotation, flipping, color jitter)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score (macro/weighted)
- **Visualization**: Confusion matrix plots
- **Early Stopping**: Prevents overfitting
- **Cosine Learning Rate Scheduling**: Smooth learning rate decay

## Output Files

After training, the following files are generated in `result/`:
- `han_dcn_galaxy_8_classes_best.pth` - Best model weights
- `han_dcn_galaxy_8_classes_results.json` - Complete evaluation results
- `han_dcn_galaxy_8_classes_history.csv` - Training history
- `han_dcn_galaxy_8_classes_predictions.csv` - Detailed predictions
- `han_dcn_galaxy_8_classes_confusion_matrix.png` - Confusion matrix plot
- `han_dcn_galaxy_8_classes_confusion_matrix_data.json` - Raw confusion matrix data


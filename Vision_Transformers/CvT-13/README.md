# CVT Galaxy Classification Baseline

A baseline implementation of Convolutional Vision Transformer (CVT) for galaxy morphological classification.

## Overview

This repository contains the implementation of CVT-13 model for classifying galaxies into 8 morphological categories:
- `barred_spirals`
- `cigar_shaped_elliptical` 
- `edge_on`
- `in_between_elliptical`
- `irregular`
- `merger`
- `round_elliptical`
- `unbarred_spirals`

## Model Architecture

- **Model**: CVT-13 (Convolutional Vision Transformer)
- **Parameters**: ~19.6M
- **Input Size**: 224×224
- **Classes**: 8

## Key Files

### Core Implementation
- `cvt_galaxy_classification.py` - Main training and evaluation script
- `experiments/galaxy/cvt/cvt-13-galaxy-224x224.yaml` - Model configuration

### CVT Library (Original:https://github.com/C-JIe123/Galaxy-Morphology)
- `lib/models/` - CVT model implementations
- `lib/config/` - Configuration system
- `lib/dataset/` - Data loading utilities

## Quick Start

### Requirements
```bash
pip install torch torchvision
pip install timm yacs tensorboardX
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
python cvt_galaxy_classification.py \
    --cfg experiments/galaxy/cvt/cvt-13-galaxy-224x224.yaml \
    --data_path /path/to/baselines_dataset \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --gpu 0

# Background training
nohup python cvt_galaxy_classification.py \
    --cfg experiments/galaxy/cvt/cvt-13-galaxy-224x224.yaml \
    --data_path /path/to/baselines_dataset \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --gpu 0 \
    > cvt_galaxy_training.log 2>&1 &
```

### Testing Only
```bash
python cvt_galaxy_classification.py \
    --test_only \
    --model_path cvt_galaxy_8_classes_best.pth \
    --data_path /path/to/baselines_dataset
```

## Features

- **Class Imbalance Handling**: Automatic class weight calculation
- **Data Augmentation**: Astronomy-specific augmentations (rotation, flipping)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score (macro/weighted)
- **Visualization**: Confusion matrix plots
- **Early Stopping**: Prevents overfitting
- **Model Checkpointing**: Saves best model weights

## Files

**Main Code**:
- `cvt_galaxy_classification.py` - Training and evaluation script
- `experiments/galaxy/cvt/cvt-13-galaxy-224x224.yaml` - Model configuration

**Training Results** (in `result/` folder):
- `cvt_galaxy_8_classes_best.pth` - Best model weights
- `cvt_galaxy_8_classes_results.json` - Complete evaluation results
- `cvt_galaxy_8_classes_history.csv` - Training history
- `cvt_galaxy_8_classes_predictions.csv` - Detailed predictions
- `cvt_galaxy_8_classes_confusion_matrix.png` - Confusion matrix plot
- `cvt_galaxy_8_classes_confusion_matrix_data.json` - Raw confusion matrix data

**Training Log**:
- `cvt_galaxy_training.log` - Complete training log


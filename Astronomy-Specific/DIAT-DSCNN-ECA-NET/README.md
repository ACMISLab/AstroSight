# ECA-CNN Galaxy Classification Baseline

A baseline implementation of ECA-CNN (DIAT-DSCNN-ECA-Net) for galaxy morphological classification using Separable Convolution and Efficient Channel Attention mechanism.

## Overview

This repository contains the implementation of ECA-CNN model for classifying galaxies into 8 morphological categories:
- `barred_spirals`
- `cigar_shaped_elliptical` 
- `edge_on`
- `in_between_elliptical`
- `irregular`
- `merger`
- `round_elliptical`
- `unbarred_spirals`

## Model Architecture

- **Model**: ECA-CNN (DIAT-DSCNN-ECA-Net)
- **Parameters**: ~133K (very lightweight)
- **Input Size**: 224×224
- **Key Features**: 
  - Separable Convolution layers
  - Efficient Channel Attention (ECA) modules
  - Multi-scale feature fusion
  - Residual connections
- **Classes**: 8

## Key Files

- `eca_cnn_galaxy_classification.py` - Main training and evaluation script
- `Galaxy_Classification_Using_ECA_Attention_Mechanism_(1)_(1)_(2).ipynb` - Original Jupyter notebook（https://github.com/ajaywagh007/Galaxy-Classification-Using-CNN.）

## Quick Start

### Requirements
```bash
pip install tensorflow
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
# Basic training
python eca_cnn_galaxy_classification.py \
    --data_path /path/to/baselines_dataset \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --gpu 0

# Background training
nohup python eca_cnn_galaxy_classification.py \
    --data_path /path/to/baselines_dataset \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --gpu 0 \
    > eca_cnn_training.log 2>&1 &
```

### Testing Only
```bash
python eca_cnn_galaxy_classification.py \
    --test_only \
    --model_path eca_cnn_galaxy_8_classes_best.h5 \
    --data_path /path/to/baselines_dataset
```

## Features

- **Lightweight Architecture**: Only ~133K parameters, very efficient
- **Multi-scale Processing**: 3×3, 5×5, 7×7 separable convolutions
- **Efficient Channel Attention**: ECA modules for better feature representation
- **Class Imbalance Handling**: Comprehensive data augmentation
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score (macro/weighted)
- **Visualization**: Confusion matrix plots
- **Early Stopping**: Prevents overfitting
- **Learning Rate Scheduling**: Adaptive learning rate decay

## Output Files

After training, the following files are generated:
-/result/   
- `eca_cnn_galaxy_8_classes_best.h5` - Best model weights
- `eca_cnn_galaxy_8_classes_results.json` - Complete evaluation results
- `eca_cnn_galaxy_8_classes_history.csv` - Training history
- `eca_cnn_galaxy_8_classes_predictions.csv` - Detailed predictions
- `eca_cnn_galaxy_8_classes_confusion_matrix.png` - Confusion matrix plot
- `eca_cnn_galaxy_8_classes_confusion_matrix_data.json` - Raw confusion matrix data

## Configuration

Key parameters:
- `--epochs 50` - Training epochs
- `--batch_size 32` - Training batch size
- `--learning_rate 0.01` - Initial learning rate (with scheduling)
- `--image_size 224` - Input image size
- Automatic learning rate decay: 0.01 → 0.002 → 0.001 → 0.0001

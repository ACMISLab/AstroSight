# VGG16 Galaxy Classification Baseline

A baseline implementation of VGG16 for galaxy morphological classification using transfer learning from ImageNet pretrained weights.

## Overview

This repository contains the VGG16 baseline model for classifying galaxies into 8 morphological categories:
- `barred_spirals`
- `cigar_shaped_elliptical` 
- `edge_on`
- `in_between_elliptical`
- `irregular`
- `merger`
- `round_elliptical`
- `unbarred_spirals`

## Model Architecture

- **Model**: VGG16 (16-layer Network)
- **Parameters**: ~138M total
- **Input Size**: 224×224×3
- **Pretrained**: ImageNet weights
- **Classes**: 8

## Key Files

- `vgg16_galaxy_classification.py` - Main training and evaluation script

## Quick Start

### Requirements
```bash
pip install torch torchvision
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
python vgg16_galaxy_classification.py \
    --data_path /path/to/baselines_dataset \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --gpu 0

# Background training
nohup python vgg16_galaxy_classification.py \
    --data_path /path/to/baselines_dataset \
    --epochs 50 \
    --batch_size 32 \
    --lr 1e-4 \
    --gpu 0 \
    > vgg16_training.log 2>&1 &
```

### Testing Only
```bash
python vgg16_galaxy_classification.py \
    --test_only \
    --model_path ./result/vgg16_galaxy_8_classes_best.pth \
    --data_path /path/to/baselines_dataset
```

## Features

- **Transfer Learning**: ImageNet pretrained weights
- **Data Augmentation**: RandomCrop, Flip, Rotation, ColorJitter
- **Optimization**: Adam optimizer with ReduceLROnPlateau scheduler
- **Early Stopping**: Prevents overfitting (patience: 10)
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-score (macro/weighted)
- **Visualization**: Confusion matrix and training curves

## Output Files

After training, the following files are generated in `./result/`:
- `vgg16_galaxy_8_classes_best.pth` - Best model checkpoint
- `vgg16_galaxy_8_classes_results.json` - Complete evaluation results
- `vgg16_galaxy_8_classes_predictions.csv` - Detailed predictions
- `vgg16_galaxy_8_classes_confusion_matrix.png` - Confusion matrix plot
- `vgg16_galaxy_8_classes_training_curves.png` - Training history plot
- `vgg16_galaxy_8_classes_history.csv` - Epoch-by-epoch metrics

## Configuration

Key parameters:
- `--epochs 50` - Training epochs
- `--batch_size 32` - Training batch size
- `--lr 1e-4` - Learning rate
- `--freeze_features` - Freeze convolutional layers (optional)
- `--patience 10` - Early stopping patience

## Citation

```bibtex
@article{simonyan2014very,
  title={Very deep convolutional networks for large-scale image recognition},
  author={Simonyan, Karen and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1409.1556},
  year={2014}
}
```

---

**Note**: VGG16 serves as a strong baseline due to its proven performance and pretrained ImageNet features, making it an excellent comparison point for custom galaxy classification architectures.

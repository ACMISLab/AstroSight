# CNN Baseline Experiments for Galaxy Classification

This repository contains four CNN baseline models for galaxy morphology classification using transfer learning from ImageNet pretrained weights.

## Overview

We evaluate four classic CNN architectures on the 8-class galaxy morphology classification task:
- **VGG16** - Classic deep CNN (ICLR 2015)
- **ResNet-50** - Deep residual network (CVPR 2016)
- **DenseNet-121** - Densely connected network (CVPR 2017)
- **EfficientNet-B0** - Efficient architecture (ICML 2019)

## Directory Structure

```
CNN_experiment/
├── VGG16/
│   ├── vgg16_galaxy_classification.py
│   ├── result/
│   │   └── vgg16_galaxy_8_classes_results.json
│   └── README.md
├── resnet50_densnet121/
│   ├── resent50_densenet121_experiment.ipynb
│   ├── resnet50_results/
│   │   └── resnet50_metrics.json
│   └── densenet121_result/
│       └── densenet121_report.json
├── Efficientb0/
│   ├── EfficientNet-B0_experiment.ipynb
│   └── Effficientb0_result/
│       └── efficientnet_b0_reports/
└── README.md
```

## Dataset

**Galaxy Zoo 2 Dataset** - 8 morphological classes:
1. **Barred Spirals** - Spiral galaxies with bar structure
2. **Cigar-shaped Elliptical** - Elongated elliptical galaxies
3. **Edge-on** - Galaxies viewed from the side
4. **In-between Elliptical** - Intermediate elliptical shape
5. **Irregular** - Irregular morphology
6. **Merger** - Interacting/merging galaxies
7. **Round Elliptical** - Spherical elliptical galaxies
8. **Unbarred Spirals** - Spiral galaxies without bar

**Dataset Split:**
- Training: 43,796 samples
- Validation: 554 samples
- Test: 11,090 samples

## Model Details

### VGG16
- **Architecture**: 16 layers (13 conv + 3 FC)
- **Parameters**: 134.3M
- **Pretrained**: ImageNet
- **Accuracy**: 77.06%
- **Training**: 50 epochs, batch size 32, lr=1e-4

### ResNet-50
- **Architecture**: 50 layers with residual connections
- **Parameters**: 23.5M
- **Pretrained**: ImageNet
- **Accuracy**: 77.66%
- **Key Feature**: Skip connections for better gradient flow

### DenseNet-121
- **Architecture**: 121 layers with dense connections
- **Parameters**: 7.6M
- **Pretrained**: ImageNet
- **Accuracy**: 79.84% ⭐ Best
- **Key Feature**: Feature reuse through dense connectivity

### EfficientNet-B0
- **Architecture**: Compound scaling (depth + width + resolution)
- **Parameters**: 4.0M ⭐ Most efficient
- **Pretrained**: ImageNet
- **Accuracy**: 77.71%
- **Key Feature**: Balanced scaling for efficiency

## Usage

### VGG16
```bash
cd VGG16
python vgg16_galaxy_classification.py \
    --data_path /path/to/dataset \
    --epochs 50 \
    --batch_size 32 \
    --gpu 0
```

### ResNet-50 & DenseNet-121
```bash
cd resnet50_densnet121
jupyter notebook resent50_densenet121_experiment.ipynb
```

### EfficientNet-B0
```bash
cd Efficientb0
jupyter notebook EfficientNet-B0_experiment.ipynb
```

## Requirements

```bash
pip install torch torchvision
pip install scikit-learn matplotlib seaborn pandas numpy
pip install jupyter notebook
```

## Results Files

Each model directory contains:
- **Model weights**: `.pth` or `.pt` files
- **Metrics**: JSON files with accuracy, precision, recall, F1-score
- **Confusion matrices**: Visualization and raw data
- **Classification reports**: Per-class performance metrics
- **Predictions**: CSV files with detailed predictions

## Citation

If you use these baselines, please cite the original papers:

**Note**: All models use ImageNet pretrained weights and are fine-tuned on the Galaxy Zoo 2 dataset. DenseNet-121 achieves the best performance while EfficientNet-B0 offers the best parameter efficiency.

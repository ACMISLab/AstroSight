# Cross-Dataset Generalization & Color Robustness Experiments

Experiments for evaluating AstroSight models on Gz2 grayscale images and Galaxy Zoo DECaLS dataset to address reviewer concerns about color-morphology bias and dataset generalization.

## Overview

**Purpose:** Evaluate cross-dataset generalization and color robustness  
**Datasets:** GZ2 test set (11,088 galaxies), Galaxy Zoo DECaLS (31,191 galaxies) 
[https://drive.google.com/drive/folders/1-JklU4-9JuBDgdjnD7wCKCO02rTaeVT2?usp=sharing]
**Models Tested:** All 6 AstroSight variants + Swin Transformer baseline

## Experiments Conducted

### 1. Color Robustness on GZ2
- **Dataset:** GZ2 test set converted to grayscale
- **Results:** `GZ2_results/`

### 2. Cross-Dataset Generalization (DECaLS)
- **Dataset:** Galaxy Zoo DECaLS test set (color + grayscale)
- **Results:** `GalaxyZooDECaLS_results/`

### 3. Swin Transformer Baseline
- **Goal:** Compare color robustness with best-performing CNN/ViT baseline
- **Results:** `swin_decals_results/`


## Directory Structure

```
experiments_for_revision/
├── README.md                           # This file                    
│
├── Data Preparation Scripts
│   ├── convert_gz2_to_grayscale.py    # Convert GZ2 to grayscale
│   ├── convert_GZ_decals.py           # Convert DECaLS to GZ2 format
├── Testing Scripts
│   ├── test_swin_on_decals.py         # Test Swin on DECaLS(color and grayscale)
│   ├── test_swin_on_gz2_grayscale.sh  # Test Swin on gz2 grayscale
│
├── Analysis Scripts
│   ├── analyze_inference_results.py   # Main analysis tool
│   └── analyze_swin_results.py        # Swin-specific analysis
│
├── Test Datasets
│   ├── gz2_test_grayscale/            # GZ2 grayscale test set
│   ├── decals_test_gz2_format/        # DECaLS color test set
│   └── decals_test_gz2_format_grayscale/ # DECaLS grayscale test set
│
└── Results
    ├── GZ2_results/                    # GZ2 grayscale results
    ├── GalaxyZooDECaLS_results/       # DECaLS color + gray results
    ├── swin_decals_results/           # Swin baseline results
```


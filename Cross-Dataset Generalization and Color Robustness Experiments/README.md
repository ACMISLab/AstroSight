# Cross-Dataset Generalization & Color Robustness Experiments

Experiments for evaluating AstroSight models on grayscale images and Galaxy Zoo DECaLS dataset to address reviewer concerns about color-morphology bias and dataset generalization.

## Overview

**Purpose:** Evaluate cross-dataset generalization and color robustness  
**Datasets:** GZ2 test set (11,088 galaxies), Galaxy Zoo DECaLS (31,191 galaxies)  
**Models Tested:** All 6 AstroSight variants + Swin Transformer baseline

## Experiments Conducted

### 1. Color Robustness on GZ2
- **Dataset:** GZ2 test set converted to grayscale
- **Goal:** Assess if models rely on color or morphological features
- **Results:** `GZ2_results/`

### 2. Cross-Dataset Generalization (DECaLS)
- **Dataset:** Galaxy Zoo DECaLS test set (color + grayscale)
- **Goal:** Evaluate generalization to deeper, higher-resolution imaging
- **Results:** `GalaxyZooDECaLS_results/`

### 3. Swin Transformer Baseline
- **Goal:** Compare color robustness with best-performing CNN/ViT baseline
- **Results:** `swin_decals_results/`

## Key Results Summary

| Model | GZ2 Color | GZ2 Gray | DECaLS Color | DECaLS Gray |
|-------|-----------|----------|--------------|-------------|
| InternVL2.5-38B | 82.94% | 70.04% | 78.44% | 68.74% |
| Ovis2-34B | 82.01% | 65.87% | 76.48% | 64.88% |
| Swin Transformer | 80.35% | 61.63% | 67.44% | 56.12% |

**Key Findings:**
- AstroSight retains 70%+ accuracy on grayscale images
- Strong cross-dataset generalization (78.44% on DECaLS)
- Outperforms Swin Transformer on both color and grayscale

## Directory Structure

```
experiments_for_revision/
├── README.md                           # This file
├── USAGE_GUIDE.md                      # Detailed analysis script usage
├── GZ2_DATASET_README.md              # GZ2 dataset information
│
├── Data Preparation Scripts
│   ├── convert_gz2_to_grayscale.py    # Convert GZ2 to grayscale
│   ├── convert_decals_v2.py           # Convert DECaLS to GZ2 format
│   └── download_resized_reduced_gz2.py # Download GZ2 dataset
│
├── Testing Scripts
│   ├── test_swin_on_decals.py         # Test Swin on DECaLS
│   ├── run_swin_tests.sh              # Batch testing script
│   └── swift.yaml                     # MLLM inference config
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
    └── analysis_results/              # Aggregated analysis
```

## Quick Start

### 1. Prepare Grayscale Test Sets

```bash
cd /mnt/acmis_hby/Paper_experiment_one/experiments_for_revision

# Convert GZ2 to grayscale
python convert_gz2_to_grayscale.py

# Convert DECaLS to grayscale
python convert_decals_v2.py --grayscale
```

### 2. Run MLLM Inference

```bash
# Activate environment
conda activate swift

# Infer on GZ2 grayscale
swift infer --model_type internvl2_5-38b-awq \
    --model_id_or_path /path/to/model \
    --eval_dataset /path/to/gz2_test_grayscale

# Infer on DECaLS color
swift infer --model_type internvl2_5-38b-awq \
    --eval_dataset /path/to/decals_test_gz2_format

# Infer on DECaLS grayscale
swift infer --model_type internvl2_5-38b-awq \
    --eval_dataset /path/to/decals_test_gz2_format_grayscale
```

### 3. Test Swin Transformer Baseline

```bash
# Test on DECaLS color
python test_swin_on_decals.py \
    --checkpoint /path/to/swin_checkpoint.pth \
    --test-dir decals_test_gz2_format

# Test on DECaLS grayscale
python test_swin_on_decals.py \
    --checkpoint /path/to/swin_checkpoint.pth \
    --test-dir decals_test_gz2_format_grayscale \
    --grayscale
```

### 4. Analyze Results

```bash
# Analyze single model
python analyze_inference_results.py \
    GalaxyZooDECaLS_results/decals_infer_results/InternVL2.5-38B.jsonl \
    --output-dir analysis_results

# Compare multiple models
python analyze_inference_results.py \
    GalaxyZooDECaLS_results/decals_infer_results/InternVL2.5-38B.jsonl \
    --output-dir analysis_results \
    --compare GalaxyZooDECaLS_results/decals_infer_results/Ovis2-34B.jsonl \
             GalaxyZooDECaLS_results/decals_infer_results/InternVL3-8B.jsonl
```

## Analysis Output

The `analyze_inference_results.py` script generates:

1. **Overall Metrics:** Accuracy, Macro-Precision, Macro-Recall, Macro-F1
2. **Per-Class Performance:** Detailed metrics for each morphology class
3. **Confusion Matrix:** 8×8 classification confusion matrix
4. **Error Analysis:** Top-10 most common misclassification patterns
5. **Prediction Distribution:** Predicted vs. ground-truth class distribution

See `USAGE_GUIDE.md` for detailed analysis script documentation.

## Paper Integration

Results from these experiments are reported in:
- **Section 4.3:** "Generalization to Galaxy Zoo DECaLS and Color Robustness Analysis"
- **Table 6:** Color robustness on GZ2 test set
- **Table 7:** Cross-dataset generalization on DECaLS

## References

- Galaxy Zoo DECaLS: Walmsley et al. 2022, MNRAS, 509, 3966
- DECaLS Survey: Dey et al. 2019, AJ, 157, 168
- Color-Morphology Correlation: Strateva et al. 2001, AJ, 122, 1861

## Notes

- All models trained on GZ2 color images only
- DECaLS provides deeper imaging (g=24.7 vs SDSS g=23.3)
- Grayscale conversion: RGB → luminosity (0.299R + 0.587G + 0.114B)
- Test sets maintain original 8-class taxonomy for consistency

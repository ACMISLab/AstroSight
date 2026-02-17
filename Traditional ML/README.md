# SVM + Zernike Moments Galaxy Classification Baseline

## Overview

Traditional machine learning baseline using Zernike Moments (ZMs) as hand-crafted features combined with Support Vector Machine (SVM) classifier for 8-class galaxy morphological classification.

**Method**: Feature Engineering + Classical Machine Learning
- **Feature Extraction**: Zernike Moments (45th order, 1,081 dimensions)
- **Classifier**: Support Vector Machine with RBF kernel
- **Purpose**: Baseline comparison with deep learning methods
- **Reference**: https://github.com/hmddev1/machine_learning_for_morphological_galaxy_classification

## Model Architecture

### Pipeline
```
Input Image (224×224×3) - Your dataset
    ↓
Preprocessing (Resize to 200×200, Convert to R channel)
    ↓
Zernike Moments Extraction (Order 45, 1081-dim features)
    ↓
Feature Standardization (StandardScaler)
    ↓
SVM Classification (RBF kernel, C=1.5)
    ↓
8-class Output
```

**Note**: Images are automatically resized from 224×224 to 200×200 for Zernike Moments computation. This is standard practice in astronomical image analysis.

### Zernike Moments
- **What**: Orthogonal moments defined on unit disk
- **Order**: 45 (standard for galaxy classification)
- **Dimensions**: (n+1)(n+2)/2 = 1,081 features
- **Properties**: 
  - Rotation invariant
  - Robust to noise
  - Captures shape information
  - Well-established in astronomy

### SVM Classifier
- **Kernel**: RBF (Radial Basis Function)
- **C**: 1.5 (regularization parameter)
- **Gamma**: 'scale' (auto-computed)
- **Class Weight**: 'balanced' (handles class imbalance)

## Key Files

- `extract_zernike_features.py` - Extract ZMs features from images
- `svm_zms_galaxy_classification.py` - Train and evaluate SVM classifier
- `README_Galaxy_Classification.md` - This documentation

## Quick Start

### Requirements
```bash
# Core dependencies
pip install ZEMO              # Zernike Moments computation
pip install scikit-learn      # SVM and evaluation
pip install opencv-python     # Image processing
pip install numpy pandas      # Data processing
pip install matplotlib seaborn # Visualization
pip install tqdm              # Progress bar
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

## Usage

### Step 1: Extract Zernike Features

**First time** - Extract features from all images:
```bash
cd /mnt/acmis_hby/Paper_experiment_one/SVM_ZMs_baseline

# Extract features (takes 2-3 hours)
python extract_zernike_features.py \
    --data_path /path/to/baselines_dataset \
    --output_path ./features \
    --image_size 200 \
    --zernike_order 45
```

**Output**:
```
features/
├── train_zernike_features.npz    # Training features
├── val_zernike_features.npz      # Validation features
└── test_zernike_features.npz     # Test features
```

### Step 2: Train SVM Classifier

**Training**:
```bash
# Train SVM with extracted features (takes ~5-10 minutes)
python svm_zms_galaxy_classification.py \
    --feature_path ./features \
    --output_dir ./result \
    --use_class_weight
```

**Testing Only**:
```bash
python svm_zms_galaxy_classification.py \
    --test_only \
    --model_path ./result/svm_zms_galaxy_8_classes_model.pkl \
    --feature_path ./features
```

### Complete Workflow (Background)
```bash
# Step 1: Extract features
nohup python extract_zernike_features.py \
    --data_path /mnt/acmis_hby/Paper_experiment_one/baselines_dataset \
    --output_path ./features \
    > extract_features.log 2>&1 &

# Wait for completion, then Step 2: Train SVM
nohup python svm_zms_galaxy_classification.py \
    --feature_path ./features \
    --output_dir ./result \
    > svm_training.log 2>&1 &

# Monitor progress
tail -f svm_training.log
```

## Configuration Parameters

### Feature Extraction
- `--data_path`: Path to dataset directory
- `--output_path`: Feature output directory (default: `./features`)
- `--image_size`: Image size for ZMs computation (default: 200)
- `--zernike_order`: Zernike order (default: 45)

### SVM Training
- `--feature_path`: Path to extracted features (default: `./features`)
- `--output_dir`: Output directory (default: `./result`)
- `--use_class_weight`: Use balanced class weights (default: True)
- `--test_only`: Only run testing
- `--model_path`: Pre-trained model path for testing

## Features

### Zernike Moments Properties
- **Rotation Invariant**: Insensitive to object orientation
- **Orthogonal**: Independent basis functions
- **Multi-scale**: Captures both local and global features
- **Shape Descriptors**: Effective for morphological analysis

### SVM Advantages
- **Non-linear Classification**: RBF kernel for complex boundaries
- **Robust**: Less prone to overfitting than deep learning (with limited data)
- **Interpretable**: Feature importance can be analyzed
- **Efficient**: Fast training and inference

### Evaluation Metrics
- Overall: Accuracy, Precision (macro/weighted), Recall (macro/weighted), F1-score (macro/weighted)
- Per-class: Precision, Recall, F1-score, Support
- Confusion Matrix (raw counts and normalized)

## Output Files

After training, the following files are generated:

### Model Files
- `svm_zms_galaxy_8_classes_model.pkl` - Trained SVM model
  - Contains: SVM model, StandardScaler, class names, training time

### Features (after extraction)
- `train_zernike_features.npz` - Training features (1,081-dim per image)
- `val_zernike_features.npz` - Validation features
- `test_zernike_features.npz` - Test features

### Results
- `svm_zms_galaxy_8_classes_results.json` - Complete evaluation results
  - Model info, training config, all metrics, confusion matrices
- `svm_zms_galaxy_8_classes_predictions.csv` - Prediction details
  - Columns: true_label, predicted_label, correct

### Visualizations
- `svm_zms_galaxy_8_classes_confusion_matrix.png` - Confusion matrices
  - Raw counts and normalized side-by-side

## Expected Performance

### Computational Cost
| Stage | Time | Resources |
|-------|------|-----------|
| Feature Extraction | 2-3 hours | CPU only |
| SVM Training | 5-10 minutes | CPU only, 8GB RAM |
| SVM Inference | <1 second | CPU only |


**Comparison with Deep Learning**:
```
Expected Results:
- SVM + ZMs:        70-80%
- ECA-CNN:          85-90%
- VGG16:            88-92%
- Swin Transformer: 90-95%

Performance Gap: 10-15%
→ Demonstrates the advantage of deep learning!
```

## Comparison with Deep Learning

| Aspect | SVM + ZMs | Deep Learning |
|--------|-----------|---------------|
| **Feature Extraction** | Hand-crafted (ZMs) | Automatic (learned) |
| **Training Time** | Minutes | Hours |
| **Inference Time** | Milliseconds | Milliseconds |
| **Data Requirements** | Moderate | Large |
| **Interpretability** | High (ZMs have physical meaning) | Low (black box) |
| **Generalization** | Depends on feature quality | Usually stronger |
| **Parameters** | ~Thousands | ~Millions |
| **GPU Required** | No | Yes (for efficiency) |

## Advantages of This Baseline

### 1. Academic Value
- Demonstrates traditional feature engineering approach
- Provides comparison point for deep learning
- Shows evolution from classical to modern methods

### 2. Computational Efficiency
- No GPU required
- Fast training and inference
- Low memory footprint

### 3. Interpretability
- Zernike Moments have clear physical interpretation
- Each feature corresponds to specific shape properties
- Feature importance can be analyzed

### 4. Robustness
- Less prone to overfitting (with proper regularization)
- Works reasonably well with limited data
- Stable training (no convergence issues)

## Zernike Moments Details

### Mathematical Definition
Zernike polynomials Z_nm(ρ, θ):
- n: order (0 to 45)
- m: repetition (-n to n)
- ρ: radial coordinate [0, 1]
- θ: angular coordinate [0, 2π]

### Feature Vector
For order 45: 1,081 features
- Z_00, Z_11, Z_1-1, Z_22, Z_20, Z_2-2, ...
- Captures increasingly fine details as n increases

### Physical Interpretation
- Low orders (n < 10): Global shape
- Medium orders (10 ≤ n < 30): Intermediate features
- High orders (n ≥ 30): Fine details and texture

## References

### Zernike Moments in Astronomy
1. Shamir, L. (2011). "Ganalyzer: A tool for automatic galaxy image analysis." *ApJ*.
2. Sreejith, S. et al. (2018). "Machine learning for galaxy morphology." *MNRAS*.

### SVM for Galaxy Classification
3. Huertas-Company, M. et al. (2015). "Galaxy classification with support vector machines." *ApJS*.
4. Ghaderi, H. et al. (2024). "Machine learning for morphological galaxy classification." *ApJS*.

### ZEMO Package
5. ZEMO: Python package for Zernike Moments
   - PyPI: https://pypi.org/project/ZEMO/
   - GitHub: https://github.com/hmddev1/ZEMO

## Troubleshooting

### ZEMO Installation Issues
```bash
# If ZEMO installation fails
pip install --upgrade pip setuptools wheel
pip install ZEMO

# Or install from source
git clone https://github.com/hmddev1/ZEMO.git
cd ZEMO
pip install -e .
```

### Memory Issues During Feature Extraction
```bash
# Process one split at a time
python extract_zernike_features.py --data_path /path/to/dataset/train
python extract_zernike_features.py --data_path /path/to/dataset/val
python extract_zernike_features.py --data_path /path/to/dataset/test
```

### SVM Training Too Slow
```bash
# Reduce training data or use linear kernel
# Edit svm_zms_galaxy_classification.py:
# kernel='linear' instead of 'rbf'
```

## Citation

If you use this SVM + Zernike Moments baseline, please cite:

```bibtex
@article{ghaderi2024machine,
  title={Machine learning for morphological galaxy classification},
  author={Ghaderi, Hamed and Alipour, Nasibe and Safari, Hossein},
  journal={The Astrophysical Journal Supplement Series},
  year={2024}
}
```

## License

This implementation uses:
- ZEMO: MIT License
- scikit-learn: BSD License
- This code: Follow your project's license

---

**Note**: This traditional machine learning baseline provides a valuable comparison point to demonstrate the advantages of deep learning methods in automatic feature learning and performance for galaxy morphological classification.

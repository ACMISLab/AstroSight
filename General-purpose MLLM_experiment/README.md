# Galaxy Morphology Classification - Baseline Experiments

This directory contains baseline comparison experiments using general-purpose vision-language models for galaxy morphology classification.

## Overview

We evaluated multiple state-of-the-art vision-language models on our galaxy classification test set with 8 morphological categories (A-H):

| Label | Category | Description |
|-------|----------|-------------|
| **A** | Round elliptical | Spherical/round elliptical galaxies |
| **B** | In-between elliptical | Intermediate elliptical shapes |
| **C** | Cigar-shaped elliptical | Elongated elliptical galaxies |
| **D** | Edge-on | Galaxies viewed edge-on |
| **E** | Barred spirals | Spiral galaxies with central bar structure |
| **F** | Unbarred spirals | Spiral galaxies without bar |
| **G** | Irregular | Galaxies with irregular morphology |
| **H** | Merger | Interacting/merging galaxies |

**Test Set**: 11,088 galaxy images（https://huggingface.co/datasets/kk1999ddk/galaxy-morphology-classification）

## File Structure

### Inference Scripts
- `gemini_flash_galaxy_classifier.py` - Gemini-2.5-flash implementation
- `qwen_vl_galaxy_classifier.py` - Qwen-VL implementation
- (models use similar structure)

### Results Files
Each model generates three types of outputs:

1. **`*_galaxy_classification_results.json`**
   - Raw prediction results 
   - Contains `detailed_results` array with per-image predictions
   - Includes API response metadata 

2. **`*_recomputed_metrics.json`**
   - Standardized evaluation metrics
   - Overall accuracy, Macro/Weighted averages
   - Per-class precision, recall, F1-score
   - Confusion matrix for error analysis

3. **`*_galaxy_classification_results_summary.xlsx`**
   - Excel format summary with two sheets:
     - Overall_Metrics: Accuracy, Precision, Recall, F1
     - Per_Class_Metrics: Detailed breakdown by category

### Evaluation Script
- **`re_evaluate_metrics.py`** - Metric recalculation script ensuring consistent evaluation across all models

### Dataset
- **`test.jsonl`** - Test dataset in JSONL format (11,088 samples)

## Metrics Calculation

The `re_evaluate_metrics.py` script provides standardized metric computation:

**Metrics Computed:**
- **Overall Accuracy**: Percentage of correctly classified galaxies
- **Macro Average**: Unweighted mean across all 8 classes (treats each class equally)
- **Weighted Average**: Mean weighted by class support (reflects dataset distribution)
- **Per-Class Metrics**: Precision, Recall, F1-score, Support for each category (A-H)
- **Confusion Matrix**: 8×8 matrix showing prediction patterns and common misclassifications

### Usage

```bash
# Basic usage
python re_evaluate_metrics.py --input <result_file.json>

# Save metrics to file
python re_evaluate_metrics.py --input <result_file.json> --save-json <output.json>
```

**Examples:**
```bash
# Re-evaluate Gemini results
python re_evaluate_metrics.py -i gemini_flash_galaxy_classification_results.json \
                              --save-json gemini_flash_recomputed_metrics.json

# Re-evaluate GPT-5 results
python re_evaluate_metrics.py -i gpt5_galaxy_classification_results.json \
                              --save-json gpt5_recomputed_metrics.json
```

## Experimental Setup

- **Prompt**: All models use identical prompt template for fair comparison
- **Temperature**: 0.1 (low temperature for consistent predictions)

## Reproducing Results

1. Ensure you have the test dataset (`test.jsonl`)
2. Set up API keys for your chosen model
3. Run the corresponding classifier script:
   ```bash
   python <model>_galaxy_classifier.py
   ```
4. Re-evaluate with standardized metrics:
   ```bash
   python re_evaluate_metrics.py --input <results>.json --save-json <metrics>.json
   ```

## Notes

- All models were tested on the same 11,088 galaxy images
- Minor sample count variations (11,076-11,088) due to API failures or data processing issues


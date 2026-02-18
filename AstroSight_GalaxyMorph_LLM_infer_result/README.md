# AstroSight — LLM Inference Results

This directory contains the raw inference outputs, evaluation scripts, and computed metrics for all AstroSight (MLLM-based) models on the Galaxy Zoo 2 (GZ2) test set.

---

## Directory Structure

```
AstroSight_GalaxyMorph_LLM_infer_result/
├── README.md                                        # This file
├── AstroSight_LLM_infer_raw_result.zip              # Archived raw inference outputs (all models)
│
├── # --- Classification inference outputs (one line per test sample) ---
├── InternVL2.5_38B_lora.jsonl                       # InternVL2.5-38B + LoRA
├── InternVL2.5_8B_lora.jsonl                        # InternVL2.5-8B + LoRA
├── InternVL2.5_8B_full.jsonl                        # InternVL2.5-8B + Full fine-tuning
├── InternVL3_8B_lora.jsonl                          # InternVL3-8B + LoRA
├── InternVL3_8B_full.jsonl                          # InternVL3-8B + Full fine-tuning
├── Ovis_34B_lora.jsonl                              # Ovis2-34B + LoRA
│
├── # --- Attribute regression inference output ---
├── InternVL2.5_38B_attribute_regression_lora.jsonl  # InternVL2.5-38B + LoRA (17-attribute regression)
│
├── # --- Evaluation scripts ---
├── compute_llm_metrics_from_jsonl.py                # Compute classification metrics from .jsonl
├── evaluate_astrosight_attributes.py                # Compute regression metrics (MAE/MSE/R²) from .jsonl
│
└── LLM_metrics/                                     # Computed metric results
    ├── InternVL2.5_38B_lora.metrics.json            # Classification metrics
    ├── InternVL2.5_8B_lora.metrics.json
    ├── InternVL2.5_8B_full.metrics.json
    ├── InternVL3_8B_lora.metrics.json
    ├── InternVL3_8B_full.metrics.json
    ├── Ovis_34B_lora.metrics.json
    └── InternVL2.5_38B_attribute_regression.json    # Attribute regression metrics (MAE/MSE/R²)
```

---

## JSONL Format

Each `.jsonl` file contains one JSON object per line. Classification files have the structure:

```json
{
  "response": "... option E is selected.",
  "labels":   "... option E is selected."
}
```

The attribute regression file additionally includes predicted and ground-truth numerical vectors for all 17 morphological attributes.

---

## Running Evaluation

### Classification metrics (Accuracy, Macro-F1, etc.)

```bash
python compute_llm_metrics_from_jsonl.py --input InternVL2.5_38B_lora.jsonl
```

Output is saved to `LLM_metrics/<model>.metrics.json`.

### Attribute regression metrics (MAE, MSE, R²)

```bash
python evaluate_astrosight_attributes.py --input InternVL2.5_38B_attribute_regression_lora.jsonl
```

Output is saved to `LLM_metrics/InternVL2.5_38B_attribute_regression.json`.

---


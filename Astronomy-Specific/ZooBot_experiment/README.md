# ZooBot Baseline Experiments

Fine-tuning ZooBot pretrained models on GZ2 dataset for comparison with AstroSight.

## Overview

**Task:** 8-class galaxy morphology classification  
**Dataset:** GZ2 (44,352 train / 11,088 test)  
**Models:** ZooBot variants pretrained on GZ Evo (820k galaxies, 100M+ votes)

## Models Tested

| Model | Parameters | Accuracy | Macro-F1 | Pretrained Checkpoint |
|-------|-----------|----------|----------|----------------------|
| ZooBot-ConvNeXt-Nano | 15.6M | 79.29% | 78.64% | `mwalmsley/zoobot-encoder-convnext_nano` |
| ZooBot-MaxViT-Small | 64.9M | 79.89% | 79.03% | `mwalmsley/zoobot-encoder-maxvit_rmlp_small_rw_224` |
| ZooBot-MaxViT-Base | 124.5M | 79.42% | 78.27% | `mwalmsley/zoobot-encoder-maxvit_base_rw_224` |

## Training Configuration

```python
training_mode = "full"        # Full model fine-tuning
learning_rate = 1e-5          # Low LR to preserve pretrained weights
layer_decay = 0.8             # Deeper layers use lower LR
batch_size = 32
max_epochs = 30
```

## Quick Start

```bash
# Activate environment
conda activate zoobot_env
cd /mnt/acmis_hby/Paper_experiment_one/ZooBot_experiments

# Train all models (sequential)
nohup python train_zoobot.py > training.log 2>&1 &

# Test trained models
python test_zoobot.py --model convnext_nano
python test_zoobot.py --model maxvit_rmlp_small
python test_zoobot.py --model maxvit_base

# Monitor training
tail -f training.log
```

## File Structure

```
ZooBot_experiments/
├── train_zoobot.py              # Training script
├── test_zoobot.py               # Evaluation script
├── extract_results.py           # Results aggregation
├── data/
│   ├── train_catalog.csv
│   ├── val_catalog.csv
│   └── test_catalog.csv
├── checkpoints/                 # Trained model checkpoints
│   ├── convnext_nano/
│   ├── maxvit_rmlp_small/
│   └── maxvit_base/
└── results/finetuned/           # Evaluation results
    ├── convnext_nano_finetuned.json
    ├── maxvit_rmlp_small_finetuned.json
    ├── maxvit_base_finetuned.json
    ├── overall_metrics_summary.csv
    └── perclass_metrics_summary.csv
```

## Results Summary

All models fine-tuned on unified 8-class GZ2 dataset. Results differ from original publications due to dataset and taxonomy differences.

**Comparison with AstroSight:**
- AstroSight (InternVL2.5-38B): 82.94% accuracy, 82.33% macro-F1
- Best ZooBot (MaxViT-Small): 79.89% accuracy, 79.03% macro-F1
- Performance gap: ~3% accuracy

## References

- ZooBot Documentation: https://zoobot.readthedocs.io/
- Walmsley et al. 2022: https://doi.org/10.1093/mnras/stab2093
- Pretrained Models: https://zoobot.readthedocs.io/en/latest/data_notes/pretrained_models.html
# AstroSight

This is the repository of **AstroSight: Galaxy Morphology Classification with Multimodal Large
Language Models**.

## 🛠️ Installation

### For Large Language Model Fine-tuning

To install using pip:
```bash
pip install ms-swift -U
```

To install from source:
```bash
git clone https://github.com/modelscope/ms-swift.git
cd ms-swift
pip install -e .
```

### Running Environment:
| Package | Range | Recommended | Notes |
|---------|-------|-------------|-------|
| python | >=3.9 | 3.10/3.11 | |
| cuda | | cuda12 | No need to install if using CPU |
| torch | >=2.0 | 2.7.1 | |
| transformers | >=4.33 | 4.56.1 | |
| modelscope | >=1.23 | | |
| peft | >=0.11,<0.18 | | |
| flash_attn | | 2.7.4.post1/3.0.0b1 | |
| deepspeed | >=0.14 | 0.17.5 | Training |
| vllm | >=0.5.1 | 0.10.1.1 | Inference/Deployment |

### For Baseline Experiments

For reproducing baseline model experiments, install PyTorch and navigate to the corresponding directories:

- **CNNs/**: ResNet, DenseNet, EfficientNet implementations
- **Vision_Transformers/**: Swin Transformer, CVT, Linformer models  
- **Astronomy-Specific/**: AstroCLIP, Deformable CNNs, DIAT-DSCNN-ECA
- **Traditional ML/**: Classical machine learning baselines

## 📥 Dataset

The galaxy classification datasets are provided in JSONL format:
- `dataset/Galaxy_morphology/` - Galaxy morphological classification data
- `dataset/Galaxy_attributes/` - Galaxy attribute regression data


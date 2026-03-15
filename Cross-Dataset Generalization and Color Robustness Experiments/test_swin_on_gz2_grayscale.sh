#!/bin/bash
# 在GZ2灰度测试集上测试Swin Transformer

CHECKPOINT="/mnt/acmis_hby/Paper_experiment_one/Transformer_experiemnt/Swin_transformer/swin_transformer_of_GalaxyMorphology_best.pth"
SCRIPT="/mnt/acmis_hby/Paper_experiment_one/experiments_for_revision/test_swin_on_decals.py"
OUTPUT_DIR="/mnt/acmis_hby/Paper_experiment_one/experiments_for_revision/swin_results"

# 创建输出目录
mkdir -p $OUTPUT_DIR

echo "========================================================================"
echo "Testing Swin Transformer on GZ2 Grayscale Test Set"
echo "========================================================================"
echo "Checkpoint: $CHECKPOINT"
echo "Test data: gz2_test_grayscale/test.jsonl"
echo "Output: $OUTPUT_DIR/swin_gz2_grayscale_results.json"
echo ""

# 运行测试
CUDA_VISIBLE_DEVICES=0 python $SCRIPT \
    --checkpoint $CHECKPOINT \
    --jsonl /mnt/acmis_hby/Paper_experiment_one/experiments_for_revision/gz2_test_grayscale/test.jsonl \
    --output $OUTPUT_DIR/swin_gz2_grayscale_results.json \
    --batch-size 32 \
    --gpu 0

echo ""
echo "========================================================================"
echo "Testing completed!"
echo "Results saved to: $OUTPUT_DIR/swin_gz2_grayscale_results.json"
echo "========================================================================"

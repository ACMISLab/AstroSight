#!/bin/bash
# ECA-CNN 模型评估脚本 - 使用GPU 1

echo "=========================================="
echo "🚀 启动 ECA-CNN 模型评估"
echo "=========================================="
echo "使用GPU: 1"
echo "模型: eca_cnn_galaxy_8_classes_best.h5"
echo ""

# 设置使用GPU 1
export CUDA_VISIBLE_DEVICES=1

# 运行评估脚本
python3 eca_cnn_evaluate_only.py 2>&1 | tee eca_cnn_evaluation.log

echo ""
echo "=========================================="
echo "✅ 评估完成，日志已保存到 eca_cnn_evaluation.log"
echo "=========================================="

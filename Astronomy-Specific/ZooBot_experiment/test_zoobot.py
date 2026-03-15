#!/usr/bin/env python3
"""
测试微调后的ZooBot模型
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from galaxy_datasets.transforms import default_view_config, get_galaxy_transform
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
from zoobot.pytorch.predictions import predict_on_catalog

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_model(model_name, checkpoint_path):
    """测试单个模型"""
    
    logger.info("="*80)
    logger.info(f"Testing {model_name}")
    logger.info("="*80)
    
    # 路径配置
    test_csv = "/mnt/acmis_hby/Paper_experiment_one/ZooBot_experiments/data/test_catalog.csv"
    output_dir = Path(f"/mnt/acmis_hby/Paper_experiment_one/ZooBot_experiments/results/finetuned")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取测试数据
    logger.info(f"Loading test data from: {test_csv}")
    test_catalog = pd.read_csv(test_csv)
    
    if 'id_str' not in test_catalog.columns:
        test_catalog['id_str'] = test_catalog.index.astype(str)
    
    logger.info(f"Test samples: {len(test_catalog)}")
    
    # 加载模型
    logger.info(f"Loading model from: {checkpoint_path}")
    model = FinetuneableZoobotClassifier.load_from_checkpoint(checkpoint_path)
    logger.info("✓ Model loaded")
    
    # 创建transform
    transform_cfg = default_view_config()
    transform = get_galaxy_transform(transform_cfg)
    
    # 预测
    logger.info("\nMaking predictions...")
    
    # 为8个类别创建列名
    label_cols = [f'class_{i}_pred' for i in range(8)]
    
    # 创建临时保存路径
    temp_save_loc = output_dir / f'{model_name}_predictions.csv'
    
    predictions = predict_on_catalog.predict(
        test_catalog,
        model,
        label_cols=label_cols,
        save_loc=str(temp_save_loc),
        inference_transform=transform,
        trainer_kwargs={'accelerator': 'gpu', 'devices': 1},  # 强制使用单GPU避免分布式问题
        datamodule_kwargs={'num_workers': 4, 'batch_size': 32}
    )
    
    # 合并真实标签
    predictions = pd.merge(
        predictions,
        test_catalog[['id_str', 'label']],
        on='id_str',
        how='inner'
    )
    
    # 计算预测类别（argmax）
    pred_probs = predictions[[f'class_{i}_pred' for i in range(8)]].values
    pred_labels = np.argmax(pred_probs, axis=1)
    true_labels = predictions['label'].values
    
    # 计算指标
    logger.info("\nComputing metrics...")
    
    accuracy = accuracy_score(true_labels, pred_labels) * 100
    
    class_names = [
        'barred_spirals',
        'cigar_shaped_elliptical',
        'edge_on',
        'in_between_elliptical',
        'irregular',
        'merger',
        'round_elliptical',
        'unbarred_spirals'
    ]
    
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 提取macro和weighted指标
    macro_avg = report['macro avg']
    weighted_avg = report['weighted avg']
    
    results = {
        'model': model_name,
        'accuracy': accuracy,
        'macro_precision': macro_avg['precision'] * 100,
        'macro_recall': macro_avg['recall'] * 100,
        'macro_f1': macro_avg['f1-score'] * 100,
        'weighted_precision': weighted_avg['precision'] * 100,
        'weighted_recall': weighted_avg['recall'] * 100,
        'weighted_f1': weighted_avg['f1-score'] * 100,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    
    # 打印结果
    logger.info("\n" + "="*80)
    logger.info("Test Results")
    logger.info("="*80)
    logger.info(f"Accuracy:           {results['accuracy']:.2f}%")
    logger.info(f"Macro Precision:    {results['macro_precision']:.2f}%")
    logger.info(f"Macro Recall:       {results['macro_recall']:.2f}%")
    logger.info(f"Macro F1-Score:     {results['macro_f1']:.2f}%")
    logger.info(f"Weighted Precision: {results['weighted_precision']:.2f}%")
    logger.info(f"Weighted Recall:    {results['weighted_recall']:.2f}%")
    logger.info(f"Weighted F1-Score:  {results['weighted_f1']:.2f}%")
    logger.info("="*80)
    
    # 保存结果
    output_path = output_dir / f'{model_name}_finetuned.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_path}")
    
    return results


def resolve_checkpoint(checkpoint_dir: Path, model_name: str) -> Path | None:
    model_dir = checkpoint_dir / model_name / "checkpoints"
    checkpoints = list(model_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else x.stat().st_mtime)
    return checkpoints[-1]


def main():
    """测试模型（支持单模型命令行调用）"""
    parser = argparse.ArgumentParser(description="Test finetuned ZooBot models")
    parser.add_argument("--model", default="convnext_nano", help="Model name to test")
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (.ckpt)")
    args = parser.parse_args()

    checkpoint_dir = Path("/mnt/acmis_hby/Paper_experiment_one/ZooBot_experiments/checkpoints")

    models = [args.model]
    
    logger.info("="*80)
    logger.info("Testing All Finetuned ZooBot Models")
    logger.info("="*80)
    
    all_results = {}
    
    for model_name in models:
        # 查找最佳checkpoint（在checkpoints子目录中）
        if args.checkpoint:
            checkpoint_path = Path(args.checkpoint)
        else:
            checkpoint_path = resolve_checkpoint(checkpoint_dir, model_name)

        if checkpoint_path is None or not checkpoint_path.exists():
            logger.warning(f"No checkpoint found for {model_name}")
            continue
        
        logger.info(f"\n[{model_name}] Checkpoint: {checkpoint_path}")
        
        try:
            results = test_model(model_name, checkpoint_path)
            all_results[model_name] = results
        except Exception as e:
            logger.error(f"Testing {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    logger.info("\n" + "="*80)
    logger.info("Testing Summary")
    logger.info("="*80)
    for model_name, results in all_results.items():
        logger.info(f"{model_name}: {results['accuracy']:.2f}% accuracy")
    logger.info("="*80)


if __name__ == "__main__":
    main()

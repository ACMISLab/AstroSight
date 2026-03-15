#!/usr/bin/env python3
"""
ZooBot训练脚本 - 参照官方notebook
在GZ2数据集上微调ZooBot模型
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ZooBot imports
from galaxy_datasets.pytorch.galaxy_datamodule import CatalogDataModule
from galaxy_datasets.transforms import default_view_config, get_galaxy_transform
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier, get_trainer


def train_model(model_name, model_hf_name, gpu_id=0, epochs=30, batch_size=64, lr=1e-4):
    """
    训练单个ZooBot模型
    
    Args:
        model_name: 模型名称 
        model_hf_name: HuggingFace模型名称
        gpu_id: GPU设备ID
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
    """
    logger.info("="*80)
    logger.info(f"Training {model_name}")
    logger.info("="*80)
    
    # 路径配置
    data_dir = Path("/mnt/acmis_hby/Paper_experiment_one/ZooBot_experiments/data")
    save_dir = Path(f"/mnt/acmis_hby/Paper_experiment_one/ZooBot_experiments/checkpoints/{model_name}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = data_dir / "train_catalog.csv"
    val_csv = data_dir / "val_catalog.csv"
    
    logger.info(f"Train CSV: {train_csv}")
    logger.info(f"Val CSV: {val_csv}")
    logger.info(f"Save dir: {save_dir}")
    # Resolve local GPU index when CUDA_VISIBLE_DEVICES is set
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    local_gpu_id = gpu_id
    if visible_devices is not None:
        visible_list = [d.strip() for d in visible_devices.split(",") if d.strip() != ""]
        if visible_list:
            if str(gpu_id) in visible_list:
                local_gpu_id = visible_list.index(str(gpu_id))
            else:
                local_gpu_id = 0
                logger.warning(
                    "GPU %s not in CUDA_VISIBLE_DEVICES=%s, falling back to local GPU 0",
                    gpu_id,
                    visible_devices,
                )
    logger.info(f"GPU: {gpu_id} (local index: {local_gpu_id})")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Learning rate: {lr}")
    
    # 读取数据
    logger.info("\nLoading data...")
    train_catalog = pd.read_csv(train_csv)
    
    # 确保有id_str列
    if 'id_str' not in train_catalog.columns:
        train_catalog['id_str'] = train_catalog.index.astype(str)
    
    logger.info(f"Train samples: {len(train_catalog)}")
    
    # 创建transforms
    logger.info("\nCreating transforms...")
    transform_cfg = default_view_config()
    transform = get_galaxy_transform(transform_cfg)
    
    # 创建DataModule
    logger.info("Creating DataModule...")
    datamodule = CatalogDataModule(
        label_cols=['label'],
        catalog=train_catalog,
        train_transform=transform,
        test_transform=transform,
        batch_size=batch_size,
        num_workers=4
    )
    
    # 创建模型
    logger.info(f"\nLoading pretrained model: {model_hf_name}")
    model = FinetuneableZoobotClassifier(
        name=model_hf_name,
        training_mode="full",  # finetune the full model (not just head)
        num_classes=8,
        learning_rate=lr,
        layer_decay=0.8,  # reduce learning rate for deeper layers
        label_col='label'
    )
    logger.info("✓ Model loaded")
    
    # 创建trainer
    logger.info("\nCreating trainer...")
    trainer = get_trainer(
        save_dir=str(save_dir),
        accelerator='gpu',
        devices=[local_gpu_id],
        max_epochs=epochs
    )
    logger.info("✓ Trainer created")
    
    # 开始训练
    logger.info("\n" + "="*80)
    logger.info("Starting training...")
    logger.info("="*80)
    
    try:
        trainer.fit(model, datamodule)
        
        # 保存训练信息
        training_info = {
            'model': model_name,
            'best_checkpoint': str(trainer.checkpoint_callback.best_model_path),
            'best_val_loss': float(trainer.checkpoint_callback.best_model_score),
            'total_epochs': trainer.current_epoch + 1,
            'hyperparameters': {
                'batch_size': batch_size,
                'learning_rate': lr,
                'epochs': epochs
            }
        }
        
        import json
        info_path = save_dir / 'training_info.json'
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("\n" + "="*80)
        logger.info("Training completed!")
        logger.info("="*80)
        logger.info(f"Best checkpoint: {trainer.checkpoint_callback.best_model_path}")
        logger.info(f"Best val loss: {trainer.checkpoint_callback.best_model_score:.4f}")
        logger.info(f"Training info saved to: {info_path}")
        logger.info("="*80)
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数 - 训练所有模型"""
    
    # 模型配置 - 使用官方推荐的三个模型，分配到两张GPU
    models = [
        {
            'name': 'convnext_nano',
            'hf_name': 'hf_hub:mwalmsley/zoobot-encoder-convnext_nano',
            'params': '15.6M',
            'gpu': 0  # GPU 0
        },
        {
            'name': 'maxvit_rmlp_small',
            'hf_name': 'hf_hub:mwalmsley/zoobot-encoder-maxvit_rmlp_small_rw_224',
            'params': '64.9M',
            'gpu': 1  # GPU 1
        },
        {
            'name': 'maxvit_base',
            'hf_name': 'hf_hub:mwalmsley/zoobot-encoder-maxvit_base_rw_224',
            'params': '124.5M',
            'gpu': 0  # GPU 0 
        }
    ]
    
    # 训练参数（参照官方文档推荐）
    training_params = {
        'epochs': 30,
        'batch_size': 32,  
        'lr': 1e-5  
    }
    
    logger.info("="*80)
    logger.info("ZooBot Training Pipeline")
    logger.info("="*80)
    logger.info(f"Models to train: {len(models)}")
    logger.info(f"Training params: {training_params}")
    logger.info("="*80)
    
    # 训练每个模型
    results = {}
    for i, model_config in enumerate(models, 1):
        logger.info(f"\n[{i}/{len(models)}] Training {model_config['name']}...")
        
        success = train_model(
            model_name=model_config['name'],
            model_hf_name=model_config['hf_name'],
            gpu_id=model_config['gpu'],
            **training_params
        )
        
        results[model_config['name']] = 'Success' if success else 'Failed'
        
        logger.info(f"\n{model_config['name']}: {'✓ Success' if success else '✗ Failed'}")
    
    # 总结
    logger.info("\n" + "="*80)
    logger.info("Training Summary")
    logger.info("="*80)
    for model_name, status in results.items():
        logger.info(f"{model_name}: {status}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

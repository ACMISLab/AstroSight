#!/usr/bin/env python
# evaluate_all_models.py

import os
import re
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms as tfs

# ==== 1. 脚本所在目录（CVT/） ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==== 2. 将 CVT/ 加入模块搜索路径，便于 import lib/* ====
sys.path.insert(0, SCRIPT_DIR)

# ==== 3. 导入项目模块 ====
from lib.config import config, update_config
from lib.dataset.galaxy_zoo import GalaxyZoo
from lib.models import build_model

# ==== 4. 解析 checkpoint 文件名中的 epoch 数 ====
def parse_epoch(fname):
    m = re.match(r'.*weight_e(\d+)\.pth$', fname)
    return int(m.group(1)) if m else None

# ==== 5. 自定义验证函数：返回 (accuracy, avg_loss) ====
def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total, loss_sum / total

# ==== 6. 主流程 ====
def main():
    # ——— 6.1 加载配置 —— 
    cfg_file = os.path.join(
        SCRIPT_DIR,
        "experiments/imagenet/cvt/CVT-13-224x224.yaml"
    )
    if not os.path.isfile(cfg_file):
        raise FileNotFoundError(f"配置文件不存在：{cfg_file}")
    dummy = type("X", (), {"cfg": cfg_file, "opts": []})()
    update_config(config, dummy)
    print(f"=> merge config from {cfg_file}")

    # ——— 6.2 构建验证集 DataLoader —— 
    val_transform = tfs.Compose([
        tfs.Resize([224, 224]),
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.5, 0.5, 0.5],
                      std=[0.5, 0.5, 0.5])
    ])
    val_dataset = GalaxyZoo(
        root=config.DATASET.ROOT,
        mode="val",
        transform=val_transform
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False,
        num_workers=0,               # 避免死锁
        pin_memory=True,
        collate_fn=GalaxyZoo.collate_fn
    )

    # ——— 6.3 构建模型 & 损失 —— 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # ——— 6.4 遍历 runs/ 下所有 checkpoint —— 
    ckpt_dir = os.path.join(SCRIPT_DIR, "runs")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint 目录不存在：{ckpt_dir}")

    results = []
    for fname in sorted(os.listdir(ckpt_dir)):
        epoch = parse_epoch(fname)
        if epoch is None:
            continue
        path = os.path.join(ckpt_dir, fname)
        print(f"Evaluating epoch {epoch:02d} ... ", end="", flush=True)

        # 加载权重
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        # 验证
        acc, loss = evaluate(model, val_loader, criterion, device)
        print(f"acc={acc:.4f}, loss={loss:.4f}")
        results.append((epoch, acc, loss))

    # ——— 6.5 输出最佳模型 —— 
    if not results:
        print("⚠️ 未找到任何 weight_e*.pth 文件。")
        return

    best_epoch, best_acc, best_loss = max(results, key=lambda x: x[1])
    print("\n✅ 最佳模型:")
    print(f"   Epoch    : {best_epoch}")
    print(f"   Accuracy : {best_acc:.4f}")
    print(f"   Loss     : {best_loss:.4f}")
    print("   Checkpoint 路径:",
          os.path.join(ckpt_dir, f"weight_e{best_epoch}.pth"))

if __name__ == "__main__":
    main()

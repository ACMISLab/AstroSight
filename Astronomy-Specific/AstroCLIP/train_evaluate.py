#!/usr/bin/env python3
"""
AstroCLIP Baseline Comparison Experiments (Rewritten with 17 attributes)
Supports:
- Galaxy morphology classification 
- Galaxy attribute prediction 

Usage:
python astroclip_baseline_experiments_fixed.py --task regression
python astroclip_baseline_experiments_fixed.py --task classification
python astroclip_baseline_experiments_fixed.py --task both
"""

import argparse
import json
import os
import sys
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


print("🔧 Setting up virtual module mappings...")
import types

datasets_module = types.ModuleType('datasets')

class _VirtualGeneratorBasedBuilder:
    def __init__(self, *args, **kwargs):
        pass

class _VirtualDataset:
    def __init__(self, *args, **kwargs):
        pass

class _VirtualVersion:
    def __init__(self, *args, **kwargs):
        pass

dclass_placeholder = True

class _VirtualBuilderConfig:
    def __init__(self, name=None, version=None, description=None, **kwargs):
        self.name = name
        self.version = version
        self.description = description
        for k, v in kwargs.items():
            setattr(self, k, v)

class _VirtualSplit:
    TRAIN = 'train'
    VALIDATION = 'validation'
    TEST = 'test'

class _VirtualDownloadManager:
    def __init__(self, *args, **kwargs):
        pass

datasets_module.Dataset = _VirtualDataset
datasets_module.GeneratorBasedBuilder = _VirtualGeneratorBasedBuilder
datasets_module.Version = _VirtualVersion
datasets_module.BuilderConfig = _VirtualBuilderConfig
datasets_module.Split = _VirtualSplit
datasets_module.DownloadManager = _VirtualDownloadManager
datasets_module.load_dataset = lambda *args, **kwargs: {}
datasets_module.DatasetInfo = dict
datasets_module.Features = dict
datasets_module.Value = str
datasets_module.Image = str

import importlib.machinery
datasets_module.__spec__ = importlib.machinery.ModuleSpec('datasets', None)
datasets_module.__file__ = '<virtual>'
datasets_module.__package__ = None
sys.modules['datasets'] = datasets_module

import pytorch_lightning as pl
sys.modules['lightning'] = pl
sys.modules['lightning.pytorch'] = pl
print("✅ Virtual modules set up")


sys.path.insert(0, '/mnt/acmis_hby/galaxy_contranst/AstroCLIP')
from astroclip.models import AstroClipModel
print("✅ AstroCLIP successfully imported")


class GalaxyClassificationDataset(Dataset):
    def __init__(self, jsonl_path, transform=None):
        self.transform = transform
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        print(f"Loaded {len(self.data)} classification samples from {jsonl_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['images'][0]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ 无法加载图像 {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')

        assistant_content = item['messages'][1]['content']
        label_char = None
        for char in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
            if f'option {char}' in assistant_content:
                label_char = char
                break
        if label_char is None:
            label = 0
        else:
            CLASS_MAPPING = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
            label = CLASS_MAPPING[label_char]

        if self.transform:
            image = self.transform(image)
        return image, label


class GalaxyRegressionDataset(Dataset):
    def __init__(self, jsonl_path, transform=None):
        self.transform = transform
        self.data = []

        self.attribute_names = [
            'f_smooth', 'f_features/disk', 'f_edge-on/yes', 'f_edge-on/no',
            'f_bar/yes', 'f_bar/no', 'f_spiral/yes', 'f_odd/yes', 'f_odd/no',
            'f_completelyround', 'f_in-between', 'f_cigar-shaped',
            'f_disturbed', 'f_irregular', 'f_other', 'f_merger', 'f_dustlane'
        ]

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line.strip()))
        print(f"Loaded {len(self.data)} regression samples from {jsonl_path}")
        print("✅ Using 17 attributes (fixed from 16)")

    def __len__(self):
        return len(self.data)

    def parse_attributes(self, text: str):
        attributes = np.zeros(17)
        for i, attr_name in enumerate(self.attribute_names):
            pattern = rf'{re.escape(attr_name)}=([0-9]+\.?[0-9]*)'
            match = re.search(pattern, text)
            if match:
                attributes[i] = float(match.group(1))
        return attributes

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item['images'][0]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"❌ 无法加载图像 {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        attributes = self.parse_attributes(item['messages'][1]['content'])
        if self.transform:
            image = self.transform(image)
        return image, torch.FloatTensor(attributes)


class AstroCLIPClassifier(nn.Module):
    def __init__(self, num_classes=8, freeze_backbone=True):
        super().__init__()
        print("Loading AstroCLIP model for classification...")
        checkpoint_path = \
            "/mnt/acmis_hby/galaxy_contranst/AstroCLIP/astroclip.ckpt"
        if not os.path.exists(checkpoint_path):
            print("Downloading AstroCLIP checkpoint...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://huggingface.co/polymathic-ai/astroclip/resolve/main/astroclip.ckpt",
                checkpoint_path
            )
            print("✅ Checkpoint downloaded")

        astroclip_full = AstroClipModel.load_from_checkpoint(checkpoint_path)
        self.backbone = astroclip_full.image_encoder.backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("✅ AstroCLIP DINOv2 backbone frozen")

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 252, 252)
            if next(self.backbone.parameters()).is_cuda:
                dummy = dummy.cuda()
            x = self.backbone.patch_embed(dummy)
            for blk in self.backbone.blocks:
                x = blk(x)
            features = self.backbone.norm(x)
            feature_dim = features.shape[-1]
        print(f"✅ AstroCLIP loaded, feature dim: {feature_dim}")

        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feature_dim, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            x = self.backbone.patch_embed(x)
            for blk in self.backbone.blocks:
                x = blk(x)
            features = self.backbone.norm(x)
            features = features[:, 0]
        return self.classifier(features)


class AstroCLIPRegressor(nn.Module):
    def __init__(self, num_attributes=17, freeze_backbone=True):
        super().__init__()
        print("Loading AstroCLIP model for regression...")
        checkpoint_path = \
            "/mnt/acmis_hby/galaxy_contranst/AstroCLIP/astroclip.ckpt"
        if not os.path.exists(checkpoint_path):
            print("Downloading AstroCLIP checkpoint...")
            import urllib.request
            urllib.request.urlretrieve(
                "https://huggingface.co/polymathic-ai/astroclip/resolve/main/astroclip.ckpt",
                checkpoint_path
            )
            print("✅ Checkpoint downloaded")

        import types as _types
        fabric_module = _types.ModuleType('lightning.fabric')
        fabric_utilities = _types.ModuleType('lightning.fabric.utilities')
        fabric_utilities_data = _types.ModuleType('lightning.fabric.utilities.data')
        if hasattr(pl.utilities, 'AttributeDict'):
            AttributeDict = pl.utilities.AttributeDict
        else:
            class AttributeDict(dict):
                def __getattr__(self, key):
                    try:
                        return self[key]
                    except KeyError as e:
                        raise AttributeError(str(e))
                def __setattr__(self, key, value):
                    self[key] = value
        fabric_utilities_data.AttributeDict = AttributeDict
        fabric_utilities.data = fabric_utilities_data
        fabric_module.utilities = fabric_utilities
        sys.modules['lightning.fabric'] = fabric_module
        sys.modules['lightning.fabric.utilities'] = fabric_utilities
        sys.modules['lightning.fabric.utilities.data'] = fabric_utilities_data

        astroclip_full = AstroClipModel.load_from_checkpoint(checkpoint_path)
        self.backbone = astroclip_full.image_encoder.backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("✅ AstroCLIP DINOv2 regression backbone frozen")

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, 252, 252)
            if next(self.backbone.parameters()).is_cuda:
                dummy = dummy.cuda()
            x = self.backbone.patch_embed(dummy)
            for blk in self.backbone.blocks:
                x = blk(x)
            features = self.backbone.norm(x)
            feature_dim = features.shape[-1]
        print(f"✅ AstroCLIP loaded for regression, feature dim: {feature_dim}")

        self.regressor = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(feature_dim, 512), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2), nn.Linear(256, num_attributes)
        )

    def forward(self, x):
        with torch.set_grad_enabled(self.backbone.training):
            x = self.backbone.patch_embed(x)
            for blk in self.backbone.blocks:
                x = blk(x)
            features = self.backbone.norm(x)
            features = features[:, 0]
        return self.regressor(features)


def train_classification_model(model, train_loader, val_loader, device, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_acc, best_model_state, best_epoch = 0.0, None, 0
    for epoch in range(epochs):
        model.train(); train_loss=0.0; train_correct=0; train_total=0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model(images)
            loss = criterion(outputs, labels); loss.backward(); optimizer.step()
            train_loss += loss.item()
            _, pred = outputs.max(1); train_total += labels.size(0)
            train_correct += pred.eq(labels).sum().item()
            acc = 100.*train_correct/train_total
            progress_bar.set_postfix({'Loss': f'{train_loss/(len(progress_bar.iterable)):.4f}', 'Acc': f'{acc:.2f}%'})
        model.eval(); val_correct=0; val_total=0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = outputs.max(1)
                val_total += labels.size(0)
                val_correct += pred.eq(labels).sum().item()
        val_acc = 100.*val_correct/val_total; scheduler.step()
        print(f'Epoch {epoch+1}: Train Acc: {100.*train_correct/train_total:.2f}% | Val Acc: {val_acc:.2f}%')
        if val_acc > best_acc:
            best_acc = val_acc; best_model_state = model.state_dict().copy(); best_epoch = epoch+1
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_epoch


def train_regression_model(model, train_loader, val_loader, device, epochs=100):
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_loss, best_model_state, best_epoch = float('inf'), None, 0
    for epoch in range(epochs):
        model.train(); train_loss=0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, targets in progress_bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(); outputs = model(images)
            loss = criterion(outputs, targets); loss.backward(); optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{train_loss/(len(progress_bar.iterable)):.6f}'})
        model.eval(); val_loss=0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets); val_loss += loss.item()
        val_loss /= len(val_loader); scheduler.step()
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.6f} | Val Loss: {val_loss:.6f}')
        if val_loss < best_loss:
            best_loss = val_loss; best_model_state = model.state_dict().copy(); best_epoch = epoch+1
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model, best_epoch


def evaluate_classification(model, test_loader, device):
    model.eval(); all_predictions=[]; all_labels=[]
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images); _, pred = outputs.max(1)
            all_predictions.extend(pred.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted', zero_division=0)
    return accuracy, precision, recall, f1


def evaluate_regression(model, test_loader, device, attribute_names):
    model.eval(); all_predictions=[]; all_targets=[]
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="Testing"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            all_predictions.append(outputs.cpu().numpy()); all_targets.append(targets.cpu().numpy())
    predictions = np.vstack(all_predictions); targets = np.vstack(all_targets)
    overall_mae = mean_absolute_error(targets, predictions)
    overall_mse = mean_squared_error(targets, predictions)
    overall_r2 = r2_score(targets, predictions)
    attribute_results = {}
    for i, attr_name in enumerate(attribute_names):
        mae = mean_absolute_error(targets[:, i], predictions[:, i])
        mse = mean_squared_error(targets[:, i], predictions[:, i])
        r2 = r2_score(targets[:, i], predictions[:, i])
        attribute_results[attr_name] = {'MAE': mae, 'MSE': mse, 'R2': r2}
    return overall_mae, overall_mse, overall_r2, attribute_results


def main():
    parser = argparse.ArgumentParser(description='AstroCLIP Galaxy Baselines (17 attributes)')
    parser.add_argument('--task', choices=['classification', 'regression', 'both'], default='regression')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((252, 252)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((252, 252)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    results = {}

    if args.task in ['classification', 'both']:
        print("\n" + "="*80)
        print("GALAXY MORPHOLOGICAL CLASSIFICATION EXPERIMENT")
        print("="*80)

        cls_train = '/mnt/acmis_hby/galaxy_contranst/galaxy_classification/train.jsonl'
        cls_test = '/mnt/acmis_hby/galaxy_contranst/galaxy_classification/test.jsonl'
        if os.path.exists(cls_train) and os.path.exists(cls_test):
            train_dataset = GalaxyClassificationDataset(cls_train, transform=train_transform)
            test_dataset = GalaxyClassificationDataset(cls_test, transform=test_transform)
            train_size = int(0.8 * len(train_dataset)); val_size = len(train_dataset) - train_size
            train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
            val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            model = AstroCLIPClassifier(num_classes=8).to(device)
            model, best_epoch = train_classification_model(model, train_loader, val_loader, device, args.epochs)
            accuracy, precision, recall, f1 = evaluate_classification(model, test_loader, device)
            results['classification'] = {
                'accuracy': accuracy * 100,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'epochs': best_epoch
            }
            print(f"\n🎉 Classification Results:\nAccuracy: {accuracy*100:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Best Epoch: {best_epoch}")
        else:
            print("⚠️")

    # 回归（17属性，使用 galaxy_attributes）
    if args.task in ['regression', 'both']:
        print("\n" + "="*80)
        print("GALAXY ATTRIBUTE PREDICTION EXPERIMENT (17 attributes)")
        print("="*80)

        reg_train = '/mnt/acmis_hby/galaxy_contranst/galaxy_attributes/train.jsonl'
        reg_test = '/mnt/acmis_hby/galaxy_contranst/galaxy_attributes/test.jsonl'
        train_dataset = GalaxyRegressionDataset(reg_train, transform=train_transform)
        test_dataset = GalaxyRegressionDataset(reg_test, transform=test_transform)
        train_size = int(0.8 * len(train_dataset)); val_size = len(train_dataset) - train_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        model = AstroCLIPRegressor(num_attributes=17).to(device)
        model, best_epoch = train_regression_model(model, train_loader, val_loader, device, args.epochs)
        overall_mae, overall_mse, overall_r2, attribute_results = evaluate_regression(
            model, test_loader, device, train_dataset.attribute_names
        )
        results['regression'] = {
            'overall_mae': overall_mae,
            'overall_mse': overall_mse,
            'overall_r2': overall_r2,
            'attribute_results': attribute_results,
            'epochs': best_epoch,
            'num_attributes': 17
        }
        print(f"\n🎉 Regression Results (17): MAE={overall_mae:.4f} | MSE={overall_mse:.4f} | R²={overall_r2:.4f} | Best Epoch={best_epoch}")

    out_path = '/mnt/acmis_hby/galaxy_contranst/AstroCLIP_results_fixed.json'
    with open(out_path, 'w') as f:
        json.dump(convert_numpy_types(results), f, indent=2)
    print("\n" + "="*80)
    print("PAPER FORMAT RESULTS (FIXED VERSION)")
    print("="*80)
    if 'classification' in results:
        r = results['classification']
        print(f"Classification\t{r['accuracy']:.2f}\t{r['precision']:.2f}\t{r['recall']:.2f}\t{r['f1_score']:.2f}\t{r['epochs']}")
    if 'regression' in results:
        r = results['regression']
        print(f"Regression\t{r['overall_mae']:.4f}\t{r['overall_mse']:.4f}\t{r['overall_r2']:.4f}")
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()






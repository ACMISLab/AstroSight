#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import models.han_dcn as han_dcn

class DetailedEvaluator:
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        
    def evaluate_dataset(self, data_loader, dataset_name="Test"):
        self.model.eval()
        
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for images, targets in data_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        all_probs = np.array(all_probs)
        
        accuracy = accuracy_score(all_targets, all_preds) * 100
        
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None, zero_division=0
        )
        
        macro_precision = np.mean(precision) * 100
        macro_recall = np.mean(recall) * 100
        macro_f1 = np.mean(f1) * 100
        
        cm = confusion_matrix(all_targets, all_preds)
        
        results = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_precision': precision * 100,
            'per_class_recall': recall * 100,
            'per_class_f1': f1 * 100,
            'per_class_support': support,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
        
        return results
    
    def print_results(self, results):
        print(f"\n=== {results['dataset']} Dataset Evaluation Results ===")
        print(f"Overall Accuracy: {results['accuracy']:.2f}%")
        print(f"Macro Average Precision: {results['macro_precision']:.2f}%")
        print(f"Macro Average Recall: {results['macro_recall']:.2f}%")
        print(f"Macro Average F1-Score: {results['macro_f1']:.2f}%")
        
        print(f"\nPer-class Results:")
        print(f"{'Class':<20} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
        print("-" * 60)
        
        for i in range(len(self.class_names)):
            class_name = self.class_names[i]
            precision = results['per_class_precision'][i]
            recall = results['per_class_recall'][i]
            f1 = results['per_class_f1'][i]
            support = results['per_class_support'][i]
            
            print(f"{class_name:<20} {precision:<10.2f} {recall:<10.2f} {f1:<10.2f} {support:<8}")
    
    def save_confusion_matrix(self, results, output_path):
        plt.figure(figsize=(10, 8))
        
        class_labels = [self.class_names[i].split(':')[1] for i in range(len(self.class_names))]
        
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d',
            xticklabels=class_labels,
            yticklabels=class_labels,
            cmap='Blues'
        )
        
        plt.title(f'{results["dataset"]} Dataset - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path / f'{results["dataset"].lower()}_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results_json(self, results, output_path):
        json_results = {
            'dataset': results['dataset'],
            'accuracy': float(results['accuracy']),
            'macro_precision': float(results['macro_precision']),
            'macro_recall': float(results['macro_recall']),
            'macro_f1': float(results['macro_f1']),
            'per_class_results': {}
        }
        
        for i in range(len(self.class_names)):
            class_name = self.class_names[i]
            json_results['per_class_results'][class_name] = {
                'precision': float(results['per_class_precision'][i]),
                'recall': float(results['per_class_recall'][i]),
                'f1_score': float(results['per_class_f1'][i]),
                'support': int(results['per_class_support'][i])
            }
        
        with open(output_path / f'{results["dataset"].lower()}_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)

def load_model(checkpoint_path, num_classes=8, device='cuda'):
    model = han_dcn.resnet_HAN_DCN(num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def setup_data_loaders(data_path, batch_size=64, num_workers=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.18334192, 0.17221707, 0.16791163], [0.15241465, 0.13768229, 0.12769352])
    ])
    
    test_dataset = datasets.ImageFolder(
        data_path / "test", 
        transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader, test_dataset

def generate_comparison_table(results):
    class_names = {
        0: 'A:round_elliptical',
        1: 'B:in_between_elliptical', 
        2: 'C:cigar_shaped_elliptical',
        3: 'D:edge_on',
        4: 'E:barred_spirals',
        5: 'F:unbarred_spirals',
        6: 'G:irregular',
        7: 'H:merger'
    }
    
    print(f"\n=== Deformable CNNs Performance Comparison Table ===")
    print(f"Method Category: Deformable CNNs")
    print(f"Model: Resnet_HAN_DCN") 
    print(f"Accuracy: {results['accuracy']:.2f}")
    print(f"Precision: {results['macro_precision']:.2f}")
    print(f"Recall: {results['macro_recall']:.2f}")
    print(f"F1-Score: {results['macro_f1']:.2f}")
    print(f"Training Epochs: 50")
    
    return {
        'method_category': 'Deformable CNNs',
        'model': 'Resnet_HAN_DCN',
        'accuracy': results['accuracy'],
        'precision': results['macro_precision'],
        'recall': results['macro_recall'],
        'f1_score': results['macro_f1'],
        'training_epochs': 50
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='model checkpoint path')
    parser.add_argument('--data-path', required=True, help='dataset path')
    parser.add_argument('--output-dir', required=True, help='output directory')
    parser.add_argument('--batch-size', default=64, type=int, help='batch size')
    parser.add_argument('--device', default='cuda', help='device')
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    class_names = {
        0: 'A:round_elliptical',
        1: 'B:in_between_elliptical', 
        2: 'C:cigar_shaped_elliptical',
        3: 'D:edge_on',
        4: 'E:barred_spirals',
        5: 'F:unbarred_spirals',
        6: 'G:irregular',
        7: 'H:merger'
    }
    
    print("Loading model...")
    model, checkpoint_info = load_model(args.checkpoint, device=args.device)
    print(f"Model loaded successfully, trained for {checkpoint_info.get('epoch', 'Unknown')} epochs")
    
    print("Loading test data...")
    test_loader, test_dataset = setup_data_loaders(
        Path(args.data_path), 
        batch_size=args.batch_size
    )
    print(f"Test set: {len(test_dataset)} samples")
    
    evaluator = DetailedEvaluator(model, args.device, class_names)
    
    print("Starting evaluation...")
    test_results = evaluator.evaluate_dataset(test_loader, "Test")
    
    evaluator.print_results(test_results)
    
    evaluator.save_confusion_matrix(test_results, output_path)
    evaluator.save_results_json(test_results, output_path)
    
    comparison_results = generate_comparison_table(test_results)
    
    with open(output_path / 'comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nEvaluation complete! Results saved to: {output_path}")
    
    return test_results

if __name__ == '__main__':
    main()
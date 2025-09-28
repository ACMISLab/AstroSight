#!/usr/bin/env python3

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
import random

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
import torchvision.datasets as datasets
from engine import train_one_epoch, evaluate
from samplers import RASampler

import utils
import models.han_dcn as han_dcn
import torchvision.transforms as transforms

def get_args_parser():
    parser = argparse.ArgumentParser('Galaxy Morphology Classification Training', add_help=False)
    
    parser.add_argument('--data-path', default='/mnt/acmis_hby/galaxy_contranst/Deformable_CNNS/data/galaxy_morph_imagefolder', 
                        type=str, help='dataset path')
    parser.add_argument('--data-set', default='galaxy', choices=['galaxy'], 
                        type=str, help='dataset type')
    
    parser.add_argument('--model', default='resnet_HAN_DCN', type=str, metavar='MODEL',
                        help='model name')
    parser.add_argument('--num-classes', default=8, type=int, help='number of classes')
    parser.add_argument('--input-size', default=224, type=int, help='input image size')
    
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=50, type=int, help='training epochs')
    
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER',
                        help='optimizer type')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='optimizer epsilon')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='optimizer betas')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='gradient clipping norm')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='learning rate scheduler')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='minimum learning rate')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for plateau LR scheduler')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate')
    
    parser.add_argument('--repeated-aug', action='store_true', default=True)
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    parser.add_argument('--output_dir', default='/mnt/acmis_hby/galaxy_contranst/Deformable_CNNS/output/galaxy_morph_8class',
                        help='output directory')
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='perform evaluation only')
    parser.add_argument('--dist-eval', action='store_true', default=False, 
                        help='enable distributed evaluation')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--pin-mem', action='store_true', default=True,
                        help='pin CPU memory in DataLoader')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--dist_url', default='env://', help='distributed training url')
    
    return parser

def setup_transforms():
    transform_train = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.18334192, 0.17221707, 0.16791163], [0.15241465, 0.13768229, 0.12769352])
    ])
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.18334192, 0.17221707, 0.16791163], [0.15241465, 0.13768229, 0.12769352])
    ])
    
    return transform_train, transform_val

def main(args):
    utils.init_distributed_mode(args)
    
    print(f"Training galaxy morphology classification model")
    print(f"Parameters: {args}")
    
    device = torch.device(args.device)
    
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    transform_train, transform_val = setup_transforms()
    
    print("Loading dataset...")
    dataset_train = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), 
        transform=transform_train
    )
    dataset_val = datasets.ImageFolder(
        os.path.join(args.data_path, "val"), 
        transform=transform_val
    )
    dataset_test = datasets.ImageFolder(
        os.path.join(args.data_path, "test"), 
        transform=transform_val
    )
    
    print(f"Training set: {len(dataset_train)} samples")
    print(f"Validation set: {len(dataset_val)} samples")
    print(f"Test set: {len(dataset_test)} samples")
    print(f"Class mapping: {dataset_train.class_to_idx}")
    
    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    
    if args.repeated_aug:
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    
    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    print(f"Creating model: {args.model} (num_classes={args.num_classes})")
    model = han_dcn.__dict__[args.model](num_classes=args.num_classes)
    model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {n_parameters:,}')
    
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Validation accuracy: {test_stats['acc1']:.2f}%")
        return
    
    print(f"Starting training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    test_acc = []
    train_acc = []
    
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
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
        )
        
        lr_scheduler.step(epoch)
        
        train_test_stats = evaluate(data_loader_train, model, device)
        print(f"Epoch {epoch+1} training accuracy: {train_test_stats['acc1']:.2f}%")
        
        test_stats = evaluate(data_loader_val, model, device)
        test_acc.append(test_stats['acc1'])
        train_acc.append(train_test_stats['acc1'])
        print(f"Epoch {epoch+1} validation accuracy: {test_stats['acc1']:.2f}%")
        
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Best validation accuracy: {max_accuracy:.2f}%')
        
        if args.output_dir and ((epoch > args.epochs//2) and (epoch % 5 == 0)):
            checkpoint_paths = [output_dir / f'checkpoint_epoch_{epoch}.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'max_accuracy': max_accuracy,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                    'class_names': class_names,
                }, checkpoint_path)
        
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
            'max_accuracy': max_accuracy
        }
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "training_log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    if args.output_dir:
        final_checkpoint = output_dir / 'final_model.pth'
        utils.save_on_master({
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': args.epochs,
            'max_accuracy': max_accuracy,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'scaler': loss_scaler.state_dict(),
            'args': args,
            'class_names': class_names,
        }, final_checkpoint)
    
    print("\nFinal evaluation on test set...")
    final_test_stats = evaluate(data_loader_test, model, device)
    print(f"Final test accuracy: {final_test_stats['acc1']:.2f}%")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')
    
    return final_test_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Galaxy Morphology Classification Training', 
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
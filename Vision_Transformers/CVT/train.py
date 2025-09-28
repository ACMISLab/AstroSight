from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import time

import torch
import torch.nn.parallel
import torch.optim as optim
from torch import nn
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import DataLoader
from torchvision import transforms as tfs
from sklearn import metrics



import _init_paths
from lib.config import config
from lib.config import update_config
from lib.config import save_config
from lib.core.loss import build_criterion
from lib.core.function import train_one_epoch, test
from lib.dataset import build_dataloader
from lib.dataset.galaxy_zoo import GalaxyZoo
from lib.models import build_model
from lib.optim import build_optimizer
from lib.scheduler import build_lr_scheduler
from lib.utils.comm import comm
from lib.utils.utils import create_logger
from lib.utils.utils import init_distributed
from lib.utils.utils import setup_cudnn
from lib.utils.utils import summary_model_on_master
from lib.utils.utils import resume_checkpoint
from lib.utils.utils import save_checkpoint_on_master
from lib.utils.utils import save_model_on_master


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='/remote-home/cs_acmis_hby/Galaxy-Zoo-Classification/Contrast_experiment/Galaxy-Morphology/CVT/experiments/imagenet/cvt/CVT-13-224x224.yaml',
                        type=str)

    parser.add_argument('--num_gpus', default=1, type=int, help='Number of GPUs to use')

    
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--pretrain', default="/remote-home/cs_acmis_hby/Galaxy-Zoo-Classification/Contrast_experiment/Galaxy-Morphology/CVT/CvT-13-224x224-IN-1k.pth", type=str)
    parser.add_argument('--work_dir', default="../runs", type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args() 
    

    return args

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    torch.distributed.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )
    torch.distributed.barrier()



def main():
    args = parse_args()

    # init_distributed(args)
    init_distributed_mode(args)
    setup_cudnn(config)

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir

    if comm.is_main_process():
        logging.info("=> collecting env info (might take some time)")
        logging.info("\n" + get_pretty_env_info())
        logging.info(pprint.pformat(args))
        logging.info(config)
        logging.info("=> using {} GPUs".format(args.num_gpus))

        output_config_path = os.path.join(final_output_dir, 'config.yaml')
        logging.info("=> saving config into: {}".format(output_config_path))
        save_config(config, output_config_path)

    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device(args.device)

    model = build_model(config)

    state_dict = torch.load(args.pretrain, map_location="cpu")
    for key in ["head.weight", "head.bias"]:
        del state_dict[key]
    print(model.load_state_dict(state_dict, strict=False))

    model.to(device)

    transform = {
        "train":
            tfs.Compose([
                tfs.Resize([224, 224]),
                tfs.RandomHorizontalFlip(),
                tfs.ToTensor(),
                tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ]),
        "val":
            tfs.Compose([
                tfs.Resize([224, 224]),
                tfs.ToTensor(),
                tfs.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    }

    dataset_train = GalaxyZoo(root="/remote-home/cs_acmis_hby/Galaxy-Zoo-Classification/Contrast_experiment/Galaxy-Classification-Using-CNN/output_dataset", mode='train', transform=transform["train"])
    dataset_val = GalaxyZoo(root="/remote-home/cs_acmis_hby/Galaxy-Zoo-Classification/Contrast_experiment/Galaxy-Classification-Using-CNN/output_dataset", mode='val', transform=transform["val"])

    dataloader_train = DataLoader(dataset=dataset_train,
                                  batch_size=args.batch_size,
                                  drop_last=True,
                                  shuffle=True,
                                  pin_memory=True,
                                  num_workers=8,
                                  collate_fn=GalaxyZoo.collate_fn)

    dataloader_val = DataLoader(dataset=dataset_val,
                                batch_size=args.batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=8,
                                collate_fn=GalaxyZoo.collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()


    print_freq = 50
    for epoch in range(args.epochs):
        total, correct, cnt = 0, 0, 0
        for batch_id, (images, labels) in enumerate(dataloader_train, start=1):
            # images: tensor: [batch_size, 3, 224, 224]
            # labels: tensor: [batch_size]

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)  # [batch_size, num_classes(5)]

            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = output.max(1)  # pred: [batch_size]

            total += len(labels)
            correct += sum(pred.detach().cpu() == labels.detach().cpu())

            accuracy = correct / total
            
    
            cnt += 1
            if cnt % print_freq == 0:
                print(f'epoch[{epoch}/{args.epochs}  {batch_id}/{len(dataloader_train.dataset)//args.batch_size}]  '
                      f'loss: {loss.item():.3f}  accuracy: {accuracy:.3f} ' )

           

        with torch.no_grad():
            total, correct, cnt = 0, 0, 0
            for batch_id, (images, labels) in enumerate(dataloader_val, start=1):
                images = images.to(device)
                labels = labels.to(device)

                output = model(images)
                loss = criterion(output, labels)

                _, pred = output.max(1)  # pred: [batch_size]

                total += len(labels)
                correct += sum(pred.detach().cpu() == labels.detach().cpu())

                accuracy = correct / total

                cnt += 1
                if cnt % print_freq == 0:
                    print(f'valid {batch_id}/{len(dataloader_val.dataset)//args.batch_size}  '
                          f'loss: {loss.item():.3f}  accuracy: {accuracy:.3f} ')

            torch.save(model.state_dict(), os.path.join(args.work_dir, f".weight_e{epoch}.pth"))
            print("model saved")


if __name__ == '__main__':
    main()

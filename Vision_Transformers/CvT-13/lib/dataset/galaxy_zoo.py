import os

import torch
from torch.utils.data import Dataset
from PIL import Image

def make_dataset(path):
    samples = []

    cls_names = os.listdir(path)
    cls_names = [cls_name for cls_name in cls_names if "." not in cls_name]

    for cls_idx, cls_name in enumerate(cls_names):
        dir_path = os.path.join(path, cls_name)
        for image_file in os.listdir(dir_path):
            image_path = os.path.join(dir_path, image_file)
            samples.append((image_path, cls_idx))

    return samples, cls_names


class GalaxyZoo(Dataset):
    def __init__(self, root, mode='train', transform=None):
        self.samples, self.classes = make_dataset(os.path.join(root, mode))  # [(image1, label1), (image2, label2), ...]
        self.transform = transform

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        # batch: [(image1, label1), (image2, label2), ..., (image_b, label_b)]
        # images: [image1, image2, ..., image_b]  batch_size张(3, 224, 224)的图片
        # labels: [label1, label2, ..., label_b]

        # images, labels = list(*zip(batch))

        images = []
        labels = []
        for image, label in batch:
            images.append(image)
            labels.append(label)

        images = torch.stack(images, dim=0)  # [batch_size, 3, 224, 224]
        labels = torch.tensor(labels, dtype=torch.long)

        return images, labels

import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import CIFAR100

from config import NORMALIZATION, BATCH_SIZE


def get_cifar100_dataloader():
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomCrop((32, 32), padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*NORMALIZATION),
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(*NORMALIZATION),
    ])

    train_set = CIFAR100(root='./data', train=True,
                         download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=-1)

    test_set = CIFAR100(root='./data', train=False,
                        download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=-1)
    return train_loader, test_loader


class FilterDataset(Dataset):
    def __init__(self, path, kernel_size):
        imgs = []
        for root, __, files in os.walk(path):
            for f in files:
                if f.endswith(".png"):
                    imgs.append(Image.open(os.path.join(root, f)).convert('L'))
        self.imgs = imgs
        self.kernel_size = kernel_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image = self.imgs[index]
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((self.kernel_size, self.kernel_size)),
            torchvision.transforms.ToTensor()])
        X = transform(image)
        return X

import argparse

import torch
import torch.nn as nn
from pytorch_lightning import Trainer

from config import AVAIL_GPUS
from src.datasets import get_cifar100_dataloader, FilterDataset
from src.models import Classifier, GAN, Discriminator


def train(args):
    # Initialize a trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=5,
    )

    model = GAN(Classifier(), Discriminator(args.kernel_size**2))

    train_loader, _ = get_cifar100_dataloader()

    filter_dataset = FilterDataset(path=args.kernelset_path, kernel_size=args.kernel_size)
    filter_loader = torch.utils.data.DataLoader(filter_dataset, batch_size=64, shuffle=True, num_workers=2)

    trainer.fit(model, [train_loader, filter_loader])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_size", help="Kernel size for discriminator")
    parser.add_argument("kernelset_path", help="Dataset of kernels")
    args = parser.parse_args()
    train(args)

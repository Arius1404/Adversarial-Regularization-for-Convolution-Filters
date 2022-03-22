import torch.nn as nn
from pytorch_lightning import Trainer

from config import AVAIL_GPUS
from src.datasets import get_cifar100_dataloader
from src.models import CustomModel

if __name__ == '__main__':
    criterion = nn.CrossEntropyLoss()
    # Initialize a trainer
    trainer = Trainer(
        gpus=AVAIL_GPUS,
        max_epochs=55,
    )

    model = CustomModel(criterion)

    train_loader, _ = get_cifar100_dataloader()

    trainer.fit(model, train_loader)
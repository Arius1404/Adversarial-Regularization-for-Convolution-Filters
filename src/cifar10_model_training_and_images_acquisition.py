import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR100

from pytorch_lightning import LightningModule, Trainer
from utils import extract_filters

# For Google Colab
#from google.colab import drive
#drive.mount('/content/drive')

# Preparing CIFAR-10 dataset

from torch.utils.data import Subset
train_set = CIFAR10(root='./data', train=True,
                    download=True, transform=train_transform)

subset = []
for k in range(10):
  n = 0
  i = 0
  while n != 5:
    if train_set[i][1] == k:
      subset.append(i)
      n += 1
    i += 1

new_train_set = Subset(train_set, subset)
train_loader = torch.utils.data.DataLoader(new_train_set,
	batch_size=batch_size, shuffle=True, num_workers=2)

test_set = CIFAR10(root='./data',
	train=False, download=True, transform=test_transform)

test_loader = torch.utils.data.DataLoader(test_set,
	batch_size=batch_size, shuffle=False, num_workers=2)
	
## Training the model on a subset from CIFAR-10

# Check that the output size of the network is BATCH_SIZE x NUM_CLASSES
X = next(iter(train_loader))[0]
with torch.no_grad():
    clf_X = model(X)
    assert len(clf_X) == len(X)
    assert clf_X.shape[1] == 10
	
criterion = nn.CrossEntropyLoss()

trainer = Trainer(
    gpus=AVAIL_GPUS,
    max_epochs=55,
)

model = CustomModel(criterion, n_classes = 10)
trainer.fit(model, train_loader)

model.cuda()
print("Accuracy of predictions on test set is:", epoch_test(test_loader, model, criterion)[1])

# First convolution layer kernels save to an image
extract_filters(model, 0, '/content/drive/MyDrive/ML_project/by_channel_CIFAR10_layer0')

# Second convolution layer kernels save to an image
extract_filters(model, 3, '/content/drive/MyDrive/ML_project/by_channel_CIFAR10_layer3')

# Third convolution layer kernels save to an image
extract_filters(model, 6, '/content/drive/MyDrive/ML_project/by_channel_CIFAR10_layer6')

# Obtained pictures are then used to compare to kernels of other models
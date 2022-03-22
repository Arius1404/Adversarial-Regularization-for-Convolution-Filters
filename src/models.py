from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule, Trainer


class CustomModel(LightningModule):
    def __init__(self, criterion, n_classes=100):
        super().__init__()
        self.criterion = criterion

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AvgPool2d((32, 32)),
        )

        self.head = nn.Sequential(
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_preds = self(X)
        loss = self.criterion(y_preds, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)


class GAN(LightningModule):
    def __init__(
            self,
            generator,
            discriminator,
            # latent_dim: int = 100,
            lr: float = 0.0002,
            b1: float = 0.5,
            b2: float = 0.999,
            batch_size: int = 20,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # networks
        # data_shape = (channels, width, height)
        self.generator = generator
        self.discriminator = discriminator

        # self.validation_z = torch.randn(8, self.hparams.latent_dim)

        # self.example_input_array = torch.zeros(2, self.hparams.latent_dim)

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def training_step(self, batches, batch_idx, optimizer_idx):
        cifar_imgs, cifar_labels = batches[0]
        kernel_true = batches[1]
        filters_amount = 192  # depends on layer number

        # train generator
        if optimizer_idx == 0:
            # generate images
            g_pred = self(cifar_imgs)

            # GET KERNELS FROM GENERATOR
            idx = np.random.choice(np.arange(filters_amount), size=64)
            kernel_fake = self.generator.features[0].weight.reshape(-1, 1, 7, 7)[idx]

            label_fake = torch.zeros(kernel_fake.size(0), 1)  # zeros -> fake labels
            label_fake = label_fake.type_as(kernel_fake)

            d_loss = F.binary_cross_entropy_with_logits(self.discriminator(kernel_fake), label_fake)
            ce_loss = F.cross_entropy(g_pred, cifar_labels)

            g_loss = d_loss + ce_loss
            tqdm_dict = {"g_loss": g_loss}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

        # train discriminator
        if optimizer_idx == 1:
            # real
            valid = torch.ones(kernel_true.size(0), 1)  # ones -> true labels
            valid = valid.type_as(kernel_true)

            real_loss = F.binary_cross_entropy_with_logits(self.discriminator(kernel_true), valid)

            # GET KERNELS FROM GENERATOR
            idx = np.random.choice(np.arange(filters_amount), size=64)
            kernel_fake = self.generator.features[0].weight.reshape(-1, 1, 7, 7)[idx]

            # fake
            fake = torch.zeros(kernel_fake.size(0), 1)
            fake = fake.type_as(kernel_fake)

            fake_loss = F.binary_cross_entropy_with_logits(self.discriminator(kernel_fake), fake)

            # discriminator loss is the average of these
            d_loss = real_loss + fake_loss
            tqdm_dict = {"d_loss": d_loss}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output

    def configure_optimizers(self):
        lr = self.hparams.lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def on_epoch_end(self):
        z = self.validation_z.type_as(self.generator.model[0].weight)

        # log sampled images
        sample_imgs = self(z)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image("generated_images", grid, self.current_epoch)


class Classifier(nn.Module):
    def __init__(self, n_classes=100):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, bias=False),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AvgPool2d((32, 32)),
        )

        self.head = nn.Sequential(
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
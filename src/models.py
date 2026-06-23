"""Generator and Discriminator networks for the Conditional GAN."""
import numpy as np
import torch
from torch import nn


class Generator(nn.Module):
    """Maps a latent noise vector and a class label to a 28x28 image."""

    def __init__(self, classes=10, channels=1, img_size=28, latent_dim=100):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_embedding = nn.Embedding(classes, classes)
        self.model = nn.Sequential(
            *self._layer(latent_dim + classes, 128, normalize=False),
            *self._layer(128, 256),
            *self._layer(256, 512),
            *self._layer(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh(),
        )

    @staticmethod
    def _layer(size_in, size_out, normalize=True):
        layers = [nn.Linear(size_in, size_out)]
        if normalize:
            layers.append(nn.BatchNorm1d(size_out))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, noise, labels):
        z = torch.cat((self.label_embedding(labels), noise), -1)
        x = self.model(z)
        return x.view(x.size(0), *self.img_shape)


class Discriminator(nn.Module):
    """Classifies an image conditioned on its class label as real or fake."""

    def __init__(self, classes=10, channels=1, img_size=28):
        super().__init__()
        self.img_shape = (channels, img_size, img_size)
        self.label_embedding = nn.Embedding(classes, classes)
        self.adv_loss = nn.BCELoss()
        self.model = nn.Sequential(
            *self._layer(classes + int(np.prod(self.img_shape)), 1024, drop_out=False),
            *self._layer(1024, 512),
            *self._layer(512, 256),
            *self._layer(256, 128, drop_out=False, act_func=False),
            *self._layer(128, 1, drop_out=False, act_func=False),
            nn.Sigmoid(),
        )

    @staticmethod
    def _layer(size_in, size_out, drop_out=True, act_func=True):
        layers = [nn.Linear(size_in, size_out)]
        if drop_out:
            layers.append(nn.Dropout(0.4))
        if act_func:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        return layers

    def forward(self, image, labels):
        x = torch.cat((image.view(image.size(0), -1), self.label_embedding(labels)), -1)
        return self.model(x)

    def loss(self, output, label):
        return self.adv_loss(output, label)

"""Central configuration for the Conditional GAN."""
from dataclasses import dataclass

import torch


@dataclass
class Config:
    # Data
    data_root: str = "./data"
    download: bool = True

    # Model
    classes: int = 10
    channels: int = 1
    img_size: int = 28
    latent_dim: int = 100

    # Training
    batch_size: int = 128
    epochs: int = 500
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    log_interval: int = 1

    # Paths
    checkpoint_path: str = "./checkpoints/CGAN.pth"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# FashionMNIST class labels
CLASS_NAMES = [
    "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]

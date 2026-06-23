"""Training loop for the Conditional GAN."""
import os

import torch

from .config import Config
from .dataset import get_dataloader
from .models import Discriminator, Generator


def train(cfg: Config = Config()):
    device = cfg.device
    loader = get_dataloader(cfg.data_root, cfg.batch_size, train=True, download=cfg.download)

    gen = Generator(cfg.classes, cfg.channels, cfg.img_size, cfg.latent_dim).to(device)
    disc = Discriminator(cfg.classes, cfg.channels, cfg.img_size).to(device)

    optG = torch.optim.Adam(gen.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    optD = torch.optim.Adam(disc.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    gen.train()
    disc.train()

    for epoch in range(cfg.epochs):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            bsz = data.size(0)
            real_label = torch.full((bsz, 1), 1.0, device=device)
            fake_label = torch.full((bsz, 1), 0.0, device=device)

            # Train generator
            optG.zero_grad()
            noise = torch.randn(bsz, cfg.latent_dim, device=device)
            fake_labels = torch.randint(0, cfg.classes, (bsz,), device=device)
            x_fake = gen(noise, fake_labels)
            g_loss = disc.loss(disc(x_fake, fake_labels), real_label)
            g_loss.backward()
            optG.step()

            # Train discriminator
            optD.zero_grad()
            d_real_loss = disc.loss(disc(data, target), real_label)
            d_fake_loss = disc.loss(disc(x_fake.detach(), fake_labels), fake_label)
            d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss.backward()
            optD.step()

        if (epoch + 1) % cfg.log_interval == 0:
            print(
                f"Epoch [{epoch + 1}/{cfg.epochs}] "
                f"loss_D: {d_loss.item():.4f} loss_G: {g_loss.item():.4f}"
            )

    save_checkpoint(cfg, gen, disc, optG, optD)
    return gen, disc


def save_checkpoint(cfg, gen, disc, optG, optD):
    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
    checkpoint = {
        "epoch": cfg.epochs,
        "generator": gen.state_dict(),
        "discriminator": disc.state_dict(),
        "optimizer_generator": optG.state_dict(),
        "optimizer_discriminator": optD.state_dict(),
        "lr": cfg.lr,
    }
    torch.save(checkpoint, cfg.checkpoint_path)
    print(f"Saved checkpoint to {cfg.checkpoint_path}")


if __name__ == "__main__":
    train()

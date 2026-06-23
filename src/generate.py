"""Load a trained checkpoint and generate sample images."""
import argparse

import torch

from .config import Config
from .models import Generator
from .utils import show_generated


def load_generator(checkpoint_path, cfg: Config = Config(), device="cpu"):
    gen = Generator(cfg.classes, cfg.channels, cfg.img_size, cfg.latent_dim).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gen.load_state_dict(checkpoint["generator"])
    gen.eval()
    return gen


def main():
    parser = argparse.ArgumentParser(description="Generate images with a trained CGAN")
    parser.add_argument("--checkpoint", default=Config().checkpoint_path)
    args = parser.parse_args()

    cfg = Config()
    device = cfg.device
    gen = load_generator(args.checkpoint, cfg, device)
    show_generated(gen, latent_dim=cfg.latent_dim, device=device)


if __name__ == "__main__":
    main()

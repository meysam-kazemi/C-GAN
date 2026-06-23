# Conditional GAN (CGAN) — FashionMNIST

A Conditional Generative Adversarial Network implemented in PyTorch that
generates 28×28 FashionMNIST images conditioned on a class label.

By providing a class label alongside the latent noise vector, the generator
learns to produce images of a *specific* category (e.g. "Sneaker", "Bag",
"Trouser") rather than random samples from the whole dataset.

## Classes

`T-shirt`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`,
`Sneaker`, `Bag`, `Ankle boot`

## Project structure

```
C-GAN/
├── src/
│   ├── config.py      # hyperparameters and class names
│   ├── models.py      # Generator and Discriminator
│   ├── dataset.py     # FashionMNIST DataLoader
│   ├── train.py       # training loop + checkpointing
│   ├── generate.py    # load a checkpoint and sample images
│   └── utils.py       # visualization helpers
├── notebooks/
│   ├── CGAN.ipynb     # exploration & training notebook
│   └── test_model.ipynb  # load checkpoint and visualize results
├── requirements.txt
├── LICENSE
└── README.md
```

## Architecture

Both networks are fully-connected MLPs. The class label is mapped through an
`nn.Embedding` layer and concatenated with the input:

- **Generator** — `latent(100) + label(10)` → `128 → 256 → 512 → 1024 → 784`,
  `Tanh` output, reshaped to `1×28×28`.
- **Discriminator** — `image(784) + label(10)` → `1024 → 512 → 256 → 128 → 1`,
  `Sigmoid` output, with dropout in the early layers.

## Installation

```bash
git clone https://github.com/meysam-kazemi/C-GAN.git
cd C-GAN
pip install -r requirements.txt
```

## Training

```bash
python -m src.train
```

The FashionMNIST dataset is downloaded automatically to `./data`, and a
checkpoint is written to `./checkpoints/CGAN.pth`. Hyperparameters live in
`src/config.py`.

## Generating images

```bash
python -m src.generate --checkpoint ./checkpoints/CGAN.pth
```

## License

Released under the [MIT License](LICENSE).

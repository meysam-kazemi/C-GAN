"""Visualization helpers."""
import matplotlib.pyplot as plt
import torch

from .config import CLASS_NAMES


def show_dataset(dataset, rows=8, cols=8):
    """Plot a grid of random samples from a dataset."""
    fig, axs = plt.subplots(figsize=(8, 9), nrows=rows, ncols=cols)
    for ax_row in axs:
        for ax in ax_row:
            idx = torch.randint(low=0, high=len(dataset), size=(1,)).item()
            img, lbl = dataset[idx]
            ax.imshow(img.squeeze(), "gray")
            ax.set_title(CLASS_NAMES[lbl], fontsize=6)
            ax.axis(False)
    plt.show()


def show_generated(generator, labels=None, latent_dim=100, device="cpu"):
    """Generate one image per class and display the grid."""
    generator.eval()
    if labels is None:
        labels = torch.arange(len(CLASS_NAMES))
    labels = labels.to(device)
    noise = torch.randn(len(labels), latent_dim, device=device)

    with torch.no_grad():
        images = generator(noise, labels)

    cols = 5
    rows = (len(labels) + cols - 1) // cols
    _, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 2 * rows + 1))
    for i, ax in enumerate(axs.ravel()):
        if i < len(labels):
            ax.imshow(images[i].cpu().reshape(28, 28), "gray")
            ax.set_title(CLASS_NAMES[labels[i]])
        ax.axis(False)
    plt.show()

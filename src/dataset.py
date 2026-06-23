"""FashionMNIST data loading."""
import torchvision as tv
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def get_dataloader(root="./data", batch_size=128, train=True, download=True):
    """Return a DataLoader over the FashionMNIST dataset."""
    dataset = tv.datasets.FashionMNIST(
        root=root,
        train=train,
        download=download,
        transform=ToTensor(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

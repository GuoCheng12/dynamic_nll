from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class DummyRegressionDataset(Dataset):
    """
    Minimal dataset placeholder; replace with UCI/Depth loaders.
    """

    def __init__(self, size: int = 1000):
        self.x = torch.randn(size, 10)
        self.y = self.x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(size, 1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def build_dataloaders(batch_size: int, splits=(0.8, 0.1, 0.1)):
    dataset = DummyRegressionDataset()
    n = len(dataset)
    train_len = int(splits[0] * n)
    val_len = int(splits[1] * n)
    test_len = n - train_len - val_len
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    return train_loader, val_loader, test_loader

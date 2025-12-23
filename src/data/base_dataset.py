from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split


class ToyRegressionDataset(Dataset):
    """
    Input-domain long-tail regression toy dataset.
    x ~ Gamma(k=2.0, theta=1.5); y = sin(x) * (0.5x + 1) + eps with heteroscedastic noise.
    eps ~ N(0, (0.1 + 0.05x)^2)
    Targets are normalized; y_mean/y_std stored for denormalization.
    """

    def __init__(self, size: int = 5000, normalize: bool = True):
        super().__init__()
        self.size = size
        self.normalize = normalize
        self.x, y = self._generate(size)
        self.y_mean = y.mean()
        self.y_std = y.std().clamp(min=1e-6)
        if normalize:
            y = (y - self.y_mean) / self.y_std
        self.y = y

    @staticmethod
    def _generate(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma_dist = torch.distributions.Gamma(concentration=2.0, rate=1 / 1.5)
        x = gamma_dist.sample((size,)).unsqueeze(1)
        sigma = 0.1 + 0.05 * x
        noise = torch.randn_like(x) * sigma
        y = torch.sin(x) * (0.5 * x + 1.0) + noise
        return x, y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def unnormalize(self, y_pred: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return y_pred
        return y_pred * self.y_std + self.y_mean


def build_dataloaders(batch_size: int, splits=(0.8, 0.1, 0.1), seed: int | None = None):
    dataset = ToyRegressionDataset()
    n = len(dataset)
    train_len = int(splits[0] * n)
    val_len = int(splits[1] * n)
    test_len = n - train_len - val_len
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, generator=generator)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, generator=generator)
    return train_loader, val_loader, test_loader

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


class PaperSineDataset(Dataset):
    """
    Heteroscedastic Sine dataset from Beta-NLL papers.
    x ~ Uniform(0, 10)
    y = x*sin(x) + x*xi1 + xi2, with xi1, xi2 ~ N(0, 0.3^2)
    True variance: 0.09 * (x^2 + 1)
    """

    def __init__(self, n_samples: int = 500, mode: str = "train", seed: int = 0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed if seed is not None else torch.seed())
        if mode == "test":
            x = torch.linspace(0.0, 10.0, n_samples)
        else:
            x = torch.rand(n_samples, generator=generator) * 10.0
        x = x.unsqueeze(1)
        xi1 = torch.randn(x.shape, generator=generator) * 0.3
        xi2 = torch.randn(x.shape, generator=generator) * 0.3
        self.x = x
        self.true_mean = x * torch.sin(x)
        self.true_var = 0.09 * (x ** 2 + 1.0)
        self.y = self.true_mean + x * xi1 + xi2

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def get_true_variance(self):
        return self.true_var


def build_paper_sine_dataloaders(
    data_cfg,
    batch_size: int,
    seed: int | None = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    eval_distributed: bool | None = None,
    num_workers: int = 0,
    eval_batch_size: int | None = None,
):
    if eval_distributed is None:
        eval_distributed = distributed
    if eval_batch_size is None:
        eval_batch_size = batch_size

    n_samples = data_cfg.get("n_samples", 500)
    val_samples = data_cfg.get("val_samples", n_samples)
    test_samples = data_cfg.get("test_samples", n_samples)
    base_seed = 0 if seed is None else seed

    train_ds = PaperSineDataset(n_samples=n_samples, mode="train", seed=base_seed)
    val_ds = PaperSineDataset(n_samples=val_samples, mode="train", seed=base_seed + 1)
    test_ds = PaperSineDataset(n_samples=test_samples, mode="test", seed=base_seed + 2)

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if eval_distributed
        else None
    )
    test_sampler = (
        DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if eval_distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader

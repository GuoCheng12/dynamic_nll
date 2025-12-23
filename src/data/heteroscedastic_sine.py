import numpy as np
import torch
from torch.utils.data import Dataset


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

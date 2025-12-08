import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPRegressor(nn.Module):
    """
    Simple MLP backbone producing mean and log_variance outputs.
    Variance is enforced positive via softplus in forward.
    """

    def __init__(self, input_dim: int, hidden_sizes: list[int], activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        act_map = {"relu": nn.ReLU(), "gelu": nn.GELU(), "tanh": nn.Tanh()}
        self.activation = act_map.get(activation, nn.ReLU())
        layers = []
        last = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(last, h))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(last, 1)
        self.log_var_head = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        mean = self.mean_head(feats)
        log_var = self.log_var_head(feats)
        variance = F.softplus(log_var) + 1e-6
        return mean, variance

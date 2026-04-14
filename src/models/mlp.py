from __future__ import annotations

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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def mean_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        return self.mean_head(feats)

    def variance_from_features(self, feats: torch.Tensor) -> torch.Tensor:
        log_var = self.log_var_head(feats)
        return F.softplus(log_var) + 1e-6

    supports_variance_trunk_scaling = True

    def forward(
        self,
        x: torch.Tensor,
        faithful: bool = False,
        variance_trunk_scale: float = 1.0,
    ):
        feats = self.forward_features(x)
        if faithful:
            var_feats = feats.detach()
        elif variance_trunk_scale >= 1.0:
            var_feats = feats
        elif variance_trunk_scale <= 0.0:
            var_feats = feats.detach()
        else:
            detached = feats.detach()
            var_feats = detached + variance_trunk_scale * (feats - detached)

        mean = self.mean_from_features(feats)
        variance = self.variance_from_features(var_feats)
        return mean, variance

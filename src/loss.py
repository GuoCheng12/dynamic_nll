from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_LOG_2PI = np.log(2 * np.pi)


class GaussianLogLikelihoodLoss(nn.Module):
    def __init__(self, beta=0):
        super(GaussianLogLikelihoodLoss, self).__init__()
        self.name = "NLL"
        self.beta = beta  # This will be updated dynamically during training
        # Name update logic can be handled externally for logging

    def forward(self, input, target, mask=None, interpolate=True, variance=None):
        """
        Args:
            input: Predicted Mean
            target: Ground Truth
            variance: Predicted Variance (sigma^2)
        """
        # [Insert Interpolation Logic if needed for Depth Estimation tasks]
        if interpolate:
            input = nn.functional.interpolate(
                input, target.shape[-2:], mode="bilinear", align_corners=True
            )
            if variance is not None:
                variance = nn.functional.interpolate(
                    variance, target.shape[-2:], mode="bilinear", align_corners=True
                )

        # [Masking Logic]
        if mask is not None:
            input = input[mask]
            target = target[mask]
            if variance is not None:
                variance = variance[mask]

        mean = input
        # Standard NLL Term
        ll = -0.5 * ((target - mean) ** 2 / variance + torch.log(variance) + _LOG_2PI)

        # Beta-NLL Mechanism: Gradient Reweighting
        if self.beta > 0:
            weight = variance.detach() ** self.beta
            ll = ll * weight

        return -torch.mean(ll)


class BetaScheduler:
    """
    Handles beta scheduling strategies for Beta-NLL.
    """

    def __init__(
        self,
        strategy: str,
        start_beta: float,
        end_beta: float,
        total_steps: int,
        warmup_steps: int = 0,
    ):
        self.strategy = strategy
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.total_steps = max(total_steps, 1)
        self.warmup_steps = max(warmup_steps, 0)
        self.best_loss = float("inf")

    def get_beta(self, current_step: int, current_loss: float | None = None) -> float:
        step = max(current_step, 0)
        if self.strategy == "constant":
            return self.start_beta
        if self.strategy == "linear_decay":
            progress = min(step / self.total_steps, 1.0)
            return self.start_beta + progress * (self.end_beta - self.start_beta)
        if self.strategy == "delayed_linear":
            if step < self.warmup_steps:
                return self.start_beta
            effective = step - self.warmup_steps
            total = max(self.total_steps - self.warmup_steps, 1)
            progress = min(effective / total, 1.0)
            return self.start_beta + progress * (self.end_beta - self.start_beta)
        if self.strategy == "cosine":
            progress = min(step / self.total_steps, 1.0)
            cosine = 0.5 * (1 + np.cos(np.pi * progress))
            return self.end_beta + (self.start_beta - self.end_beta) * cosine
        if self.strategy == "plateau":
            # Simple plateau: decay when loss stops improving; requires validation loss.
            if current_loss is not None:
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                else:
                    decay = (self.end_beta - self.start_beta) * min(
                        step / self.total_steps, 1.0
                    )
                    return self.start_beta + decay
            return self.start_beta
        return self.end_beta

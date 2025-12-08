import math
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(torch.mean((pred - target) ** 2).item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def nll(mean: torch.Tensor, target: torch.Tensor, variance: torch.Tensor) -> float:
    return (-0.5 * ((target - mean) ** 2 / variance + torch.log(variance))).mean().item()


def expected_calibration_error(pred_mean: torch.Tensor, pred_var: torch.Tensor, target: torch.Tensor, bins: int = 10) -> float:
    # Simple ECE placeholder for regression uncertainty.
    std = torch.sqrt(pred_var)
    z = torch.abs(pred_mean - target) / (std + 1e-8)
    confidences = torch.erf(z / math.sqrt(2))
    bin_boundaries = torch.linspace(0, 1, bins + 1, device=pred_mean.device)
    ece = torch.tensor(0.0, device=pred_mean.device)
    for i in range(bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.any():
            acc = (z[mask] < 1.0).float().mean()
            conf = confidences[mask].mean()
            ece += torch.abs(acc - conf) * mask.float().mean()
    return ece.item()


def grad_norms(model: torch.nn.Module) -> Dict[str, float]:
    norms: Dict[str, float] = {}
    mean_sq, var_sq = 0.0, 0.0
    mean_count, var_count = 0, 0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        g = param.grad.detach()
        if "mean" in name:
            mean_sq += g.norm().item() ** 2
            mean_count += 1
        if "var" in name or "log_var" in name:
            var_sq += g.norm().item() ** 2
            var_count += 1
    if mean_count:
        norms["grad_norm_mean"] = math.sqrt(mean_sq / mean_count)
    if var_count:
        norms["grad_norm_var"] = math.sqrt(var_sq / var_count)
    return norms

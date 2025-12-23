from __future__ import annotations

import math
import random
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device: str | None = None) -> torch.device:
    if device and device != "auto":
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return math.sqrt(torch.mean((pred - target) ** 2).item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


def nll(mean: torch.Tensor, target: torch.Tensor, variance: torch.Tensor) -> float:
    return (-0.5 * ((target - mean) ** 2 / variance + torch.log(variance))).mean().item()


def expected_calibration_error(pred_mean: torch.Tensor, pred_var: torch.Tensor, target: torch.Tensor, bins: int = 10) -> float:
    """
    Calibration proxy: confidence = exp(-0.5 z^2); accuracy = indicator(z<=1).
    Lower ECE is better.
    """
    std = torch.sqrt(pred_var + 1e-8)
    z = torch.abs(pred_mean - target) / std
    confidences = torch.exp(-0.5 * z**2)
    bin_boundaries = torch.linspace(0, 1, bins + 1, device=pred_mean.device)
    ece = torch.tensor(0.0, device=pred_mean.device)
    for i in range(bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if mask.any():
            acc = (z[mask] <= 1.0).float().mean()
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


def evaluate_head_tail(
    model: torch.nn.Module,
    dataset,
    indices: Optional[torch.Tensor] = None,
    batch_size: int = 512,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate global/head/tail metrics on a (subset of) ToyRegressionDataset.
    Head: x <= 3; Tail: x > 3. Uses unnormalized targets/preds.
    """
    model.eval()
    device = device or next(model.parameters()).device
    if indices is None:
        indices = torch.arange(len(dataset))
    x = dataset.x[indices]
    y_norm = dataset.y[indices]
    subset = torch.utils.data.TensorDataset(x, y_norm)
    loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)

    y_std = dataset.y_std
    threshold = 3.0
    metrics = {
        "MSE_Global": 0.0,
        "NLL_Global": 0.0,
        "ECE_Global": 0.0,
        "MSE_Head": 0.0,
        "NLL_Head": 0.0,
        "ECE_Head": 0.0,
        "MSE_Tail": 0.0,
        "NLL_Tail": 0.0,
        "ECE_Tail": 0.0,
    }
    counts = {"global": 0, "head": 0, "tail": 0}

    with torch.no_grad():
        for xb, yb_norm in loader:
            xb = xb.to(device)
            yb_norm = yb_norm.to(device)
            mean_norm, var_norm = model(xb)
            mean_norm = mean_norm.squeeze()
            var_norm = var_norm.squeeze()
            target_norm = yb_norm.squeeze()

            mean = dataset.unnormalize(mean_norm.cpu())
            target = dataset.unnormalize(target_norm.cpu())
            var = var_norm.cpu() * (y_std**2)

            batch = len(xb)
            counts["global"] += batch
            metrics["MSE_Global"] += torch.mean((mean - target) ** 2).item() * batch
            metrics["NLL_Global"] += nll(mean, target, var) * batch
            metrics["ECE_Global"] += expected_calibration_error(mean, var, target) * batch

            head_mask = (xb.cpu().squeeze() <= threshold)
            tail_mask = (xb.cpu().squeeze() > threshold)

            if head_mask.any():
                hm = head_mask
                counts["head"] += hm.sum().item()
                metrics["MSE_Head"] += torch.mean((mean[hm] - target[hm]) ** 2).item() * hm.sum().item()
                metrics["NLL_Head"] += nll(mean[hm], target[hm], var[hm]) * hm.sum().item()
                metrics["ECE_Head"] += expected_calibration_error(mean[hm], var[hm], target[hm]) * hm.sum().item()

            if tail_mask.any():
                tm = tail_mask
                counts["tail"] += tm.sum().item()
                metrics["MSE_Tail"] += torch.mean((mean[tm] - target[tm]) ** 2).item() * tm.sum().item()
                metrics["NLL_Tail"] += nll(mean[tm], target[tm], var[tm]) * tm.sum().item()
                metrics["ECE_Tail"] += expected_calibration_error(mean[tm], var[tm], target[tm]) * tm.sum().item()

    metrics["MSE_Global"] /= max(counts["global"], 1)
    metrics["NLL_Global"] /= max(counts["global"], 1)
    metrics["ECE_Global"] /= max(counts["global"], 1)
    metrics["MSE_Head"] = metrics["MSE_Head"] / max(counts["head"], 1)
    metrics["NLL_Head"] = metrics["NLL_Head"] / max(counts["head"], 1)
    metrics["ECE_Head"] = metrics["ECE_Head"] / max(counts["head"], 1)
    metrics["MSE_Tail"] = metrics["MSE_Tail"] / max(counts["tail"], 1)
    metrics["NLL_Tail"] = metrics["NLL_Tail"] / max(counts["tail"], 1)
    metrics["ECE_Tail"] = metrics["ECE_Tail"] / max(counts["tail"], 1)

    return metrics

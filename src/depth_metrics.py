from __future__ import annotations

from typing import Dict

import torch


def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute standard depth estimation metrics.
    pred/target: tensors with shape [B, 1, H, W].
    """
    eps = 1e-6
    pred = pred.squeeze(1)
    target = target.squeeze(1)
    mask = target > 0
    pred = pred[mask].clamp(min=eps)
    target = target[mask].clamp(min=eps)

    thresh = torch.max(target / pred, pred / target)
    delta1 = (thresh < 1.25).float().mean().item()
    delta2 = (thresh < 1.25 ** 2).float().mean().item()
    delta3 = (thresh < 1.25 ** 3).float().mean().item()

    abs_rel = torch.mean(torch.abs(target - pred) / target).item()
    rmse = torch.sqrt(torch.mean((target - pred) ** 2)).item()
    log10 = torch.mean(torch.abs(torch.log10(target) - torch.log10(pred))).item()

    return {
        "rmse": rmse,
        "abs_rel": abs_rel,
        "log10": log10,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }

from __future__ import annotations

from typing import Dict

import torch


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
    use_eigen_crop: bool = True,
) -> Dict[str, float]:
    """
    Compute standard depth estimation metrics.
    pred/target: tensors with shape [B, 1, H, W].
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    mask = (target > min_depth) & (target < max_depth)
    if use_eigen_crop and target.shape[-2:] == (480, 640):
        crop_mask = torch.zeros_like(mask, dtype=torch.bool)
        crop_mask[:, 45:471, 41:601] = True
        mask = mask & crop_mask

    pred_val = pred[mask]
    target_val = target[mask]
    if target_val.numel() == 0:
        return {"rmse": 0.0, "abs_rel": 0.0, "log10": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}

    pred_val = pred_val.clamp(min=min_depth, max=max_depth)
    target_val = target_val.clamp(min=min_depth, max=max_depth)

    thresh = torch.max(target_val / pred_val, pred_val / target_val)
    delta1 = (thresh < 1.25).float().mean().item()
    delta2 = (thresh < 1.25 ** 2).float().mean().item()
    delta3 = (thresh < 1.25 ** 3).float().mean().item()

    abs_rel = torch.mean(torch.abs(target_val - pred_val) / target_val).item()
    rmse = torch.sqrt(torch.mean((target_val - pred_val) ** 2)).item()
    log10 = torch.mean(torch.abs(torch.log10(target_val) - torch.log10(pred_val))).item()

    return {
        "rmse": rmse,
        "abs_rel": abs_rel,
        "log10": log10,
        "delta1": delta1,
        "delta2": delta2,
        "delta3": delta3,
    }

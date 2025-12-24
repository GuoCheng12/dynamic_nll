from __future__ import annotations

from typing import Dict
import warnings

import torch


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 10.0,
    use_eigen_crop: bool = True,
    dataset: str = "nyu_depth",
) -> Dict[str, float]:
    """
    Compute standard depth estimation metrics.
    pred/target: tensors with shape [B, 1, H, W].
    """
    eps = 1e-6
    if pred.shape[-2:] != target.shape[-2:]:
        pred = torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=True)
    pred = pred.squeeze(1)
    target = target.squeeze(1)

    mask = (target > min_depth) & (target < max_depth)
    if use_eigen_crop and dataset == "nyu_depth":
        height, width = target.shape[-2], target.shape[-1]
        if height == 480 and width == 640:
            crop_mask = torch.zeros_like(mask, dtype=torch.bool)
            crop_mask[:, 45:471, 41:601] = True
            mask = mask & crop_mask
        else:
            warnings.warn(
                f"Skipping Eigen crop because target size is {height}x{width}, expected 480x640.",
                RuntimeWarning,
            )

    pred = pred[mask].clamp(min=min_depth, max=max_depth)
    target = target[mask].clamp(min=min_depth, max=max_depth)
    if target.numel() == 0:
        return {"rmse": 0.0, "abs_rel": 0.0, "log10": 0.0, "delta1": 0.0, "delta2": 0.0, "delta3": 0.0}

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

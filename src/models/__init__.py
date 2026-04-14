from __future__ import annotations

from .mlp import MLPRegressor

__all__ = ["MLPRegressor", "DepthUNet"]


def __getattr__(name: str):
    if name == "DepthUNet":
        from .depth_unet import DepthUNet

        return DepthUNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

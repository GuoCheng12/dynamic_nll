from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass
class ClosedLoopControllerState:
    beta_t: float
    gamma_t: float
    rmse_slope: float
    controller_event: str
    grad_ratio: float | None = None
    grad_cosine: float | None = None
    mean_grad_norm: float | None = None
    var_grad_norm: float | None = None


class ClosedLoopCouplingController:
    """
    Rule-based controller for adaptive beta and variance-to-trunk coupling.

    The controller is updated after validation and its state is used for the
    next epoch. `beta_only` keeps gamma fixed at 1.0; `beta_gamma` gates the
    variance branch's trunk gradient via the model.
    """

    def __init__(
        self,
        mode: str,
        signal: str,
        update_interval: int,
        warmup_epochs: int,
        beta_min: float,
        beta_max: float,
        gamma_min: float,
        gamma_max: float,
        beta_step: float,
        gamma_step: float,
        collapse_thresh: float,
        align_thresh: float,
        rmse_plateau_thresh: float,
        grad_ratio_thresh: float,
        grad_cosine_thresh: float,
        gamma_release_beta_thresh: float,
        supports_gamma: bool,
        fixed_gamma_t: float | None = None,
    ):
        if mode not in {"beta_only", "beta_gamma"}:
            raise ValueError(f"Unsupported controller mode: {mode}")
        if signal not in {"metrics", "gradient"}:
            raise ValueError(f"Unsupported controller signal: {signal}")
        self.mode = mode
        self.signal = signal
        self.update_interval = max(int(update_interval), 1)
        self.warmup_epochs = max(int(warmup_epochs), 0)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.beta_step = beta_step
        self.gamma_step = gamma_step
        self.collapse_thresh = collapse_thresh
        self.align_thresh = align_thresh
        self.rmse_plateau_thresh = rmse_plateau_thresh
        self.grad_ratio_thresh = grad_ratio_thresh
        self.grad_cosine_thresh = grad_cosine_thresh
        self.gamma_release_beta_thresh = gamma_release_beta_thresh
        self.supports_gamma = supports_gamma
        self.fixed_gamma_t = fixed_gamma_t
        self.uses_gamma = mode == "beta_gamma" and supports_gamma and fixed_gamma_t is None

        self.beta_t = beta_max
        if fixed_gamma_t is not None:
            self.gamma_t = fixed_gamma_t
        else:
            self.gamma_t = gamma_min if self.uses_gamma else 1.0
        self.rmse_history: Deque[float] = deque(maxlen=4)
        self.collapse_event_count = 0
        self.first_beta_release_epoch: int | None = None
        self.first_gamma_release_epoch: int | None = None

    def state(self) -> tuple[float, float]:
        return self.beta_t, self.gamma_t

    def _is_controller_step(self, epoch: int) -> bool:
        return (epoch + 1) % self.update_interval == 0

    def uses_gradient_signal(self) -> bool:
        return self.signal == "gradient"

    def _rmse_slope_for_rules(self) -> float:
        if len(self.rmse_history) < 4:
            return float("inf")
        values = list(self.rmse_history)
        drops = [values[idx] - values[idx + 1] for idx in range(len(values) - 1)]
        return sum(drops[-3:]) / 3.0

    def observe(
        self,
        epoch: int,
        rmse_val: float,
        collapse_ratio: float,
        align_score: float,
        grad_ratio: float | None = None,
        grad_cosine: float | None = None,
        mean_grad_norm: float | None = None,
        var_grad_norm: float | None = None,
    ) -> ClosedLoopControllerState:
        if not self._is_controller_step(epoch):
            event = "warmup" if (epoch + 1) <= self.warmup_epochs else "hold"
            return ClosedLoopControllerState(
                beta_t=self.beta_t,
                gamma_t=self.gamma_t,
                rmse_slope=self._rmse_slope_for_rules(),
                controller_event=event,
                grad_ratio=grad_ratio,
                grad_cosine=grad_cosine,
                mean_grad_norm=mean_grad_norm,
                var_grad_norm=var_grad_norm,
            )

        self.rmse_history.append(rmse_val)
        rmse_slope = self._rmse_slope_for_rules()

        if (epoch + 1) <= self.warmup_epochs:
            self.beta_t = self.beta_max
            if self.uses_gamma:
                self.gamma_t = self.gamma_min
            event = "warmup"
        elif collapse_ratio > self.collapse_thresh:
            self.beta_t = min(self.beta_max, self.beta_t + self.beta_step)
            if self.uses_gamma:
                self.gamma_t = max(self.gamma_min, self.gamma_t - self.gamma_step)
            self.collapse_event_count += 1
            event = "safe_fallback"
        elif self.signal == "gradient":
            if grad_ratio is None or grad_cosine is None:
                event = "hold"
            elif grad_cosine < self.grad_cosine_thresh or grad_ratio > self.grad_ratio_thresh:
                event = "hold"
            elif self.uses_gamma and self.beta_t <= self.gamma_release_beta_thresh and self.gamma_t < self.gamma_max:
                next_gamma = min(self.gamma_max, self.gamma_t + self.gamma_step)
                if next_gamma > self.gamma_t:
                    self.gamma_t = next_gamma
                    if self.first_gamma_release_epoch is None:
                        self.first_gamma_release_epoch = epoch + 1
                    event = "gamma_release"
                else:
                    event = "hold"
            else:
                next_beta = max(self.beta_min, self.beta_t - self.beta_step)
                if next_beta < self.beta_t:
                    self.beta_t = next_beta
                    if self.first_beta_release_epoch is None:
                        self.first_beta_release_epoch = epoch + 1
                    event = "beta_release"
                else:
                    event = "hold"
        elif rmse_slope > self.rmse_plateau_thresh:
            event = "hold"
        elif align_score >= self.align_thresh and collapse_ratio <= self.collapse_thresh:
            if self.uses_gamma and self.beta_t <= self.gamma_release_beta_thresh and self.gamma_t < self.gamma_max:
                next_gamma = min(self.gamma_max, self.gamma_t + self.gamma_step)
                if next_gamma > self.gamma_t:
                    self.gamma_t = next_gamma
                    if self.first_gamma_release_epoch is None:
                        self.first_gamma_release_epoch = epoch + 1
                    event = "gamma_release"
                else:
                    event = "hold"
            else:
                next_beta = max(self.beta_min, self.beta_t - self.beta_step)
                if next_beta < self.beta_t:
                    self.beta_t = next_beta
                    if self.first_beta_release_epoch is None:
                        self.first_beta_release_epoch = epoch + 1
                    event = "beta_release"
                else:
                    event = "hold"
        else:
            event = "hold"

        return ClosedLoopControllerState(
            beta_t=self.beta_t,
            gamma_t=self.gamma_t,
            rmse_slope=rmse_slope,
            controller_event=event,
            grad_ratio=grad_ratio,
            grad_cosine=grad_cosine,
            mean_grad_norm=mean_grad_norm,
            var_grad_norm=var_grad_norm,
        )

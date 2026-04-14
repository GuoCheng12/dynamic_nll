from __future__ import annotations

import torch

from src.models import MLPRegressor
from src.modules import BetaScheduler, ClosedLoopCouplingController, FaithfulHeteroscedasticLoss


def _is_missing_or_zero(grad: torch.Tensor | None) -> bool:
    return grad is None or torch.count_nonzero(grad) == 0


def test_beta_scheduler_hits_endpoints() -> None:
    scheduler = BetaScheduler(
        strategy="linear_decay",
        start_beta=1.0,
        end_beta=0.0,
        total_steps=5,
    )

    values = [scheduler.get_beta(step) for step in range(5)]

    assert values[0] == 1.0
    assert values[-1] == 0.0
    assert values[2] == 0.5


def test_faithful_loss_detaches_mean_in_nll_term() -> None:
    criterion = FaithfulHeteroscedasticLoss(lambda_weight=1.0)
    mean = torch.tensor([[1.0]], requires_grad=True)
    variance = torch.tensor([[2.0]], requires_grad=True)
    target = torch.tensor([[0.0]])

    loss = criterion(mean, target, variance=variance, interpolate=False)
    loss.backward()

    assert torch.allclose(mean.grad, torch.tensor([[1.0]]))
    assert variance.grad is not None
    assert torch.count_nonzero(variance.grad) > 0


def test_faithful_objective_blocks_trunk_grad_from_variance_path() -> None:
    torch.manual_seed(0)
    model = MLPRegressor(input_dim=1, hidden_sizes=[8], activation="relu")
    criterion = FaithfulHeteroscedasticLoss(lambda_weight=1.0, mse_weight=0.0)
    x = torch.tensor([[0.5]])
    y = torch.tensor([[1.0]])

    mean, variance = model(x, faithful=True)
    loss = criterion(mean, y, variance=variance, interpolate=False)
    loss.backward()

    backbone_grads = [param.grad for param in model.backbone.parameters()]
    mean_head_grads = [param.grad for param in model.mean_head.parameters()]
    var_head_grads = [param.grad for param in model.log_var_head.parameters()]

    assert all(_is_missing_or_zero(grad) for grad in backbone_grads)
    assert all(_is_missing_or_zero(grad) for grad in mean_head_grads)
    assert all(grad is not None and torch.count_nonzero(grad) > 0 for grad in var_head_grads)


def test_variance_trunk_scale_zero_blocks_backbone_grad_without_changing_forward_value() -> None:
    torch.manual_seed(0)
    model = MLPRegressor(input_dim=1, hidden_sizes=[8], activation="relu")
    x = torch.tensor([[0.5]])

    mean_full, variance_full = model(x, faithful=False, variance_trunk_scale=1.0)
    mean_blocked, variance_blocked = model(x, faithful=False, variance_trunk_scale=0.0)

    assert torch.allclose(mean_full, mean_blocked)
    assert torch.allclose(variance_full, variance_blocked)

    loss = variance_blocked.sum()
    loss.backward()

    backbone_grads = [param.grad for param in model.backbone.parameters()]
    var_head_grads = [param.grad for param in model.log_var_head.parameters()]

    assert all(_is_missing_or_zero(grad) for grad in backbone_grads)
    assert all(grad is not None and torch.count_nonzero(grad) > 0 for grad in var_head_grads)


def test_faithful_loss_with_gamma_one_restores_variance_grad_to_backbone() -> None:
    torch.manual_seed(0)
    model = MLPRegressor(input_dim=1, hidden_sizes=[8], activation="relu")
    criterion = FaithfulHeteroscedasticLoss(lambda_weight=1.0, mse_weight=0.0)
    x = torch.tensor([[0.5]])
    y = torch.tensor([[1.0]])

    mean, variance = model(x, faithful=False, variance_trunk_scale=1.0)
    loss = criterion(mean, y, variance=variance, interpolate=False)
    loss.backward()

    backbone_grads = [param.grad for param in model.backbone.parameters()]
    mean_head_grads = [param.grad for param in model.mean_head.parameters()]
    var_head_grads = [param.grad for param in model.log_var_head.parameters()]

    assert all(grad is not None and torch.count_nonzero(grad) > 0 for grad in backbone_grads)
    assert all(_is_missing_or_zero(grad) for grad in mean_head_grads)
    assert all(grad is not None and torch.count_nonzero(grad) > 0 for grad in var_head_grads)


def test_closed_loop_controller_warmup_and_release_rules() -> None:
    controller = ClosedLoopCouplingController(
        mode="beta_gamma",
        signal="metrics",
        update_interval=2,
        warmup_epochs=4,
        beta_min=0.2,
        beta_max=1.0,
        gamma_min=0.0,
        gamma_max=1.0,
        beta_step=0.1,
        gamma_step=0.1,
        collapse_thresh=0.02,
        align_thresh=0.30,
        rmse_plateau_thresh=0.005,
        grad_ratio_thresh=1.0,
        grad_cosine_thresh=0.0,
        gamma_release_beta_thresh=0.6,
        supports_gamma=True,
    )

    state = controller.observe(epoch=1, rmse_val=1.0, collapse_ratio=0.0, align_score=0.5)
    assert state.controller_event == "warmup"
    assert state.beta_t == 1.0
    assert state.gamma_t == 0.0

    state = controller.observe(epoch=5, rmse_val=0.9, collapse_ratio=0.05, align_score=0.5)
    assert state.controller_event == "safe_fallback"
    assert state.beta_t == 1.0
    assert state.gamma_t == 0.0

    controller.rmse_history.extend([1.0, 0.998, 0.997, 0.996])
    controller.beta_t = 0.7
    state = controller.observe(epoch=7, rmse_val=0.996, collapse_ratio=0.0, align_score=0.5)
    assert state.controller_event == "beta_release"
    assert state.beta_t == 0.6

    controller.rmse_history.clear()
    controller.rmse_history.extend([1.0, 0.999, 0.998, 0.997])
    state = controller.observe(epoch=9, rmse_val=0.997, collapse_ratio=0.0, align_score=0.5)
    assert state.controller_event == "gamma_release"
    assert state.gamma_t == 0.1


def test_closed_loop_controller_respects_fixed_gamma() -> None:
    controller = ClosedLoopCouplingController(
        mode="beta_only",
        signal="gradient",
        update_interval=1,
        warmup_epochs=0,
        beta_min=0.2,
        beta_max=1.0,
        gamma_min=0.0,
        gamma_max=1.0,
        beta_step=0.1,
        gamma_step=0.1,
        collapse_thresh=0.02,
        align_thresh=0.30,
        rmse_plateau_thresh=0.005,
        grad_ratio_thresh=1.0,
        grad_cosine_thresh=0.0,
        gamma_release_beta_thresh=0.6,
        supports_gamma=True,
        fixed_gamma_t=0.0,
    )

    state = controller.observe(
        epoch=0,
        rmse_val=1.0,
        collapse_ratio=0.0,
        align_score=0.0,
        grad_ratio=0.5,
        grad_cosine=0.25,
        mean_grad_norm=1.0,
        var_grad_norm=0.5,
    )
    assert state.controller_event == "beta_release"
    assert state.beta_t == 0.9
    assert state.gamma_t == 0.0


def test_closed_loop_controller_gradient_signal_releases_on_aligned_probe() -> None:
    controller = ClosedLoopCouplingController(
        mode="beta_gamma",
        signal="gradient",
        update_interval=1,
        warmup_epochs=0,
        beta_min=0.2,
        beta_max=1.0,
        gamma_min=0.0,
        gamma_max=1.0,
        beta_step=0.1,
        gamma_step=0.1,
        collapse_thresh=0.02,
        align_thresh=0.30,
        rmse_plateau_thresh=0.005,
        grad_ratio_thresh=1.0,
        grad_cosine_thresh=0.0,
        gamma_release_beta_thresh=0.6,
        supports_gamma=True,
    )

    state = controller.observe(
        epoch=0,
        rmse_val=1.0,
        collapse_ratio=0.0,
        align_score=0.0,
        grad_ratio=0.5,
        grad_cosine=0.25,
        mean_grad_norm=1.0,
        var_grad_norm=0.5,
    )
    assert state.controller_event == "beta_release"
    assert state.beta_t == 0.9

    controller.beta_t = 0.6
    state = controller.observe(
        epoch=1,
        rmse_val=1.0,
        collapse_ratio=0.0,
        align_score=0.0,
        grad_ratio=0.4,
        grad_cosine=0.1,
        mean_grad_norm=1.0,
        var_grad_norm=0.4,
    )
    assert state.controller_event == "gamma_release"
    assert state.gamma_t == 0.1

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from src.data import build_dataloaders


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    if x.numel() < 2 or y.numel() < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.linalg.norm(x) * torch.linalg.norm(y)
    if denom.item() <= 1e-12:
        return 0.0
    return torch.dot(x, y).item() / denom.item()


def compute_toy_epoch_statistics(
    squared_error: torch.Tensor,
    variance: torch.Tensor,
    true_variance: torch.Tensor | None = None,
    collapse_threshold: float = 1e-4,
) -> Dict[str, float]:
    variance = variance.reshape(-1).float()
    squared_error = squared_error.reshape(-1).float()
    stats = {
        "collapse_ratio": torch.mean((variance < collapse_threshold).float()).item(),
        "align_score": pearson_corr(squared_error, variance),
        "sigma2_min": torch.min(variance).item(),
        "sigma2_median": torch.median(variance).item(),
        "sigma2_max": torch.max(variance).item(),
    }
    if true_variance is not None:
        true_variance = true_variance.reshape(-1).float()
        stats["true_var_corr"] = pearson_corr(true_variance, variance)
        stats["var_mse"] = torch.mean((variance - true_variance) ** 2).item()
    return stats


def summarize_toy_history(history: List[Dict[str, Any]], supports_gamma: bool) -> Dict[str, Any]:
    first_beta_release = next(
        (row["epoch"] for row in history if row.get("controller_event") == "beta_release"),
        -1,
    )
    first_gamma_release = next(
        (row["epoch"] for row in history if row.get("controller_event") == "gamma_release"),
        -1,
    )
    collapse_events = sum(1 for row in history if row.get("controller_event") == "safe_fallback")
    final_align = history[-1]["align_score"] if history else 0.0
    summary = {
        "final_align_score": final_align,
        "collapse_event_count": collapse_events,
        "first_beta_release_epoch": first_beta_release,
        "first_gamma_release_epoch": first_gamma_release,
        "supports_gamma": supports_gamma,
    }
    if history and history[-1].get("lambda_t") is not None:
        summary["first_lambda_release_epoch"] = first_beta_release
        summary["final_lambda_t"] = history[-1]["lambda_t"]
    if history and "grad_ratio" in history[-1]:
        summary["final_grad_ratio"] = history[-1]["grad_ratio"]
    if history and "grad_cosine" in history[-1]:
        summary["final_grad_cosine"] = history[-1]["grad_cosine"]
    if history and "mean_grad_norm" in history[-1]:
        summary["final_mean_grad_norm"] = history[-1]["mean_grad_norm"]
    if history and "var_grad_norm" in history[-1]:
        summary["final_var_grad_norm"] = history[-1]["var_grad_norm"]
    if history and "true_var_corr" in history[-1]:
        summary["final_true_var_corr"] = history[-1]["true_var_corr"]
    if history and "var_mse" in history[-1]:
        summary["final_var_mse"] = history[-1]["var_mse"]
    return summary


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, float):
        if value == float("inf") or value == float("-inf"):
            return None
        if value != value:
            return None
    return value


def _sanitize_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{key: _sanitize_value(value) for key, value in row.items()} for row in rows]


def save_toy_history_artifacts(
    run_dir: str | Path,
    history: List[Dict[str, Any]],
    summary: Dict[str, Any],
    prefix: str = "toy",
) -> None:
    run_dir = Path(run_dir)
    history_path = run_dir / f"{prefix}_epoch_history.json"
    csv_path = run_dir / f"{prefix}_epoch_history.csv"
    summary_path = run_dir / f"{prefix}_controller_summary.json"
    generic_history_path = run_dir / "controller_epoch_history.json"
    generic_csv_path = run_dir / "controller_epoch_history.csv"
    generic_summary_path = run_dir / "controller_summary.json"

    safe_history = _sanitize_rows(history)
    with history_path.open("w", encoding="utf-8") as fh:
        json.dump(safe_history, fh, indent=2)
    with generic_history_path.open("w", encoding="utf-8") as fh:
        json.dump(safe_history, fh, indent=2)

    if safe_history:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(safe_history[0].keys()))
            writer.writeheader()
            writer.writerows(safe_history)
        with generic_csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(safe_history[0].keys()))
            writer.writeheader()
            writer.writerows(safe_history)

    safe_summary = {key: _sanitize_value(value) for key, value in summary.items()}
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(safe_summary, fh, indent=2)
    with generic_summary_path.open("w", encoding="utf-8") as fh:
        json.dump(safe_summary, fh, indent=2)


def plot_toy_history(run_dir: str | Path, history: List[Dict[str, Any]], prefix: str = "toy") -> None:
    if not history:
        return

    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    epochs = [row["epoch"] for row in history]
    primary_key = "lambda_t" if any(row.get("lambda_t") is not None for row in history) else "beta_t"
    primary_label = "lambda_t" if primary_key == "lambda_t" else "beta_t"
    beta = [row[primary_key] if row.get(primary_key) is not None else row["beta_t"] for row in history]
    gamma = [row["gamma_t"] for row in history]
    rmse_val = [row["rmse_val"] for row in history]
    collapse_ratio = [row["collapse_ratio"] for row in history]
    align_score = [row["align_score"] for row in history]
    sigma2_min = [row["sigma2_min"] for row in history]
    sigma2_med = [row["sigma2_median"] for row in history]
    sigma2_max = [row["sigma2_max"] for row in history]

    def save_simple_plot(filename: str, series: List[tuple[List[float], str, str]], ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for values, label, color in series:
            ax.plot(epochs, values, color=color, label=label)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / filename, dpi=200)
        plt.close(fig)

    save_simple_plot(
        f"{prefix}_beta_gamma_vs_epoch.png",
        [(beta, primary_label, "steelblue"), (gamma, "gamma_t", "darkorange")],
        "controller value",
    )
    save_simple_plot(f"{prefix}_rmse_val_vs_epoch.png", [(rmse_val, "rmse_val", "firebrick")], "rmse")
    save_simple_plot(
        f"{prefix}_collapse_ratio_vs_epoch.png",
        [(collapse_ratio, "collapse_ratio", "purple")],
        "collapse ratio",
    )
    save_simple_plot(
        f"{prefix}_align_score_vs_epoch.png",
        [(align_score, "align_score", "forestgreen")],
        "correlation",
    )
    save_simple_plot(
        f"{prefix}_sigma2_stats_vs_epoch.png",
        [
            (sigma2_min, "sigma2_min", "black"),
            (sigma2_med, "sigma2_median", "royalblue"),
            (sigma2_max, "sigma2_max", "goldenrod"),
        ],
        "sigma^2",
    )


def plot_toy_predictions(
    run_dir: str | Path,
    cfg,
    model: torch.nn.Module,
    device: torch.device,
    history: List[Dict[str, Any]],
    faithful: bool,
    variance_trunk_scale: float,
) -> None:
    import matplotlib.pyplot as plt

    _, _, test_loader = build_dataloaders(
        batch_size=cfg.hyperparameters.batch_size,
        splits=(
            float(cfg.dataset.get("split", {}).get("train", 0.8)),
            float(cfg.dataset.get("split", {}).get("val", 0.1)),
            float(cfg.dataset.get("split", {}).get("test", 0.1)),
        ),
        size=cfg.dataset.get("size", 5000),
        normalize=cfg.dataset.get("normalize", True),
        seed=cfg.seed,
        eval_batch_size=cfg.hyperparameters.get("eval_batch_size", cfg.hyperparameters.batch_size),
    )
    dataset = test_loader.dataset.dataset
    test_indices = torch.tensor(test_loader.dataset.indices)
    x = dataset.x[test_indices].squeeze()
    target = dataset.unnormalize(dataset.y[test_indices].squeeze())

    model.eval()
    with torch.no_grad():
        mean_norm, variance_norm = model(
            dataset.x[test_indices].to(device),
            faithful=faithful,
            variance_trunk_scale=variance_trunk_scale,
        )
    mean = dataset.unnormalize(mean_norm.squeeze().cpu())
    variance = variance_norm.squeeze().cpu() * (dataset.y_std**2)
    std = torch.sqrt(variance.clamp(min=1e-8))

    order = torch.argsort(x)
    x_sorted = x[order]
    target_sorted = target[order]
    mean_sorted = mean[order]
    std_sorted = std[order]

    change_epochs = [
        row["epoch"]
        for row in history
        if row.get("controller_event") in {"safe_fallback", "beta_release", "gamma_release"}
    ]
    change_text = "none" if not change_epochs else ", ".join(str(epoch) for epoch in change_epochs[:10])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(x.numpy(), target.numpy(), color="gray", alpha=0.25, s=12, label="test")
    axes[0].plot(x_sorted.numpy(), mean_sorted.numpy(), color="steelblue", label="pred mean")
    axes[0].fill_between(
        x_sorted.numpy(),
        (mean_sorted - 2 * std_sorted).numpy(),
        (mean_sorted + 2 * std_sorted).numpy(),
        color="steelblue",
        alpha=0.18,
        label="pred ± 2σ",
    )
    axes[0].set_title("Toy Mean / Uncertainty")
    axes[0].legend()

    axes[1].plot(x_sorted.numpy(), std_sorted.pow(2).numpy(), color="darkorange", label="pred variance")
    axes[1].set_title("Toy Variance Curve")
    axes[1].legend()
    axes[1].text(
        0.02,
        0.98,
        f"controller change epochs: {change_text}",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "black"},
    )

    for ax in axes:
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].set_ylabel("sigma^2")

    fig.tight_layout()
    fig.savefig(Path(run_dir) / "toy_prediction.png", dpi=200)
    plt.close(fig)

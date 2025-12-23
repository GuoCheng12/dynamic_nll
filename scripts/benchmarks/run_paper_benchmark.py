import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import PaperSineDataset
from src.modules import BetaScheduler, GaussianLogLikelihoodLoss
from src.models import MLPRegressor
from src.utils import mae, nll, rmse, set_seed, expected_calibration_error


def make_loaders(n_samples: int, batch_size: int, seed: int = 0):
    train_ds = PaperSineDataset(n_samples=n_samples, mode="train", seed=seed)
    test_ds = PaperSineDataset(n_samples=n_samples, mode="test", seed=seed)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_ds, test_ds, train_loader, test_loader


def train_model(
    beta_start: float,
    beta_end: float,
    dynamic: bool,
    strategy: str,
    warmup_epochs: int,
    lr: float,
    lr_stage2: float,
    stage1_epochs: int,
    epochs: int,
    train_loader,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    model = MLPRegressor(input_dim=1, hidden_sizes=[128, 128], activation="relu").to(device)
    criterion = GaussianLogLikelihoodLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = BetaScheduler(
        strategy=strategy,
        start_beta=beta_start,
        end_beta=beta_end,
        total_steps=max(epochs, 1),
        warmup_steps=warmup_epochs,
    )

    for epoch in range(epochs):
        model.train()
        # LR/β schedule
        if dynamic:
            if epoch >= stage1_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_stage2
            epoch_beta = scheduler.get_beta(epoch)
        else:
            epoch_beta = beta_start

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            criterion.beta = epoch_beta
            mean, variance = model(data)
            loss = criterion(mean, target, variance=variance, interpolate=False)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, {}


def evaluate(model, test_loader, true_var: torch.Tensor) -> Dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    metrics = {"MSE": 0.0, "MAE": 0.0, "RMSE": 0.0, "NLL": 0.0, "Var_MSE": 0.0, "ECE": 0.0}
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            mean, variance = model(data)
            batch = len(data)
            true_var_batch = true_var[count : count + batch].to(device)
            metrics["MSE"] += torch.mean((mean - target) ** 2).item() * batch
            metrics["MAE"] += mae(mean, target) * batch
            metrics["RMSE"] += rmse(mean, target) * batch
            metrics["NLL"] += nll(mean, target, variance) * batch
            metrics["Var_MSE"] += torch.mean((variance - true_var_batch) ** 2).item() * batch
            metrics["ECE"] += expected_calibration_error(mean, variance, target) * batch
            count += batch
    for k in metrics:
        metrics[k] /= max(count, 1)
    return metrics


def plot_envelopes(models: Dict[str, torch.nn.Module], test_ds: PaperSineDataset, output: str):
    x = test_ds.x.squeeze()
    order = torch.argsort(x)
    x_sorted = x[order].numpy()
    true_mean = np.squeeze(test_ds.true_mean[order].numpy())
    true_sigma = np.squeeze(torch.sqrt(test_ds.true_var[order]).numpy())

    plt.figure(figsize=(10, 6))
    plt.plot(x_sorted, true_mean, color="black", linestyle="--", label="True mean")
    plt.fill_between(
        x_sorted,
        true_mean - 2 * true_sigma,
        true_mean + 2 * true_sigma,
        color="gray",
        alpha=0.2,
        label="True ±2σ",
    )

    colors = {
        "Beta0": "red",
        "Beta05": "orange",
        "Beta1": "red",
        "Dynamic_Fix": "blue",
        "Dynamic_Soft": "green",
        "Dynamic_Conservative": "blue",
        "Dynamic_Reverse_Fair": "purple",
    }
    device = next(iter(models.values())).parameters().__next__().device
    with torch.no_grad():
        for name, model in models.items():
            model.eval()
            mean, var = model(test_ds.x.to(device))
            mean_np = np.squeeze(mean.detach().cpu().numpy())
            var_np = np.squeeze(var.detach().cpu().numpy())
            mean_sorted = np.array(mean_np[order.numpy()]).reshape(-1)
            std_sorted = np.sqrt(np.array(var_np[order.numpy()]).reshape(-1))
            plt.plot(x_sorted, mean_sorted, color=colors.get(name, "purple"), label=f"{name} mean")
            plt.fill_between(
                x_sorted,
                mean_sorted - 2 * std_sorted,
                mean_sorted + 2 * std_sorted,
                color=colors.get(name, "purple"),
                alpha=0.15,
            )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Predicted vs True Uncertainty (Paper Sine)")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved envelope plot to {output}")


def plot_comparison(models: Dict[str, torch.nn.Module], test_ds: PaperSineDataset, names: Tuple[str, str], output: str):
    x = test_ds.x.squeeze()
    order = torch.argsort(x)
    x_sorted = x[order].numpy()
    true_mean = np.squeeze(test_ds.true_mean[order].numpy())

    plt.figure(figsize=(10, 5))
    plt.plot(x_sorted, true_mean, color="black", linestyle="--", label="True mean")
    colors = {names[0]: "red", names[1]: "blue"}
    device = next(iter(models.values())).parameters().__next__().device
    with torch.no_grad():
        for name in names:
            model = models[name]
            mean, var = model(test_ds.x.to(device))
            mean_np = np.squeeze(mean.detach().cpu().numpy())
            var_np = np.squeeze(var.detach().cpu().numpy())
            mean_sorted = np.array(mean_np[order.numpy()]).reshape(-1)
            std_sorted = np.sqrt(np.array(var_np[order.numpy()]).reshape(-1))
            plt.plot(x_sorted, mean_sorted, color=colors.get(name, "gray"), label=f"{name} mean")
            plt.fill_between(
                x_sorted,
                mean_sorted - 2 * std_sorted,
                mean_sorted + 2 * std_sorted,
                color=colors.get(name, "gray"),
                alpha=0.15,
            )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Mean Collapse Comparison ({names[0]} vs {names[1]})")
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.savefig(output, dpi=200)
    print(f"Saved comparison plot to {output}")


def plot_final(models: Dict[str, torch.nn.Module], test_ds: PaperSineDataset, output: str):
    if "Beta1" not in models or "Dynamic_Conservative" not in models:
        print("Skipping final plot; required models missing.")
        return
    x = test_ds.x.squeeze()
    order = torch.argsort(x)
    x_sorted = x[order].numpy()
    true_mean = np.squeeze(test_ds.true_mean[order].numpy())
    true_sigma = np.squeeze(torch.sqrt(test_ds.true_var[order]).numpy())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(x_sorted, true_mean, color="black", linestyle="--", label="True mean")
    axes[0].fill_between(
        x_sorted,
        true_mean - 2 * true_sigma,
        true_mean + 2 * true_sigma,
        color="gray",
        alpha=0.2,
        label="True ±2σ",
    )

    colors = {"Beta1": "red", "Dynamic_Conservative": "blue"}
    device = next(iter(models.values())).parameters().__next__().device
    with torch.no_grad():
        for name in ["Beta1", "Dynamic_Conservative"]:
            model = models[name]
            mean, var = model(test_ds.x.to(device))
            mean_np = np.squeeze(mean.detach().cpu().numpy())
            var_np = np.squeeze(var.detach().cpu().numpy())
            mean_sorted = np.array(mean_np[order.numpy()]).reshape(-1)
            std_sorted = np.sqrt(np.array(var_np[order.numpy()]).reshape(-1))
            axes[0].plot(x_sorted, mean_sorted, color=colors[name], label=f"{name} mean")
            axes[0].fill_between(
                x_sorted,
                mean_sorted - 2 * std_sorted,
                mean_sorted + 2 * std_sorted,
                color=colors[name],
                alpha=0.15,
            )

    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Prediction & Uncertainty")
    axes[0].legend()

    with torch.no_grad():
        true_var_sorted = test_ds.true_var[order].squeeze().numpy()
        for name in ["Beta1", "Dynamic_Conservative"]:
            model = models[name]
            _, var = model(test_ds.x.to(device))
            var_np = np.squeeze(var.detach().cpu().numpy())
            var_sorted = np.array(var_np[order.numpy()]).reshape(-1)
            err = np.abs(var_sorted - true_var_sorted)
            axes[1].plot(x_sorted, err, color=colors[name], label=f"{name} |σ^2-σ_true^2|")

    axes[1].set_xlabel("x")
    axes[1].set_ylabel("Variance error")
    axes[1].set_title("Variance Accuracy")
    axes[1].legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved final comparison plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Run Beta-NLL paper sine benchmark.")
    parser.add_argument("--n_samples", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="outputs/paper_benchmark")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, test_ds, train_loader, test_loader = make_loaders(args.n_samples, args.batch_size, seed=args.seed)

    experiments = {
        "Beta1": {
            "beta_start": 1.0,
            "beta_end": 1.0,
            "dynamic": False,
            "strategy": "constant",
            "warmup": 0,
            "stage1_epochs": args.epochs + 1,
            "lr_stage2": 1e-3,
        },
        "Beta0": {
            "beta_start": 0.0,
            "beta_end": 0.0,
            "dynamic": False,
            "strategy": "constant",
            "warmup": 0,
            "stage1_epochs": args.epochs + 1,
            "lr_stage2": 1e-3,
        },
        "Dynamic_Conservative": {
            "beta_start": 1.0,
            "beta_end": 0.5,
            "dynamic": True,
            "strategy": "linear_decay",
            "warmup": 0,
            "stage1_epochs": 0,
            "lr_stage2": 1e-3,
        },
        "Dynamic_Reverse_Fair": {
            "beta_start": 0.5,
            "beta_end": 1.0,
            "dynamic": True,
            "strategy": "linear_decay",
            "warmup": 0,
            "stage1_epochs": 0,
            "lr_stage2": 1e-3,
        },
        "Dynamic_Delayed_LowLR": {
            "beta_start": 1.0,
            "beta_end": 0.0,
            "dynamic": True,
            "strategy": "delayed_linear",
            "warmup": 200,
            "stage1_epochs": 200,
            "lr_stage2": 1e-5,
        },
    }

    results = {}
    models = {}
    for name, cfg in experiments.items():
        print(f"Training {name}...")
        model, _ = train_model(
            beta_start=cfg["beta_start"],
            beta_end=cfg["beta_end"],
            dynamic=cfg["dynamic"],
            strategy=cfg["strategy"],
            warmup_epochs=cfg.get("warmup", 0),
            lr=1e-3,
            lr_stage2=cfg.get("lr_stage2", 1e-3),
            stage1_epochs=cfg.get("stage1_epochs", 50),
            epochs=args.epochs,
            train_loader=train_loader,
            device=device,
        )
        models[name] = model
        metrics = evaluate(model, test_loader, test_ds.true_var)
        results[name] = metrics

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    header = f"{'Experiment':<24} | {'MSE':>8} | {'NLL':>8} | {'Var_MSE':>10}"
    print(header)
    print("-" * len(header))
    for name in ["Beta1", "Beta0", "Dynamic_Conservative", "Dynamic_Reverse_Fair"]:
        m = results[name]
        print(f"{name:<24} | {m['MSE']:8.4f} | {m['NLL']:8.4f} | {m['Var_MSE']:10.4f}")

    plot_envelopes(models, test_ds, str(out_dir / "paper_sine_envelopes.png"))
    if "Beta1" in models and "Dynamic_Reverse_Fair" in models:
        plot_comparison(models, test_ds, ("Beta1", "Dynamic_Reverse_Fair"), str(out_dir / "comparison_paper.png"))
    plot_final(models, test_ds, str(out_dir / "final_paper_plot.png"))

    if "Beta1" in results and "Dynamic_Conservative" in results:
        print("\nLaTeX Table:")
        print("\\begin{tabular}{lccc}")
        print("\\toprule")
        print("Experiment & MSE & NLL & Var\\_MSE \\\\")
        print("\\midrule")
        for name in ["Beta1", "Dynamic_Conservative"]:
            m = results[name]
            print(f"{name} & {m['MSE']:.4f} & {m['NLL']:.4f} & {m['Var_MSE']:.4f} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")


if __name__ == "__main__":
    main()

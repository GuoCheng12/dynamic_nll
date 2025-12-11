import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import ToyRegressionDataset


def plot_toy(size: int, seed: int, output: str) -> None:
    torch.manual_seed(seed)
    ds = ToyRegressionDataset(size=size, normalize=False)
    x = ds.x.squeeze().numpy()
    y = ds.y.squeeze().numpy()

    true_y = (torch.sin(ds.x) * (0.5 * ds.x + 1.0)).squeeze().numpy()
    sigma = (0.1 + 0.05 * ds.x).squeeze().numpy()

    order = x.argsort()
    x_sorted = x[order]
    true_sorted = true_y[order]
    sigma_sorted = sigma[order]

    fig, (ax_scatter, ax_hist) = plt.subplots(1, 2, figsize=(12, 5))

    ax_scatter.scatter(x, y, alpha=0.3, s=10, label="samples")
    ax_scatter.plot(x_sorted, true_sorted, color="black", lw=2, label="true mean")
    ax_scatter.fill_between(
        x_sorted,
        true_sorted - 2 * sigma_sorted,
        true_sorted + 2 * sigma_sorted,
        color="gray",
        alpha=0.2,
        label="±2σ noise",
    )
    ax_scatter.set_xlabel("x ~ Gamma(k=2, θ=1.5)")
    ax_scatter.set_ylabel("y")
    ax_scatter.set_title("Toy Regression: Long Tail with Heteroscedastic Noise")
    ax_scatter.legend()

    ax_hist.hist(x, bins=60, color="steelblue", alpha=0.8)
    ax_hist.set_xlabel("x")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Input Distribution (long tail)")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved plot to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the toy long-tail regression dataset.")
    parser.add_argument("--size", type=int, default=5000, help="Number of samples to draw.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", type=str, default="outputs/toy_long_tail.png", help="Output image path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_toy(size=args.size, seed=args.seed, output=args.output)

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ToyRegressionDataset  # noqa: E402
from src.utils import get_device, set_seed  # noqa: E402
from train import instantiate_model  # noqa: E402


def load_run(run_dir: Path, ckpt_name: str, device: torch.device) -> Tuple[Dict, torch.nn.Module]:
    cfg_path = run_dir / ".hydra" / "config.yaml"
    ckpt_path = run_dir / ckpt_name
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing Hydra config at {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")

    cfg = OmegaConf.load(cfg_path)
    set_seed(cfg.seed)
    model = instantiate_model(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return cfg, model


def gather_data(cfg) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, ToyRegressionDataset]:
    set_seed(cfg.seed)
    dataset = ToyRegressionDataset(normalize=True)
    n = len(dataset)
    train_len = int(0.8 * n)
    val_len = int(0.1 * n)
    test_len = n - train_len - val_len
    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    train_indices = train_ds.indices
    test_indices = test_ds.indices

    train_x = dataset.x[train_indices].squeeze()
    test_x = dataset.x[test_indices].squeeze()
    test_y_norm = dataset.y[test_indices].squeeze()
    test_y = dataset.unnormalize(test_y_norm)

    return train_x, test_x, test_y, dataset.y_std, dataset


def predict(
    model: torch.nn.Module,
    x: torch.Tensor,
    y_std: torch.Tensor,
    dataset: ToyRegressionDataset,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        mean_norm, var_norm = model(x.unsqueeze(-1).to(device))
    mean_norm = mean_norm.squeeze()
    var_norm = var_norm.squeeze()
    mean = dataset.unnormalize(mean_norm)
    var = var_norm * (y_std ** 2)
    return mean.cpu(), var.cpu()


def plot_comparison(train_x, test_x, test_y, preds: Dict[str, Dict[str, torch.Tensor]], output: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(train_x.numpy(), bins=60, color="steelblue", alpha=0.8)
    axes[0].axvspan(0, 8, color="green", alpha=0.1, label="Head (x <= 8)")
    axes[0].axvspan(8, train_x.max().item(), color="red", alpha=0.1, label="Tail (x > 8)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("count")
    axes[0].set_title("Training Input Distribution (Long Tail)")
    axes[0].legend()

    axes[1].scatter(test_x.numpy(), test_y.numpy(), color="gray", alpha=0.3, s=10, label="Ground Truth (test)")
    colors = {"0.0": "red", "0.5": "blue", "1.0": "green"}
    for beta, outputs in preds.items():
        x_sorted, order = torch.sort(test_x)
        mean_sorted = outputs["mean"][order]
        var_sorted = outputs["var"][order]
        std_sorted = torch.sqrt(var_sorted.clamp(min=1e-8))
        axes[1].plot(x_sorted.numpy(), mean_sorted.numpy(), color=colors.get(beta, "black"), label=f"beta={beta}")
        axes[1].fill_between(
            x_sorted.numpy(),
            (mean_sorted - 2 * std_sorted).numpy(),
            (mean_sorted + 2 * std_sorted).numpy(),
            color=colors.get(beta, "black"),
            alpha=0.15,
        )
    axes[1].axvline(8.0, color="black", ls="--", lw=1, alpha=0.6, label="Tail threshold (x=8)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Model Predictions with Uncertainty")
    axes[1].legend()

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    print(f"Saved comparison plot to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare beta runs on the toy long-tail dataset.")
    parser.add_argument("--beta0_dir", type=str, default="outputs/beta_0.0", help="Run directory for beta=0.0")
    parser.add_argument("--beta05_dir", type=str, default="outputs/beta_0.5", help="Run directory for beta=0.5")
    parser.add_argument("--beta1_dir", type=str, default="outputs/beta_1.0", help="Run directory for beta=1.0")
    parser.add_argument("--ckpt", type=str, default="checkpoint.pt", help="Checkpoint filename inside each run dir")
    parser.add_argument("--output", type=str, default="outputs/beta_comparison.png", help="Output plot path")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto|cpu|cuda).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = {
        "0.0": Path(args.beta0_dir),
        "0.5": Path(args.beta05_dir),
        "1.0": Path(args.beta1_dir),
    }

    device = get_device(args.device)
    cfg, _ = load_run(run_dirs["0.0"], args.ckpt, device)
    train_x, test_x, test_y, y_std, dataset = gather_data(cfg)

    preds: Dict[str, Dict[str, torch.Tensor]] = {}
    for beta, run_dir in run_dirs.items():
        cfg_beta, model = load_run(run_dir, args.ckpt, device)
        set_seed(cfg_beta.seed)
        mean, var = predict(model, test_x, y_std, dataset, device)
        preds[beta] = {"mean": mean, "var": var}

    plot_comparison(train_x, test_x, test_y, preds, args.output)


if __name__ == "__main__":
    main()

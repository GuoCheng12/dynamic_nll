import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ToyRegressionDataset, build_dataloaders  # noqa: E402
from src.utils import get_device, set_seed  # noqa: E402
from train import instantiate_model  # noqa: E402


def load_model(run_dir: Path, device: torch.device, ckpt_name: str = "checkpoint.pt"):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    set_seed(cfg.seed)
    model = instantiate_model(cfg).to(device)
    state = torch.load(run_dir / ckpt_name, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return cfg, model


def tail_confidence_error(model: torch.nn.Module, cfg, device: torch.device, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    _, _, test_loader = build_dataloaders(cfg.hyperparameters.batch_size, seed=cfg.seed)
    dataset = test_loader.dataset.dataset
    test_indices = torch.tensor(test_loader.dataset.indices)
    x = dataset.x[test_indices].squeeze()
    y_norm = dataset.y[test_indices].squeeze()
    mask_tail = x > 3.0
    x_tail = x[mask_tail]
    y_norm_tail = y_norm[mask_tail]
    if len(x_tail) == 0:
        return np.array([]), np.array([])

    with torch.no_grad():
        mean_norm, var_norm = model(x_tail.unsqueeze(-1).to(device))
    mean_norm = mean_norm.squeeze()
    var_norm = var_norm.squeeze()

    mean = dataset.unnormalize(mean_norm.cpu())
    target = dataset.unnormalize(y_norm_tail)
    var = var_norm.cpu() * (dataset.y_std ** 2)

    confidence = 1.0 / (var + 1e-8)
    conf_np = confidence.cpu().numpy()
    error_np = torch.abs(mean - target).cpu().numpy()

    bin_edges = np.quantile(conf_np, np.linspace(0, 1, n_bins + 1))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_errors = []
    for i in range(n_bins):
        mask = (conf_np >= bin_edges[i]) & (conf_np < bin_edges[i + 1])
        bin_errors.append(error_np[mask].mean() if mask.any() else np.nan)
    return bin_centers, np.array(bin_errors)


def plot_calibration(beta1_dir: Path, dynamic_dir: Path, output: str, device: torch.device, n_bins: int = 10) -> None:
    cfg1, model1 = load_model(beta1_dir, device)
    cfgd, modeld = load_model(dynamic_dir, device)

    x1, err1 = tail_confidence_error(model1, cfg1, device, n_bins=n_bins)
    xd, errd = tail_confidence_error(modeld, cfgd, device, n_bins=n_bins)

    plt.figure(figsize=(8, 5))
    plt.plot(x1, err1, "-o", color="red", label="Beta 1.0")
    plt.plot(xd, errd, "-o", color="blue", label="Dynamic_Fix")
    plt.xlabel("Predicted Confidence (1 / variance)")
    plt.ylabel("Average MAE in bin (Tail x>3)")
    plt.title("Confidence vs Error (Tail)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    print(f"Saved calibration plot to {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot calibration/reliability for tail region.")
    parser.add_argument("--beta1_dir", type=str, required=True, help="Run dir for Beta 1.0 model")
    parser.add_argument("--dynamic_dir", type=str, required=True, help="Run dir for Dynamic_Fix model")
    parser.add_argument("--output", type=str, default="outputs/calibration_tail.png", help="Output plot path")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto|cpu|cuda).")
    args = parser.parse_args()

    device = get_device(args.device)
    plot_calibration(Path(args.beta1_dir), Path(args.dynamic_dir), args.output, device, n_bins=args.bins)


if __name__ == "__main__":
    main()

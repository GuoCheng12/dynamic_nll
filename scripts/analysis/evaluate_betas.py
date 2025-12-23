import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import ToyRegressionDataset, build_dataloaders  # noqa: E402
from src.utils import evaluate_head_tail, get_device, mae, nll, rmse, set_seed  # noqa: E402
from train import instantiate_model  # noqa: E402


def load_model(run_dir: Path, ckpt_name: str, device: torch.device):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    set_seed(cfg.seed)
    model = instantiate_model(cfg).to(device)
    state = torch.load(run_dir / ckpt_name, map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return cfg, model


def evaluate(run_dir: Path, ckpt_name: str, device: torch.device) -> Dict[str, float]:
    cfg, model = load_model(run_dir, ckpt_name, device)
    _, _, test_loader = build_dataloaders(cfg.hyperparameters.batch_size, seed=cfg.seed)
    dataset = test_loader.dataset.dataset
    test_indices = torch.tensor(test_loader.dataset.indices)
    base_metrics = {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "nll": 0.0}
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            mean_norm, var_norm = model(data)
            mean_norm = mean_norm.squeeze()
            var_norm = var_norm.squeeze()
            target_norm = target.squeeze()

            mean = dataset.unnormalize(mean_norm.cpu())
            target_true = dataset.unnormalize(target_norm.cpu())
            var = var_norm.cpu() * (dataset.y_std ** 2)

            batch = len(data)
            base_metrics["mse"] += torch.mean((mean - target_true) ** 2).item() * batch
            base_metrics["mae"] += mae(mean, target_true) * batch
            base_metrics["rmse"] += rmse(mean, target_true) * batch
            base_metrics["nll"] += nll(mean, target_true, var) * batch
            count += batch

    for k in base_metrics:
        base_metrics[k] /= max(count, 1)

    split_metrics = evaluate_head_tail(model, dataset, indices=test_indices, device=device)
    return {
        "mse": base_metrics["mse"],
        "mae": base_metrics["mae"],
        "rmse": base_metrics["rmse"],
        "nll": base_metrics["nll"],
        **split_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baseline beta runs on test set.")
    parser.add_argument("--beta0_dir", type=str, default="outputs/beta_0.0")
    parser.add_argument("--beta05_dir", type=str, default="outputs/beta_0.5")
    parser.add_argument("--beta1_dir", type=str, default="outputs/beta_1.0")
    parser.add_argument("--ckpt", type=str, default="checkpoint.pt")
    parser.add_argument("--device", type=str, default="auto", help="Device to run on (auto|cpu|cuda).")
    args = parser.parse_args()

    runs = {"0.0": Path(args.beta0_dir), "0.5": Path(args.beta05_dir), "1.0": Path(args.beta1_dir)}
    results: Dict[str, Dict[str, float]] = {}
    device = get_device(args.device)
    for beta, run_dir in runs.items():
        metrics = evaluate(run_dir, args.ckpt, device)
        results[beta] = metrics

    header = f"{'beta':>6} | {'MSE':>10} | {'MAE':>10} | {'RMSE':>10} | {'NLL':>10}"
    print(header)
    print("-" * len(header))
    for beta, m in sorted(results.items()):
        print(f"{beta:>6} | {m['mse']:10.4f} | {m['mae']:10.4f} | {m['rmse']:10.4f} | {m['nll']:10.4f}")


if __name__ == "__main__":
    main()

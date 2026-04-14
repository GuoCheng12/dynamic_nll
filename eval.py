from __future__ import annotations

import hydra
import torch
from omegaconf import DictConfig

from src.data import build_dataloaders, build_paper_sine_dataloaders
from src.utils import expected_calibration_error, get_device, mae, nll, rmse, set_seed
from train import instantiate_model


def build_test_loader(cfg: DictConfig):
    batch_size = cfg.hyperparameters.get("batch_size", 128)
    eval_batch_size = cfg.hyperparameters.get("eval_batch_size", batch_size)
    if cfg.dataset.name == "paper_sine":
        _, _, test_loader = build_paper_sine_dataloaders(
            cfg.dataset,
            batch_size=batch_size,
            seed=cfg.seed,
            eval_batch_size=eval_batch_size,
        )
        return test_loader
    if cfg.dataset.name == "toy_regression":
        _, _, test_loader = build_dataloaders(
            batch_size=batch_size,
            splits=(
                float(cfg.dataset.split.train),
                float(cfg.dataset.split.val),
                float(cfg.dataset.split.test),
            ),
            size=cfg.dataset.get("size", 5000),
            normalize=cfg.dataset.get("normalize", True),
            seed=cfg.seed,
            eval_batch_size=eval_batch_size,
        )
        return test_loader
    raise ValueError(f"Dataset {cfg.dataset.name} is not supported by eval.py.")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.get("device", "auto"))
    test_loader = build_test_loader(cfg)

    model = instantiate_model(cfg).to(device)
    if cfg.get("checkpoint"):
        state = torch.load(cfg.checkpoint, map_location=device)
        model.load_state_dict(state["model"])

    model.eval()
    metrics = {"mse": 0.0, "mae": 0.0, "rmse": 0.0, "nll": 0.0, "ece": 0.0}
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            mean, variance = model(data)
            batch_size = len(data)
            metrics["mse"] += torch.mean((mean - target) ** 2).item() * batch_size
            metrics["mae"] += mae(mean, target) * batch_size
            metrics["rmse"] += rmse(mean, target) * batch_size
            metrics["nll"] += nll(mean, target, variance) * batch_size
            metrics["ece"] += expected_calibration_error(mean, variance, target) * batch_size
            count += batch_size

    for key in metrics:
        metrics[key] /= max(count, 1)
    print("Test metrics:", metrics)


if __name__ == "__main__":
    main()

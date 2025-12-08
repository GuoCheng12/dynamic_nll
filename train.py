import math
from typing import Any, Dict

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from src.dataset import build_dataloaders
from src.loss import BetaScheduler, GaussianLogLikelihoodLoss
from src.models import MLPRegressor
from src.utils import grad_norms, mae, nll, rmse, set_seed

try:
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None


def instantiate_model(cfg: DictConfig) -> torch.nn.Module:
    if cfg.model.name == "mlp":
        return MLPRegressor(
            input_dim=10,
            hidden_sizes=cfg.model.hidden_sizes,
            activation=cfg.model.activation,
            dropout=cfg.model.get("dropout", 0.0),
        )
    raise ValueError(f"Unknown model: {cfg.model.name}")


def compute_metrics(mean: torch.Tensor, var: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    return {
        "mse": torch.mean((mean - target) ** 2).item(),
        "mae": mae(mean, target),
        "rmse": rmse(mean, target),
        "nll": nll(mean, target, var),
    }


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate_model(cfg).to(device)
    train_loader, val_loader, _ = build_dataloaders(cfg.hyperparameters.batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    criterion = GaussianLogLikelihoodLoss()
    scheduler = BetaScheduler(
        strategy=cfg.uncertainty.beta_strategy,
        start_beta=cfg.uncertainty.beta_start,
        end_beta=cfg.uncertainty.beta_end,
        total_steps=cfg.uncertainty.total_steps,
    )

    if cfg.logging.use_wandb and wandb is not None:
        wandb.init(project=cfg.logging.project_name, config=OmegaConf.to_container(cfg, resolve=True))

    global_step = 0
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        for batch in train_loader:
            data, target = batch
            data, target = data.to(device), target.to(device)

            new_beta = scheduler.get_beta(global_step)
            criterion.beta = new_beta

            mean, variance = model(data)
            loss = criterion(mean, target, variance=variance, interpolate=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = compute_metrics(mean.detach(), variance.detach(), target)
            norms = grad_norms(model)
            log_payload: Dict[str, Any] = {
                "train/loss": loss.item(),
                "train/beta_value": new_beta,
                **{f"train/{k}": v for k, v in metrics.items()},
                **{f"train/{k}": v for k, v in norms.items()},
            }
            if cfg.logging.use_wandb and wandb is not None:
                wandb.log(log_payload, step=global_step)

            global_step += 1

        # Validation hook for plateau strategy or research logging
        model.eval()
        val_loss_accum, count = 0.0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                mean, variance = model(data)
                val_loss = criterion(mean, target, variance=variance, interpolate=False)
                val_loss_accum += val_loss.item() * len(data)
                count += len(data)
        val_loss_avg = val_loss_accum / max(count, 1)
        if cfg.uncertainty.beta_strategy == "plateau":
            scheduler.get_beta(global_step, current_loss=val_loss_avg)
        if cfg.logging.use_wandb and wandb is not None:
            wandb.log({"val/loss": val_loss_avg}, step=global_step)

    if cfg.logging.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

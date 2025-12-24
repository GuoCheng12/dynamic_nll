import math
import os
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import build_dataloaders
from src.modules import BetaScheduler, GaussianLogLikelihoodLoss
from src.models import DepthUNet, MLPRegressor
from src.utils import get_device, grad_norms, mae, nll, rmse, set_seed
from hydra.core.hydra_config import HydraConfig

try:
    import wandb
except ImportError:  # pragma: no cover - optional
    wandb = None


def instantiate_model(cfg: DictConfig) -> torch.nn.Module:
    if cfg.model.name == "mlp":
        return MLPRegressor(
            input_dim=cfg.model.get("input_dim", 1),
            hidden_sizes=cfg.model.hidden_sizes,
            activation=cfg.model.activation,
            dropout=cfg.model.get("dropout", 0.0),
        )
    if cfg.model.name == "depth_unet":
        return DepthUNet(
            encoder=cfg.model.encoder,
            pretrained=cfg.model.pretrained,
            min_depth=cfg.model.get("min_depth", 1e-3),
            min_var=cfg.model.get("min_var", 1e-6),
            max_val=cfg.model.get("max_val", 10.0),
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
    device = get_device(cfg.get("device", "auto"))
    model = instantiate_model(cfg).to(device)
    train_loader, val_loader, _ = build_dataloaders(cfg.hyperparameters.batch_size, seed=cfg.seed)
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
    stage1_epochs = cfg.hyperparameters.get("stage1_epochs", 50)
    stage2_lr = cfg.hyperparameters.get("lr_stage2", cfg.hyperparameters.lr)
    total_epochs = cfg.hyperparameters.epochs
    stage2_span = max(total_epochs - stage1_epochs, 1)
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        # Two-phase schedule: phase 1 constant beta/lr, phase 2 beta decay + lr drop
        if epoch == stage1_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = stage2_lr
        if cfg.uncertainty.beta_strategy == "linear_decay":
            if epoch < stage1_epochs:
                epoch_beta = cfg.uncertainty.beta_start
            else:
                progress = min((epoch - stage1_epochs) / stage2_span, 1.0)
                epoch_beta = cfg.uncertainty.beta_start + progress * (
                    cfg.uncertainty.beta_end - cfg.uncertainty.beta_start
                )
        else:
            epoch_beta = scheduler.get_beta(epoch)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}", leave=False):
            data, target = batch
            data, target = data.to(device), target.to(device)

            criterion.beta = epoch_beta

            mean, variance = model(data)
            loss = criterion(mean, target, variance=variance, interpolate=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = compute_metrics(mean.detach(), variance.detach(), target)
            norms = grad_norms(model)
            log_payload: Dict[str, Any] = {
                "train/loss": loss.item(),
                "train/beta_value": epoch_beta,
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

    run_dir = HydraConfig.get().runtime.output_dir
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    torch.save({"model": model.state_dict()}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()

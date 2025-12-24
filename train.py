import math
import os
from typing import Any, Dict

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.data import build_dataloaders, build_nyu_dataloaders
from src.depth_metrics import compute_depth_metrics
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
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)

    set_seed(cfg.seed + rank)
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else get_device(cfg.get("device", "auto"))
    model = instantiate_model(cfg).to(device)
    if distributed and torch.cuda.is_available():
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local_rank])

    if cfg.dataset.name == "nyu_depth":
        train_loader, val_loader, train_sampler, val_sampler = build_nyu_dataloaders(
            cfg.dataset, distributed=distributed, rank=rank, world_size=world_size
        )
    else:
        train_loader, val_loader, _ = build_dataloaders(
            cfg.hyperparameters.batch_size,
            seed=cfg.seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
        )
        train_sampler = getattr(train_loader, "sampler", None)
        val_sampler = getattr(val_loader, "sampler", None)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hyperparameters.lr)

    criterion = GaussianLogLikelihoodLoss()
    scheduler = BetaScheduler(
        strategy=cfg.uncertainty.beta_strategy,
        start_beta=cfg.uncertainty.beta_start,
        end_beta=cfg.uncertainty.beta_end,
        total_steps=cfg.uncertainty.total_steps,
    )

    if rank == 0 and cfg.logging.use_wandb and wandb is not None:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    global_step = 0
    stage1_epochs = cfg.hyperparameters.get("stage1_epochs", 50)
    stage2_lr = cfg.hyperparameters.get("lr_stage2", cfg.hyperparameters.lr)
    total_epochs = cfg.hyperparameters.epochs
    stage2_span = max(total_epochs - stage1_epochs, 1)
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        if distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
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
        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}", leave=False) if rank == 0 else train_loader
        for batch in iterator:
            data, target = batch
            data, target = data.to(device), target.to(device)

            criterion.beta = epoch_beta

            mean, variance = model(data)
            loss = criterion(mean, target, variance=variance, interpolate=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                if cfg.dataset.name == "nyu_depth":
                    depth_metrics = compute_depth_metrics(mean.detach(), target.detach())
                    log_payload: Dict[str, Any] = {
                        "train/loss": loss.item(),
                        "train/beta": epoch_beta,
                        "train/rmse": depth_metrics["rmse"],
                        "train/abs_rel": depth_metrics["abs_rel"],
                        "train/delta1": depth_metrics["delta1"],
                        "train/nll": loss.item(),
                    }
                else:
                    metrics = compute_metrics(mean.detach(), variance.detach(), target)
                    norms = grad_norms(model)
                    log_payload = {
                        "train/loss": loss.item(),
                        "train/beta": epoch_beta,
                        **{f"train/{k}": v for k, v in metrics.items()},
                        **{f"train/{k}": v for k, v in norms.items()},
                    }
                if cfg.logging.use_wandb and wandb is not None:
                    wandb.log(log_payload, step=global_step)

            global_step += 1

        if distributed:
            dist.barrier()

        # Validation hook for plateau strategy or research logging
        model.eval()
        val_loss_accum, count = 0.0, 0
        depth_sums = {"rmse": 0.0, "abs_rel": 0.0, "delta1": 0.0}
        with torch.no_grad():
            val_iter = tqdm(val_loader, desc="Val", leave=False) if rank == 0 else val_loader
            for data, target in val_iter:
                data, target = data.to(device), target.to(device)
                mean, variance = model(data)
                val_loss = criterion(mean, target, variance=variance, interpolate=False)
                batch_size = len(data)
                val_loss_accum += val_loss.item() * batch_size
                count += batch_size
                if cfg.dataset.name == "nyu_depth":
                    batch_metrics = compute_depth_metrics(mean, target)
                    for k in depth_sums:
                        depth_sums[k] += batch_metrics[k] * batch_size
        if distributed:
            tensor = torch.tensor([val_loss_accum, count], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            val_loss_accum, count = tensor.tolist()
            if cfg.dataset.name == "nyu_depth":
                depth_tensor = torch.tensor(
                    [depth_sums["rmse"], depth_sums["abs_rel"], depth_sums["delta1"], count],
                    device=device,
                )
                dist.all_reduce(depth_tensor, op=dist.ReduceOp.SUM)
                depth_sums["rmse"], depth_sums["abs_rel"], depth_sums["delta1"], _ = depth_tensor.tolist()
                count = depth_tensor.tolist()[3]
        val_loss_avg = val_loss_accum / max(count, 1)
        if cfg.uncertainty.beta_strategy == "plateau":
            scheduler.get_beta(global_step, current_loss=val_loss_avg)
        if rank == 0:
            if cfg.dataset.name == "nyu_depth":
                val_log = {
                    "val/rmse": depth_sums["rmse"] / max(count, 1),
                    "val/abs_rel": depth_sums["abs_rel"] / max(count, 1),
                    "val/delta1": depth_sums["delta1"] / max(count, 1),
                }
            else:
                val_log = {"val/loss": val_loss_avg}
            if cfg.logging.use_wandb and wandb is not None:
                wandb.log(val_log, step=global_step)

    if rank == 0 and cfg.logging.use_wandb and wandb is not None:
        wandb.finish()

    if rank == 0:
        run_dir = HydraConfig.get().runtime.output_dir
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save({"model": state}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

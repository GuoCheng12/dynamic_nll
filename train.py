import math
import os
from typing import Any, Dict, Optional

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.data import build_dataloaders, build_nyu_dataloaders
from src.depth_metrics import compute_error_accumulators, compute_depth_metrics
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

    eval_distributed = False if distributed else None
    if cfg.dataset.name == "nyu_depth":
        train_loader, val_loader, train_sampler, val_sampler = build_nyu_dataloaders(
            cfg.dataset, distributed=distributed, rank=rank, world_size=world_size, eval_distributed=eval_distributed
        )
    else:
        train_loader, val_loader, _ = build_dataloaders(
            cfg.hyperparameters.batch_size,
            seed=cfg.seed,
            distributed=distributed,
            rank=rank,
            world_size=world_size,
            eval_distributed=eval_distributed,
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
        if cfg.dataset.name == "nyu_depth":
            train_accum = torch.zeros(8, device=device)
        else:
            train_accum = torch.zeros(5, device=device)
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
            if cfg.dataset.name == "nyu_depth":
                mask = target > cfg.dataset.get("min_depth", 1e-3)
                loss = criterion(mean, target, variance=variance, interpolate=False, mask=mask)
            else:
                loss = criterion(mean, target, variance=variance, interpolate=False)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if cfg.dataset.name == "nyu_depth":
                mean_metrics = mean.detach()
                if mean_metrics.shape[-2:] != target.shape[-2:]:
                    mean_metrics = torch.nn.functional.interpolate(
                        mean_metrics, size=target.shape[-2:], mode="bilinear", align_corners=True
                    )
                batch_accum = compute_error_accumulators(
                    mean_metrics,
                    target.detach(),
                    min_depth=cfg.dataset.get("min_depth", 1e-3),
                    max_depth=cfg.dataset.get("max_depth", 10.0),
                    use_eigen_crop=False,
                )
                valid_pixels = (target > cfg.dataset.get("min_depth", 1e-3)).sum().to(dtype=loss.dtype)
                train_accum[:7] += batch_accum
                train_accum[7] += loss.detach() * valid_pixels
            else:
                err = mean.detach() - target
                sse = torch.sum(err**2)
                abs_sum = torch.sum(torch.abs(err))
                nll_sum = torch.sum(-0.5 * ((err**2) / variance.detach() + torch.log(variance.detach())))
                count = err.new_tensor(float(err.numel()))
                train_accum += torch.stack([sse, abs_sum, nll_sum, count, loss.detach() * count])

            if rank == 0 and cfg.logging.use_wandb and wandb is not None:
                wandb.log({"train/loss": loss.item(), "train/beta": epoch_beta}, step=global_step)

            global_step += 1

        if distributed:
            dist.all_reduce(train_accum, op=dist.ReduceOp.SUM)
        if rank == 0:
            if cfg.dataset.name == "nyu_depth":
                vals = train_accum.detach().cpu().tolist()
                total_pixels = max(vals[6], 1.0)
                train_log = {
                    "train/rmse": math.sqrt(vals[0] / total_pixels),
                    "train/abs_rel": vals[1] / total_pixels,
                    "train/log10": vals[2] / total_pixels,
                    "train/delta1": vals[3] / total_pixels,
                    "train/delta2": vals[4] / total_pixels,
                    "train/delta3": vals[5] / total_pixels,
                    "train/nll": vals[7] / total_pixels,
                }
            else:
                vals = train_accum.detach().cpu().tolist()
                total = max(vals[3], 1.0)
                train_log = {
                    "train/mse": vals[0] / total,
                    "train/mae": vals[1] / total,
                    "train/rmse": math.sqrt(vals[0] / total),
                    "train/nll": vals[2] / total,
                    "train/loss": vals[4] / total,
                }
            if cfg.logging.use_wandb and wandb is not None:
                wandb.log(train_log, step=global_step)

        if distributed:
            dist.barrier()

        # Validation hook for plateau strategy or research logging
        model.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        val_loss_count = torch.tensor(0.0, device=device)
        depth_accum = torch.zeros(7, device=device)
        if (not distributed) or rank == 0:
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc="Val", leave=False)
                for data, target in val_iter:
                    data, target = data.to(device), target.to(device)
                    mean, variance = model(data)
                    if cfg.dataset.name == "nyu_depth":
                        mask = target > cfg.dataset.get("min_depth_eval", cfg.dataset.get("min_depth", 1e-3))
                        val_loss = criterion(mean, target, variance=variance, interpolate=False, mask=mask)
                        valid_pixels = mask.sum().to(dtype=val_loss_sum.dtype)
                        val_loss_sum += val_loss.detach() * valid_pixels
                        val_loss_count += valid_pixels
                    else:
                        val_loss = criterion(mean, target, variance=variance, interpolate=False)
                        batch_count = target.numel()
                        val_loss_sum += val_loss.detach() * batch_count
                        val_loss_count += batch_count
                    if cfg.dataset.name == "nyu_depth":
                        mean_metrics = mean
                        target_metrics = target
                        if mean_metrics.shape[-2:] != target_metrics.shape[-2:]:
                            mean_metrics = F.interpolate(
                                mean_metrics, size=target_metrics.shape[-2:], mode="bilinear", align_corners=True
                            )
                        if target_metrics.max().item() > 80.0:
                            target_metrics = target_metrics / 1000.0
                        batch_accum = compute_error_accumulators(
                            mean_metrics,
                            target_metrics,
                            min_depth=cfg.dataset.get("min_depth_eval", cfg.dataset.get("min_depth", 1e-3)),
                            max_depth=cfg.dataset.get("max_depth_eval", cfg.dataset.get("max_depth", 10.0)),
                            use_eigen_crop=True,
                        )
                        depth_accum += batch_accum
        if distributed:
            val_loss_tensor = torch.stack([val_loss_sum, val_loss_count])
            dist.broadcast(val_loss_tensor, src=0)
            val_loss_sum, val_loss_count = val_loss_tensor
        val_loss_avg = (val_loss_sum / torch.clamp(val_loss_count, min=1.0)).item()
        if cfg.uncertainty.beta_strategy == "plateau":
            scheduler.get_beta(global_step, current_loss=val_loss_avg)
        if rank == 0:
            if cfg.dataset.name == "nyu_depth":
                vals = depth_accum.detach().cpu().tolist()
                total_pixels = max(vals[6], 1.0)
                rmse = math.sqrt(vals[0] / total_pixels)
                abs_rel = vals[1] / total_pixels
                log10 = vals[2] / total_pixels
                delta1 = vals[3] / total_pixels
                delta2 = vals[4] / total_pixels
                delta3 = vals[5] / total_pixels
                val_log = {
                    "val/rmse": rmse,
                    "val/abs_rel": abs_rel,
                    "val/log10": log10,
                    "val/delta1": delta1,
                    "val/delta2": delta2,
                    "val/delta3": delta3,
                }
            else:
                val_log = {"val/loss": val_loss_avg}
            if cfg.logging.use_wandb and wandb is not None:
                wandb.log(val_log, step=global_step)

        if distributed:
            dist.barrier()

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

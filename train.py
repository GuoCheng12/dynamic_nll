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
from src.depth_metrics import compute_metrics_per_image
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
    # --- 1. Environment Setup ---
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

    eval_distributed = False 
    
    if cfg.dataset.name == "nyu_depth":
        train_loader, val_loader, train_sampler, val_sampler = build_nyu_dataloaders(
            cfg.dataset, distributed=distributed, rank=rank, world_size=world_size, 
            eval_distributed=eval_distributed 
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

    criterion = GaussianLogLikelihoodLoss()
    scheduler = BetaScheduler(
        strategy=cfg.uncertainty.beta_strategy,
        start_beta=cfg.uncertainty.beta_start,
        end_beta=cfg.uncertainty.beta_end,
        total_steps=cfg.uncertainty.total_steps,
    )

    ddp_model = model.module if isinstance(model, DDP) else model
    if hasattr(ddp_model, "get_1x_lr_params") and hasattr(ddp_model, "get_10x_lr_params"):
        params = [
            {"params": ddp_model.get_1x_lr_params(), "lr": cfg.hyperparameters.lr / 10},
            {"params": ddp_model.get_10x_lr_params(), "lr": cfg.hyperparameters.lr},
        ]
        max_lrs = [group["lr"] for group in params]
        if rank == 0:
            print("Using differential learning rates: 1x backbone, 10x decoder/head")
    else:
        if rank == 0:
            print("WARNING: get_1x_lr_params missing, using uniform LR.")
        params = model.parameters()
        max_lrs = cfg.hyperparameters.lr

    optimizer = torch.optim.AdamW(params, lr=cfg.hyperparameters.lr, weight_decay=0.1)
    steps_per_epoch = len(train_loader)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=cfg.hyperparameters.epochs,
        steps_per_epoch=steps_per_epoch,
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25,
        final_div_factor=100,
    )
    if rank == 0 and cfg.logging.use_wandb and wandb is not None:
        wandb.init(
            project=cfg.logging.project_name,
            name=cfg.logging.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # --- 6. Training Loop ---
    global_step = 0

    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        if distributed and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
            
        # Initialize Train Accumulators
        # 9 slots for NYU: [rmse_sum, abs_rel_sum, log10_sum, d1_sum, d2_sum, d3_sum, image_count, nll_sum, nll_count]
        # 5 slots for Toy: [sse, abs, nll_sum, count, loss_sum]
        if cfg.dataset.name == "nyu_depth":
            train_accum = torch.zeros(9, device=device)
        else:
            train_accum = torch.zeros(5, device=device)

        epoch_beta = scheduler.get_beta(epoch)

        iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}", leave=False) if rank == 0 else train_loader
        
        for batch in iterator:
            data, target = batch
            data, target = data.to(device), target.to(device)

            criterion.beta = epoch_beta

            mean, variance = model(data)
            
            interpolate = cfg.dataset.name == "nyu_depth" or target.dim() == 4
            if cfg.dataset.name == "nyu_depth":
                mask = target > cfg.dataset.get("min_depth", 1e-3)
                loss = criterion(mean, target, variance=variance, interpolate=interpolate, mask=mask)
            else:
                loss = criterion(mean, target, variance=variance, interpolate=interpolate)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # --- Training Metrics Aggregation ---
            with torch.no_grad():
                if cfg.dataset.name == "nyu_depth":
                    mean_metrics = mean.detach()
                    if mean_metrics.shape[-2:] != target.shape[-2:]:
                        mean_metrics = F.interpolate(
                            mean_metrics, size=target.shape[-2:], mode="bilinear", align_corners=True
                        )
                    batch_metrics = compute_metrics_per_image(
                        mean_metrics,
                        target.detach(),
                        min_depth=cfg.dataset.get("min_depth", 1e-3),
                        max_depth=cfg.dataset.get("max_depth", 10.0),
                        use_eigen_crop=False,
                    )
                    batch_count = mean_metrics.shape[0]
                    train_accum[0] += batch_metrics["rmse"] * batch_count
                    train_accum[1] += batch_metrics["abs_rel"] * batch_count
                    train_accum[2] += batch_metrics["log10"] * batch_count
                    train_accum[3] += batch_metrics["delta1"] * batch_count
                    train_accum[4] += batch_metrics["delta2"] * batch_count
                    train_accum[5] += batch_metrics["delta3"] * batch_count
                    train_accum[6] += batch_count

                    valid_pixels = (target > cfg.dataset.get("min_depth", 1e-3)).sum().float()
                    train_accum[7] += loss.detach() * valid_pixels
                    train_accum[8] += valid_pixels
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

        # --- End of Epoch: Sync Training Metrics ---
        if distributed:
            dist.all_reduce(train_accum, op=dist.ReduceOp.SUM)
            
        if rank == 0:
            vals = train_accum.detach().cpu().tolist()
            if cfg.dataset.name == "nyu_depth":
                total_images = max(vals[6], 1.0)
                nll_count = max(vals[8], 1.0)
                train_log = {
                    "train/rmse": vals[0] / total_images,
                    "train/abs_rel": vals[1] / total_images,
                    "train/log10": vals[2] / total_images,
                    "train/delta1": vals[3] / total_images,
                    "train/delta2": vals[4] / total_images,
                    "train/delta3": vals[5] / total_images,
                    "train/nll": vals[7] / nll_count,
                }
            else:
                total = max(vals[3], 1.0)
                train_log = {
                    "train/rmse": math.sqrt(vals[0] / total),
                    "train/nll": vals[2] / total,
                }
            if cfg.logging.use_wandb and wandb is not None:
                wandb.log(train_log, step=global_step)

        # =========================================================
        #   CRITICAL FIX: VALIDATION LOOP (Avoiding Deadlock)
        # =========================================================
        
        if distributed:
            dist.barrier() # 1. Sync all processes before validation starts

        model.eval()
        
        # Initialize accumulators on ALL ranks (even if they don't run loop)
        val_loss_sum = torch.tensor(0.0, device=device)
        val_loss_count = torch.tensor(0.0, device=device)
        # [rmse_sum, abs_rel_sum, log10_sum, d1_sum, d2_sum, d3_sum, image_count]
        depth_accum = torch.zeros(7, device=device)

        # Only Rank 0 runs the loop to allow TQDM and simple metrics logic
        # (Other ranks wait at the barrier below)
        if (not distributed) or rank == 0:
            with torch.no_grad():
                val_iter = tqdm(val_loader, desc="Val", leave=False)
                for data, target in val_iter:
                    data, target = data.to(device), target.to(device)

                    # --- TTA: normal + horizontal flip (rank 0 only) ---
                    if cfg.dataset.name == "nyu_depth":
                        if distributed:
                            pred_raw, variance = model.module(data)
                            pred_flip_raw, _ = model.module(torch.flip(data, [3]))
                        else:
                            pred_raw, variance = model(data)
                            pred_flip_raw, _ = model(torch.flip(data, [3]))
                        pred_flip = torch.flip(pred_flip_raw, [3])
                        mean = 0.5 * (pred_raw + pred_flip)
                    else:
                        if distributed:
                            mean, variance = model.module(data)
                        else:
                            mean, variance = model(data)

                    # Metric Calculation
                    interpolate = cfg.dataset.name == "nyu_depth" or target.dim() == 4
                    mean_for_loss = mean
                    if cfg.dataset.name == "nyu_depth":
                        mean_for_loss = pred_raw
                    if cfg.dataset.name == "nyu_depth":
                        min_d = cfg.dataset.get("min_depth_eval", cfg.dataset.get("min_depth", 1e-3))
                        mask = target > min_d
                        
                        val_loss = criterion(mean_for_loss, target, variance=variance, interpolate=interpolate, mask=mask)
                        
                        valid_pixels = mask.sum().float()
                        val_loss_sum += val_loss.detach() * valid_pixels
                        val_loss_count += valid_pixels

                        # Unit Fix & Resolution Fix
                        if mean.shape[-2:] != target.shape[-2:]:
                            mean = F.interpolate(mean, size=target.shape[-2:], mode="bilinear", align_corners=True)
                        
                        # Defensize Unit Check (mm to meters)
                        if target.max() > 80.0:
                            target = target / 1000.0

                        batch_metrics = compute_metrics_per_image(
                            mean,
                            target,
                            min_depth=min_d,
                            max_depth=cfg.dataset.get("max_depth_eval", cfg.dataset.get("max_depth", 10.0)),
                            use_eigen_crop=True,
                        )
                        batch_count = mean.shape[0]
                        depth_accum[0] += batch_metrics["rmse"] * batch_count
                        depth_accum[1] += batch_metrics["abs_rel"] * batch_count
                        depth_accum[2] += batch_metrics["log10"] * batch_count
                        depth_accum[3] += batch_metrics["delta1"] * batch_count
                        depth_accum[4] += batch_metrics["delta2"] * batch_count
                        depth_accum[5] += batch_metrics["delta3"] * batch_count
                        depth_accum[6] += batch_count
                    else:
                        # Toy Dataset Logic
                        val_loss = criterion(mean_for_loss, target, variance=variance, interpolate=interpolate)
                        batch_count = target.numel()
                        val_loss_sum += val_loss.detach() * batch_count
                        val_loss_count += batch_count

        # =========================================================
        #   SYNC RESULTS BACK TO ALL RANKS
        # =========================================================
        
        if distributed:
            dist.barrier() # 2. Wait for Rank 0 to finish eval
            
            # Broadcast results so Scheduler works on all ranks
            val_tensors = torch.stack([val_loss_sum, val_loss_count])
            dist.broadcast(val_tensors, src=0)
            val_loss_sum, val_loss_count = val_tensors
            
            # Broadcast metrics (optional, but good if you want to log on other ranks later)
            if cfg.dataset.name == "nyu_depth":
                dist.broadcast(depth_accum, src=0)

        # Compute Final Averages
        val_loss_avg = (val_loss_sum / max(val_loss_count, 1.0)).item()

        # Update Plateau Scheduler (on all ranks, since we synced loss)
        if cfg.uncertainty.beta_strategy == "plateau":
            scheduler.get_beta(global_step, current_loss=val_loss_avg)

        # Logging (Rank 0 Only)
        if rank == 0:
            if cfg.dataset.name == "nyu_depth":
                vals = depth_accum.cpu().tolist()
                total_images = max(vals[6], 1.0)
                val_log = {
                    "val/rmse": vals[0] / total_images,
                    "val/abs_rel": vals[1] / total_images,
                    "val/log10": vals[2] / total_images,
                    "val/delta1": vals[3] / total_images,
                    "val/delta2": vals[4] / total_images,
                    "val/delta3": vals[5] / total_images,
                }
            else:
                val_log = {"val/loss": val_loss_avg}
            
            if cfg.logging.use_wandb and wandb is not None:
                wandb.log(val_log, step=global_step)

    # --- 7. Cleanup ---
    if rank == 0 and cfg.logging.use_wandb and wandb is not None:
        wandb.finish()

    if rank == 0:
        run_dir = HydraConfig.get().runtime.output_dir
        ckpt_path = os.path.join(run_dir, "checkpoint.pt")
        state = model.module.state_dict() if distributed else model.state_dict()
        torch.save({"model": state}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

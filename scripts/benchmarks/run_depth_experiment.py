import os
import sys
from pathlib import Path
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import NYUDepthDataset
from src.models import DepthUNet
from src.modules import BetaScheduler, GaussianLogLikelihoodLoss
from src.utils import get_device, set_seed
from src.utils.depth_metrics import compute_depth_metrics


def make_loader(
    data_path: str,
    gt_path: str,
    filenames_file: str,
    input_size: tuple[int, int],
    batch_size: int,
    use_dummy_data: bool,
    n_samples: int,
    do_random_rotate: bool,
    degree: float,
    num_workers: int,
    mode: str,
):
    ds = NYUDepthDataset(
        data_path=data_path,
        gt_path=gt_path,
        filenames_file=filenames_file,
        input_size=input_size,
        mode=mode,
        use_dummy_data=use_dummy_data,
        n_samples=n_samples,
        do_random_rotate=do_random_rotate,
        degree=degree,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == "train"), num_workers=num_workers, pin_memory=True)


def log_samples(images, depth_gt, depth_mean, depth_var, step: int):
    if wandb is None:
        return
    img = images[0].detach().cpu()
    gt = depth_gt[0].detach().cpu()
    pred = depth_mean[0].detach().cpu()
    var = depth_var[0].detach().cpu()
    grid = torch.cat([img, gt.repeat(3, 1, 1), pred.repeat(3, 1, 1), var.repeat(3, 1, 1)], dim=2)
    wandb.log({"samples": wandb.Image(grid, caption="Input | GT | Pred Mean | Pred Var")}, step=step)


@hydra.main(config_path=str(ROOT / "configs"), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = get_device(cfg.get("device", "auto"))

    data_cfg = cfg.dataset
    input_size = (data_cfg.input_height, data_cfg.input_width)
    batch_size = data_cfg.get("batch_size", cfg.hyperparameters.batch_size)
    num_workers = data_cfg.get("num_workers", 0)

    train_loader = make_loader(
        data_cfg.data_path,
        data_cfg.gt_path,
        data_cfg.filenames_file,
        input_size,
        batch_size,
        data_cfg.use_dummy_data,
        data_cfg.n_samples,
        data_cfg.get("do_random_rotate", False),
        data_cfg.get("degree", 2.5),
        num_workers,
        mode="train",
    )
    val_file = data_cfg.get("filenames_file_eval", "") or data_cfg.filenames_file
    val_loader = make_loader(
        data_cfg.get("data_path_eval", data_cfg.data_path),
        data_cfg.get("gt_path_eval", data_cfg.gt_path),
        val_file,
        input_size,
        batch_size,
        data_cfg.use_dummy_data,
        data_cfg.n_samples,
        False,
        data_cfg.get("degree", 2.5),
        num_workers,
        mode="val",
    )

    model = DepthUNet(
        encoder=cfg.model.encoder,
        pretrained=cfg.model.pretrained,
        min_depth=cfg.model.get("min_depth", 1e-3),
        min_var=cfg.model.get("min_var", 1e-6),
        max_val=cfg.model.get("max_val", 10.0),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.hyperparameters.lr)
    criterion = GaussianLogLikelihoodLoss()
    scheduler = BetaScheduler(
        strategy="delayed_linear",
        start_beta=1.0,
        end_beta=0.5,
        total_steps=cfg.hyperparameters.epochs,
        warmup_steps=cfg.hyperparameters.stage1_epochs,
    )

    if cfg.logging.use_wandb and wandb is not None:
        wandb.init(project=cfg.logging.project_name, config=OmegaConf.to_container(cfg, resolve=True))

    global_step = 0
    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        if epoch == cfg.hyperparameters.stage1_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.hyperparameters.lr_stage2
        beta_value = scheduler.get_beta(epoch)
        for images, depth_gt in train_loader:
            images = images.to(device)
            depth_gt = depth_gt.to(device)
            mean, var = model(images)
            loss = criterion(mean, depth_gt, variance=var, interpolate=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metrics = compute_depth_metrics(mean, depth_gt)
            if cfg.logging.use_wandb and wandb is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/rmse": metrics["rmse"],
                        "train/abs_rel": metrics["abs_rel"],
                        "train/nll": loss.item(),
                        "train/beta": beta_value,
                    },
                    step=global_step,
                )
                if global_step % 50 == 0:
                    log_samples(images, depth_gt, mean, var, global_step)

            global_step += 1

        model.eval()
        val_metrics: Dict[str, float] = {}
        count = 0
        with torch.no_grad():
            for images, depth_gt in val_loader:
                images = images.to(device)
                depth_gt = depth_gt.to(device)
                mean, var = model(images)
                batch_metrics = compute_depth_metrics(mean, depth_gt)
                for k, v in batch_metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0.0) + v * len(images)
                count += len(images)
        for k in val_metrics:
            val_metrics[k] /= max(count, 1)

        if cfg.logging.use_wandb and wandb is not None:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

    if cfg.logging.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

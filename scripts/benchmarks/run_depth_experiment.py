import argparse
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
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
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=(mode == "train"), num_workers=0, pin_memory=True)


def log_samples(images, depth_gt, depth_mean, depth_var, step: int):
    if wandb is None:
        return
    img = images[0].detach().cpu()
    gt = depth_gt[0].detach().cpu()
    pred = depth_mean[0].detach().cpu()
    var = depth_var[0].detach().cpu()
    grid = torch.cat([img, gt.repeat(3, 1, 1), pred.repeat(3, 1, 1), var.repeat(3, 1, 1)], dim=2)
    wandb.log({"samples": wandb.Image(grid, caption="Input | GT | Pred Mean | Pred Var")}, step=step)


def main():
    parser = argparse.ArgumentParser(description="Depth estimation experiment with dynamic beta.")
    parser.add_argument("--data_path", type=str, default="/path/to/nyu/rgb")
    parser.add_argument("--gt_path", type=str, default="/path/to/nyu/depth")
    parser.add_argument("--filenames_file", type=str, default="/path/to/nyu/train_files.txt")
    parser.add_argument("--val_filenames_file", type=str, default="")
    parser.add_argument("--input_height", type=int, default=256)
    parser.add_argument("--input_width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_stage2", type=float, default=1e-5)
    parser.add_argument("--use_dummy_data", action="store_true")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="beta-nll-depth")
    args = parser.parse_args()

    set_seed(42)
    device = get_device(args.device)

    input_size = (args.input_height, args.input_width)
    train_loader = make_loader(
        args.data_path,
        args.gt_path,
        args.filenames_file,
        input_size,
        args.batch_size,
        args.use_dummy_data,
        args.n_samples,
        mode="train",
    )
    val_file = args.val_filenames_file or args.filenames_file
    val_loader = make_loader(
        args.data_path,
        args.gt_path,
        val_file,
        input_size,
        args.batch_size,
        args.use_dummy_data,
        args.n_samples,
        mode="val",
    )

    model = DepthUNet(encoder="resnet50", pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = GaussianLogLikelihoodLoss()
    scheduler = BetaScheduler(
        strategy="delayed_linear",
        start_beta=1.0,
        end_beta=0.5,
        total_steps=args.epochs,
        warmup_steps=args.warmup_epochs,
    )

    if args.use_wandb and wandb is not None:
        wandb.init(project=args.wandb_project, config=vars(args))

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        if epoch == args.warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr_stage2
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
            if args.use_wandb and wandb is not None:
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

        if args.use_wandb and wandb is not None:
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=global_step)

    if args.use_wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()

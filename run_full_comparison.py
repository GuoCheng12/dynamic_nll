import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict

import torch
from omegaconf import OmegaConf

from src.dataset import build_dataloaders
from src.utils import evaluate_head_tail, set_seed
from train import instantiate_model


def run_training(run_dir: Path, overrides: list[str]) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    cmd = [sys.executable, "train.py"] + overrides + [f"hydra.run.dir={run_dir}", "logging.use_wandb=false"]
    subprocess.run(cmd, check=True)


def load_model_cfg(run_dir: Path, ckpt_name: str = "checkpoint.pt"):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    set_seed(cfg.seed)
    model = instantiate_model(cfg)
    state = torch.load(run_dir / ckpt_name, map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return cfg, model


def eval_run(run_dir: Path) -> Dict[str, float]:
    cfg, model = load_model_cfg(run_dir)
    _, _, test_loader = build_dataloaders(cfg.hyperparameters.batch_size, seed=cfg.seed)
    dataset = test_loader.dataset.dataset
    test_indices = torch.tensor(test_loader.dataset.indices)
    metrics = evaluate_head_tail(model, dataset, indices=test_indices)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Run full comparison: fixed betas and dynamic beta.")
    parser.add_argument("--out_root", type=str, default="outputs/full_compare", help="Root dir for comparison runs.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    experiments = {
        "Beta 1.0": [
            "uncertainty.beta_strategy=constant",
            "uncertainty.beta_start=1.0",
            "uncertainty.beta_end=1.0",
            f"hyperparameters.epochs={args.epochs}",
        ],
        "Dynamic_Old": [
            "uncertainty.beta_strategy=linear_decay",
            "uncertainty.beta_start=1.0",
            "uncertainty.beta_end=0.0",
            "uncertainty.total_steps=80",
            "hyperparameters.stage1_epochs=0",
            "hyperparameters.lr_stage2=0.001",
            f"hyperparameters.epochs={args.epochs}",
        ],
        "Dynamic_Fix": [
            "uncertainty.beta_strategy=linear_decay",
            "uncertainty.beta_start=1.0",
            "uncertainty.beta_end=0.0",
            "uncertainty.total_steps=80",
            "hyperparameters.stage1_epochs=50",
            "hyperparameters.lr_stage2=0.0001",
            f"hyperparameters.epochs={args.epochs}",
        ],
        "Dynamic_Soft": [
            "uncertainty.beta_strategy=linear_decay",
            "uncertainty.beta_start=1.0",
            "uncertainty.beta_end=0.1",
            "uncertainty.total_steps=80",
            "hyperparameters.stage1_epochs=50",
            "hyperparameters.lr_stage2=0.0001",
            f"hyperparameters.epochs={args.epochs}",
        ],
    }

    run_dirs = {}
    for name, overrides in experiments.items():
        run_dir = out_root / name.replace(" ", "_").lower()
        run_training(run_dir, overrides)
        run_dirs[name] = run_dir

    results: Dict[str, Dict[str, float]] = {}
    for name, run_dir in run_dirs.items():
        metrics = eval_run(run_dir)
        results[name] = metrics

    header = f"{'Experiment':<14} | {'MSE (Tail)':>10} | {'NLL (Tail)':>10} | {'ECE (Global)':>12} | {'ECE (Tail)':>11}"
    print(header)
    print("-" * len(header))
    for name in ["Beta 1.0", "Dynamic_Old", "Dynamic_Fix", "Dynamic_Soft"]:
        m = results[name]
        print(
            f"{name:<14} | {m['MSE_Tail']:10.4f} | {m['NLL_Tail']:10.4f} | {m['ECE_Global']:12.4f} | {m['ECE_Tail']:11.4f}"
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders, build_paper_sine_dataloaders
from src.utils import evaluate_head_tail, expected_calibration_error, get_device, mae, nll, rmse, set_seed
from train import instantiate_model


def run_training(run_dir: Path, overrides: list[str]) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        *overrides,
        f"hydra.run.dir={run_dir}",
    ]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def load_model_cfg(run_dir: Path, device: torch.device):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    set_seed(cfg.seed)
    model = instantiate_model(cfg).to(device)
    state = torch.load(run_dir / "checkpoint.pt", map_location=device)
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(state_dict)
    model.eval()
    return cfg, model


def evaluate_toy_run(cfg, model: torch.nn.Module, device: torch.device, run_dir: Path) -> Dict[str, float]:
    _, _, test_loader = build_dataloaders(
        batch_size=cfg.hyperparameters.batch_size,
        splits=(
            float(cfg.dataset.split.train),
            float(cfg.dataset.split.val),
            float(cfg.dataset.split.test),
        ),
        size=cfg.dataset.get("size", 5000),
        normalize=cfg.dataset.get("normalize", True),
        seed=cfg.seed,
        eval_batch_size=cfg.hyperparameters.get("eval_batch_size", cfg.hyperparameters.batch_size),
    )
    dataset = test_loader.dataset.dataset
    test_indices = torch.tensor(test_loader.dataset.indices)
    metrics = evaluate_head_tail(model, dataset, indices=test_indices, device=device)
    metrics.update(load_controller_metrics(run_dir))
    return metrics


def evaluate_paper_sine_run(cfg, model: torch.nn.Module, device: torch.device) -> Dict[str, float]:
    _, _, test_loader = build_paper_sine_dataloaders(
        cfg.dataset,
        batch_size=cfg.hyperparameters.batch_size,
        seed=cfg.seed,
        eval_batch_size=cfg.hyperparameters.get("eval_batch_size", cfg.hyperparameters.batch_size),
    )
    metrics = {"MSE": 0.0, "MAE": 0.0, "RMSE": 0.0, "NLL": 0.0, "ECE": 0.0, "Var_MSE": 0.0}
    count = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            mean, variance = model(data)
            batch = len(data)
            true_var = test_loader.dataset.true_var[count : count + batch].to(device)
            metrics["MSE"] += torch.mean((mean - target) ** 2).item() * batch
            metrics["MAE"] += mae(mean, target) * batch
            metrics["RMSE"] += rmse(mean, target) * batch
            metrics["NLL"] += nll(mean, target, variance) * batch
            metrics["ECE"] += expected_calibration_error(mean, variance, target) * batch
            metrics["Var_MSE"] += torch.mean((variance - true_var) ** 2).item() * batch
            count += batch

    for key in metrics:
        metrics[key] /= max(count, 1)
    return metrics


def evaluate_run(run_dir: Path) -> Dict[str, float]:
    device = get_device("auto")
    cfg, model = load_model_cfg(run_dir, device)
    if cfg.dataset.name == "paper_sine":
        metrics = evaluate_paper_sine_run(cfg, model, device)
        metrics.update(load_controller_metrics(run_dir))
        return metrics
    if cfg.dataset.name == "toy_regression":
        return evaluate_toy_run(cfg, model, device, run_dir)
    raise ValueError(f"Dataset {cfg.dataset.name} is not supported by this benchmark CLI.")


def load_controller_metrics(run_dir: Path) -> Dict[str, float]:
    summary_path = run_dir / "controller_summary.json"
    if not summary_path.exists():
        summary_path = run_dir / "toy_controller_summary.json"
    if not summary_path.exists():
        return {
            "Align_Final": 0.0,
            "Collapse_Events": 0.0,
            "First_Beta_Release_Epoch": -1.0,
            "First_Gamma_Release_Epoch": -1.0,
            "TrueVar_Corr_Final": 0.0,
            "Var_MSE_Final": 0.0,
            "Grad_Ratio_Final": 0.0,
            "Grad_Cosine_Final": 0.0,
        }
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)
    return {
        "Align_Final": float(summary.get("final_align_score", 0.0)),
        "Collapse_Events": float(summary.get("collapse_event_count", 0.0)),
        "First_Beta_Release_Epoch": float(summary.get("first_beta_release_epoch", -1)),
        "First_Gamma_Release_Epoch": float(summary.get("first_gamma_release_epoch", -1)),
        "TrueVar_Corr_Final": float(summary.get("final_true_var_corr", 0.0)),
        "Var_MSE_Final": float(summary.get("final_var_mse", 0.0)),
        "Grad_Ratio_Final": float(summary.get("final_grad_ratio", 0.0)),
        "Grad_Cosine_Final": float(summary.get("final_grad_cosine", 0.0)),
    }


def build_common_overrides(args: argparse.Namespace, seed: int) -> list[str]:
    overrides = [
        f"dataset={args.dataset}",
        "model=mlp",
        f"seed={seed}",
        f"hyperparameters.epochs={args.epochs}",
        f"hyperparameters.batch_size={args.batch_size}",
        f"hyperparameters.eval_batch_size={args.eval_batch_size}",
        f"hyperparameters.lr={args.lr}",
        f"hyperparameters.weight_decay={args.weight_decay}",
        f"hyperparameters.num_workers={args.num_workers}",
        f"hyperparameters.checkpoint_interval={args.checkpoint_interval}",
        f"uncertainty.total_steps={args.schedule_steps or args.epochs}",
        f"logging.use_wandb={'true' if args.use_wandb else 'false'}",
        f"logging.project_name={args.project_name}",
    ]
    if args.dataset == "toy_regression" and args.dataset_size is not None:
        overrides.append(f"dataset.size={args.dataset_size}")
    if args.dataset == "paper_sine" and args.dataset_size is not None:
        overrides.extend(
            [
                f"dataset.n_samples={args.dataset_size}",
                f"dataset.val_samples={args.dataset_size}",
                f"dataset.test_samples={args.dataset_size}",
            ]
        )
    overrides.extend(args.extra_override)
    return overrides


def format_fixed_beta_name(beta: float) -> str:
    return f"fixed_beta_{beta:.1f}"


def format_schedule_endpoint(value: float) -> str:
    return f"{value:.1f}"


def format_dynamic_beta_name(strategy: str, start_beta: float, end_beta: float) -> str:
    schedule = "linear" if strategy == "linear_decay" else strategy
    return (
        f"dynamic_beta_{schedule}_"
        f"{format_schedule_endpoint(start_beta)}_to_{format_schedule_endpoint(end_beta)}"
    )


def format_faithful_schedule_name(start_lambda: float, end_lambda: float, strategy: str) -> str:
    schedule = "linear" if strategy == "linear_decay" else strategy
    return (
        f"faithful_{schedule}_lambda_"
        f"{format_schedule_endpoint(start_lambda)}_to_{format_schedule_endpoint(end_lambda)}"
    )


def canonical_method_name(method: str, args: argparse.Namespace) -> str:
    if method in {"current_dynamic_beta", "linear", "cosine", "dynamic_beta"}:
        strategy = args.current_dynamic_strategy if method in {"current_dynamic_beta", "dynamic_beta"} else method
        return format_dynamic_beta_name(strategy, args.beta_start, args.beta_end)
    if method in {"current_faithful", "faithful"}:
        return format_faithful_schedule_name(
            args.faithful_lambda_start,
            args.faithful_lambda_end,
            args.faithful_lambda_strategy,
        )
    if method == "fixed_beta_baseline":
        return format_fixed_beta_name(args.fixed_beta_baseline)
    if method == "joint_nll_baseline":
        return format_fixed_beta_name(0.0)
    if method == "closed_loop_beta_only":
        return "adaptive_beta_nll_beta_only"
    if method == "closed_loop_beta_gamma":
        return "adaptive_beta_nll_beta_gamma"
    return method


def controller_overrides(
    args: argparse.Namespace,
    mode: str,
    fixed_gamma_t: float | None = None,
) -> list[str]:
    overrides = [
        "uncertainty.controller.enabled=true",
        f"uncertainty.controller.mode={mode}",
        f"uncertainty.controller.signal={args.controller_signal}",
        f"uncertainty.controller.update_interval={args.controller_update_interval}",
        f"uncertainty.controller.warmup_epochs={args.controller_warmup_epochs}",
        f"uncertainty.controller.beta_min={args.controller_beta_min}",
        f"uncertainty.controller.beta_max={args.controller_beta_max}",
        f"uncertainty.controller.gamma_min={args.controller_gamma_min}",
        f"uncertainty.controller.gamma_max={args.controller_gamma_max}",
        f"uncertainty.controller.beta_step={args.controller_beta_step}",
        f"uncertainty.controller.gamma_step={args.controller_gamma_step}",
        f"uncertainty.controller.collapse_thresh={args.controller_collapse_thresh}",
        f"uncertainty.controller.align_thresh={args.controller_align_thresh}",
        f"uncertainty.controller.rmse_plateau_thresh={args.controller_rmse_plateau_thresh}",
        f"uncertainty.controller.grad_ratio_thresh={args.controller_grad_ratio_thresh}",
        f"uncertainty.controller.grad_cosine_thresh={args.controller_grad_cosine_thresh}",
        f"uncertainty.controller.gamma_release_beta_thresh={args.controller_gamma_release_beta_thresh}",
    ]
    if fixed_gamma_t is not None:
        overrides.append(f"uncertainty.controller.fixed_gamma_t={fixed_gamma_t}")
    return overrides


def build_run_specs(args: argparse.Namespace) -> List[Tuple[str, list[str]]]:
    run_specs_by_name: Dict[str, list[str]] = {}
    for method in args.methods:
        run_specs_by_name[canonical_method_name(method, args)] = build_method_overrides(method, args)
    for beta in args.fixed_betas:
        run_specs_by_name[format_fixed_beta_name(beta)] = [
            "uncertainty.loss_type=beta_nll",
            "uncertainty.beta_strategy=constant",
            f"uncertainty.beta_start={beta}",
            f"uncertainty.beta_end={beta}",
        ]
    return list(run_specs_by_name.items())


def build_method_overrides(method: str, args: argparse.Namespace) -> list[str]:
    if method in {"current_dynamic_beta", "dynamic_beta"}:
        return [
            "uncertainty.loss_type=beta_nll",
            "uncertainty.controller.enabled=false",
            f"uncertainty.beta_strategy={args.current_dynamic_strategy}",
            f"uncertainty.beta_start={args.beta_start}",
            f"uncertainty.beta_end={args.beta_end}",
        ]
    if method == "current_faithful":
        return [
            "uncertainty.loss_type=faithful",
            "uncertainty.controller.enabled=false",
            f"uncertainty.faithful_lambda_strategy={args.faithful_lambda_strategy}",
            f"uncertainty.faithful_lambda_start={args.faithful_lambda_start}",
            f"uncertainty.faithful_lambda_end={args.faithful_lambda_end}",
        ]
    if method == "fixed_beta_baseline":
        return [
            "uncertainty.loss_type=beta_nll",
            "uncertainty.controller.enabled=false",
            "uncertainty.beta_strategy=constant",
            f"uncertainty.beta_start={args.fixed_beta_baseline}",
            f"uncertainty.beta_end={args.fixed_beta_baseline}",
        ]
    if method == "joint_nll_baseline":
        return [
            "uncertainty.loss_type=beta_nll",
            "uncertainty.controller.enabled=false",
            "uncertainty.beta_strategy=constant",
            "uncertainty.beta_start=0.0",
            "uncertainty.beta_end=0.0",
        ]
    if method == "closed_loop_beta_only":
        return [
            "uncertainty.loss_type=beta_nll",
            "uncertainty.beta_strategy=constant",
            f"uncertainty.beta_start={args.controller_beta_max}",
            f"uncertainty.beta_end={args.controller_beta_max}",
            *controller_overrides(args, "beta_only"),
        ]
    if method == "closed_loop_beta_gamma":
        return [
            "uncertainty.loss_type=beta_nll",
            "uncertainty.beta_strategy=constant",
            f"uncertainty.beta_start={args.controller_beta_max}",
            f"uncertainty.beta_end={args.controller_beta_max}",
            *controller_overrides(args, "beta_gamma"),
        ]
    if method == "adaptive_faithful_lambda":
        return [
            "uncertainty.loss_type=faithful",
            *controller_overrides(args, "beta_only", fixed_gamma_t=0.0),
        ]
    if method == "adaptive_faithful_lambda_gamma":
        return [
            "uncertainty.loss_type=faithful",
            *controller_overrides(args, "beta_gamma"),
        ]
    if method == "linear":
        return [
            "uncertainty.loss_type=beta_nll",
            "uncertainty.controller.enabled=false",
            "uncertainty.beta_strategy=linear_decay",
            f"uncertainty.beta_start={args.beta_start}",
            f"uncertainty.beta_end={args.beta_end}",
        ]
    if method == "cosine":
        return [
            "uncertainty.loss_type=beta_nll",
            "uncertainty.controller.enabled=false",
            "uncertainty.beta_strategy=cosine",
            f"uncertainty.beta_start={args.beta_start}",
            f"uncertainty.beta_end={args.beta_end}",
        ]
    if method == "faithful":
        return [
            "uncertainty.loss_type=faithful",
            "uncertainty.controller.enabled=false",
            f"uncertainty.faithful_lambda_strategy={args.faithful_lambda_strategy}",
            f"uncertainty.faithful_lambda_start={args.faithful_lambda_start}",
            f"uncertainty.faithful_lambda_end={args.faithful_lambda_end}",
        ]
    raise ValueError(f"Unknown method: {method}")


def summarize_metrics(rows: Dict[str, list[Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, metrics_list in rows.items():
        metric_names = metrics_list[0].keys()
        summary[method] = {}
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in metrics_list]
            summary[method][metric_name] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            }
    return summary


def print_summary(dataset: str, summary: Dict[str, Dict[str, Dict[str, float]]], methods: Iterable[str]) -> None:
    if dataset == "paper_sine":
        metric_order = [
            "MSE",
            "NLL",
            "ECE",
            "Var_MSE",
            "Align_Final",
            "TrueVar_Corr_Final",
            "Collapse_Events",
            "First_Beta_Release_Epoch",
            "First_Gamma_Release_Epoch",
        ]
    else:
        metric_order = [
            "MSE_Global",
            "NLL_Global",
            "ECE_Global",
            "MSE_Tail",
            "NLL_Tail",
            "ECE_Tail",
            "Align_Final",
            "Collapse_Events",
            "First_Beta_Release_Epoch",
            "First_Gamma_Release_Epoch",
        ]

    header = "Method".ljust(14) + " | " + " | ".join(metric.rjust(18) for metric in metric_order)
    print(header)
    print("-" * len(header))
    for method in methods:
        fields = []
        for metric in metric_order:
            metric_stats = summary[method][metric]
            fields.append(f"{metric_stats['mean']:.4f}±{metric_stats['std']:.4f}".rjust(18))
        print(method.ljust(14) + " | " + " | ".join(fields))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fair comparison between linear beta, cosine beta, and Faithful objectives."
    )
    parser.add_argument("--dataset", choices=["toy_regression", "paper_sine"], default="toy_regression")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "linear",
            "cosine",
            "dynamic_beta",
            "faithful",
            "current_dynamic_beta",
            "current_faithful",
            "adaptive_faithful_lambda",
            "adaptive_faithful_lambda_gamma",
            "fixed_beta_baseline",
            "joint_nll_baseline",
            "closed_loop_beta_only",
            "closed_loop_beta_gamma",
        ],
        default=["linear", "cosine", "faithful"],
    )
    parser.add_argument("--fixed-betas", nargs="*", type=float, default=[])
    parser.add_argument("--out-root", type=str, default="outputs/fair_uncertainty_compare")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    parser.add_argument("--schedule-steps", type=int, default=None)
    parser.add_argument("--dataset-size", type=int, default=None)
    parser.add_argument("--beta-start", type=float, default=1.0)
    parser.add_argument("--beta-end", type=float, default=0.0)
    parser.add_argument("--current-dynamic-strategy", choices=["linear_decay", "cosine"], default="linear_decay")
    parser.add_argument("--fixed-beta-baseline", type=float, default=1.0)
    parser.add_argument("--faithful-lambda-strategy", type=str, default="linear_decay")
    parser.add_argument("--faithful-lambda-start", type=float, default=0.0)
    parser.add_argument("--faithful-lambda-end", type=float, default=1.0)
    parser.add_argument("--controller-update-interval", type=int, default=50)
    parser.add_argument("--controller-warmup-epochs", type=int, default=200)
    parser.add_argument("--controller-signal", choices=["metrics", "gradient"], default="metrics")
    parser.add_argument("--controller-beta-min", type=float, default=0.2)
    parser.add_argument("--controller-beta-max", type=float, default=1.0)
    parser.add_argument("--controller-gamma-min", type=float, default=0.0)
    parser.add_argument("--controller-gamma-max", type=float, default=1.0)
    parser.add_argument("--controller-beta-step", type=float, default=0.1)
    parser.add_argument("--controller-gamma-step", type=float, default=0.1)
    parser.add_argument("--controller-collapse-thresh", type=float, default=0.02)
    parser.add_argument("--controller-align-thresh", type=float, default=0.30)
    parser.add_argument("--controller-rmse-plateau-thresh", type=float, default=0.005)
    parser.add_argument("--controller-grad-ratio-thresh", type=float, default=1.0)
    parser.add_argument("--controller-grad-cosine-thresh", type=float, default=0.0)
    parser.add_argument("--controller-gamma-release-beta-thresh", type=float, default=0.6)
    parser.add_argument("--project-name", type=str, default="uncertainty_compare")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--extra-override", action="append", default=[], help="Extra Hydra override shared by all runs.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    run_specs = build_run_specs(args)
    method_names = [name for name, _ in run_specs]

    metadata = {
        "dataset": args.dataset,
        "methods": method_names,
        "seeds": args.seeds,
        "shared_hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_workers": args.num_workers,
            "schedule_steps": args.schedule_steps or args.epochs,
            "current_dynamic_strategy": args.current_dynamic_strategy,
            "fixed_beta_baseline": args.fixed_beta_baseline,
            "controller_update_interval": args.controller_update_interval,
            "controller_warmup_epochs": args.controller_warmup_epochs,
            "controller_signal": args.controller_signal,
            "controller_grad_ratio_thresh": args.controller_grad_ratio_thresh,
            "controller_grad_cosine_thresh": args.controller_grad_cosine_thresh,
            "controller_gamma_release_beta_thresh": args.controller_gamma_release_beta_thresh,
        },
        "fairness_contract": "All methods reuse the same dataset config, seed, model, optimizer, LR schedule, epoch count, and output checkpoint policy. Only the uncertainty objective/schedule differs.",
    }
    (out_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    results: Dict[str, list[Dict[str, float]]] = defaultdict(list)
    for seed in args.seeds:
        common_overrides = build_common_overrides(args, seed)
        for method_name, method_overrides in run_specs:
            run_name = f"{args.dataset}_{method_name}_seed{seed}"
            run_dir = out_root / method_name / f"seed_{seed}"
            overrides = common_overrides + method_overrides + [f"logging.run_name={run_name}"]
            print(f"[run] method={method_name} seed={seed} -> {run_dir}")
            run_training(run_dir, overrides)
            metrics = evaluate_run(run_dir)
            results[method_name].append(metrics)

    summary = summarize_metrics(results)
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(args.dataset, summary, method_names)


if __name__ == "__main__":
    main()

# Repository Guidelines

Project: Dynamic β-NLL Uncertainty Estimation Framework — modular PyTorch research to study dynamic β scheduling for heteroscedastic regression.

## Project Structure & Module Organization
- Directory layout: `configs/` (Hydra: `config.yaml`, `dataset/`, `experiment/`), `src/data/` (datasets), `src/models/` (backbones), `src/modules/` (loss/schedulers), `src/utils.py`, root `train.py`, `eval.py`, and `README.md`.
- Models in `src/models/mlp.py` must output mean and log_variance. Keep datasets swappable via factories in `src/data/base_dataset.py`; utilities (metrics, grad norms) live in `src/utils.py`.
- Experimental scripts live under `scripts/benchmarks/`, analysis helpers under `scripts/analysis/`, and plotting under `scripts/visualization/`.

## Build, Test, and Development Commands
- Create env: `python -m venv .venv && source .venv/bin/activate`; install deps with `pip install -r requirements.txt` (add `requirements-dev.txt` for tooling).
- Run training with Hydra overrides, e.g., `python train.py hydra.run.dir=outputs +uncertainty.beta_strategy=linear_decay`.
- Evaluate with `python eval.py checkpoint=<path> dataset=<name>`.
- Tests: `pytest` or `pytest tests/test_loss.py -k beta` during iteration.

## Core Logic: Loss & Scheduler (src/modules/loss.py)
- Keep `GaussianLogLikelihoodLoss` exactly as provided (interpolation, masking, β-weighted NLL). Do not change the math.
- Implement `BetaScheduler(strategy, start_beta, end_beta, total_steps)` with `get_beta(step, current_loss=None)`; support `constant`, `linear_decay`, `cosine`, and optional `plateau` (uses validation loss).
- Ensure variance passed to the loss is positive (e.g., apply `softplus`/`exp` in `src/models/mlp.py`).

## Training Loop Expectations (train.py)
- Initialize model/optimizer/loaders, then `criterion = GaussianLogLikelihoodLoss()` and a `BetaScheduler` from config.
- Each step: `new_beta = scheduler.get_beta(global_step)` then `criterion.beta = new_beta`; forward to get `(mean, var)`, compute loss, backward, optimizer step.
- Log `loss`, `mse`, `nll`, `beta_value`, and gradient norms for mean vs variance heads (research focus). Use `wandb.init(config=cfg)`.

## Configuration & Experimentation (configs/config.yaml)
- Hydra defaults should select dataset/model; `uncertainty` block captures `loss_type`, `beta_strategy` (constant|linear_decay|cosine|plateau), `beta_start`, `beta_end`.
- Logging config: `project_name` and `run_name` interpolated from β settings; keep runs reproducible by seeding Python/NumPy/PyTorch in `train.py`.

## Testing & Evaluation
- Tests live in `tests/`, mirroring `src/`. Name files `test_<module>.py`; focus on scheduler edge cases and loss invariants (masking, interpolation).
- Evaluation should report RMSE/MAE, NLL, and ECE; compare dynamic vs fixed β runs and record metrics in WandB.

## Coding Style & PR Hygiene
- Follow PEP 8, 4-space indent, snake_case for functions/vars, PascalCase classes, UPPER_SNAKE_CASE constants; prefer full type hints.
- Format with Black/Ruff and sort imports; treat type checker and linter warnings as errors.
- Commits: imperative subject (`Add cosine beta schedule`); PRs should state intent, config used, metrics before/after, and any new flags. Never commit data or secrets; add sensitive patterns to `.gitignore`.

# Dynamic β-NLL Uncertainty Estimation Framework

Modular PyTorch research framework to study dynamic β scheduling for heteroscedastic regression.

## Getting Started
- Create env: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt` (add Hydra, PyTorch, WandB, PyTest, Ruff/Black)
- Run training: `python train.py`
- Evaluate: `python eval.py checkpoint=path/to/ckpt.pt`

## Key Files
- `configs/` Hydra configs (datasets, models, experiments)
- `src/loss.py` GaussianLogLikelihoodLoss (do not change math) and BetaScheduler
- `src/models.py` backbones outputting mean and log_variance
- `train.py` dynamic β training loop with logging and grad norms
- `eval.py` calibration and NLL evaluation

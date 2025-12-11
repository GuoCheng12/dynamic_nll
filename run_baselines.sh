#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="outputs"
mkdir -p "$RUN_DIR"

run_exp() {
  beta="$1"
  out_dir="${RUN_DIR}/beta_${beta}"
  rm -rf "$out_dir"
  mkdir -p "$out_dir"
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
  echo "Running beta=${beta} -> ${out_dir}"
  "${PYTHON_BIN}" train.py uncertainty.beta_strategy=constant uncertainty.beta_start="${beta}" uncertainty.beta_end="${beta}" hydra.run.dir="${out_dir}" logging.use_wandb=false
}

run_exp 0.0
run_exp 0.5
run_exp 1.0

echo "Done. Results saved under ${RUN_DIR}/beta_*"

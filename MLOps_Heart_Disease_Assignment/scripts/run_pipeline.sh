#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."

run_step() {
  local title="$1"; shift
  printf '\n===== %s =====\n' "$title"
  "$@"
}

run_step "Download UCI dataset" python scripts/download_data.py
run_step "Generate EDA outputs" python scripts/eda.py
run_step "Train, compare, and track models with MLflow" python scripts/train.py
run_step "Run Ruff checks" ruff check .
run_step "Run unit tests" python -m pytest -q

printf '\nPipeline completed successfully.\n'
printf 'Model: artifacts/model.joblib\n'
printf 'Metadata/comparison: artifacts/model_metadata.json, artifacts/model_comparison.csv\n'
printf 'EDA/model plots: reports/figures\n'
printf 'MLflow database: mlflow.db\n'

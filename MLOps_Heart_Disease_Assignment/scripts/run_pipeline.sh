#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
RUFF="$ROOT/.venv/bin/ruff"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Project virtual environment not found at $PYTHON"
    echo "Create or repair .venv before running the pipeline."
    exit 1
fi

if [[ ! -x "$RUFF" ]]; then
    echo "ERROR: Ruff not found at $RUFF"
    echo "Check the project's virtual environment dependencies."
    exit 1
fi

run_step() {
    local title="$1"
    shift
    printf '\n===== %s =====\n' "$title"
    "$@"
}

run_step "Verify project Python environment" \
    "$PYTHON" -c "import sys, pandas, sklearn, mlflow; print('Python:', sys.executable); print('Core dependencies: OK')"

run_step "Download UCI dataset" \
    "$PYTHON" scripts/download_data.py

run_step "Validate and preprocess dataset" \
    "$PYTHON" scripts/preprocess.py

run_step "Generate EDA outputs" \
    "$PYTHON" scripts/eda.py

run_step "Train, compare, and track models with MLflow" \
    "$PYTHON" scripts/train.py

run_step "Run Ruff checks" \
    "$RUFF" check .

run_step "Run unit tests" \
    "$PYTHON" -m pytest -q

printf '\nPipeline completed successfully.\n'
printf 'Processed data: data/processed/heart_disease_processed.csv\n'
printf 'Model: artifacts/model.joblib\n'
printf 'Metadata/comparison: artifacts/model_metadata.json, artifacts/model_comparison.csv\n'
printf 'EDA/model plots: reports/figures\n'
printf 'MLflow tracking database: mlflow.db\n'

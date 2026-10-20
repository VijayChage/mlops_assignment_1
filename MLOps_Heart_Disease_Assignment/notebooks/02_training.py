# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 02 — Model Training, Evaluation & MLflow
#
# This notebook uses the canonical `scripts/train.py` pipeline. It compares
# Logistic Regression and Random Forest using stratified 5-fold cross-validation,
# evaluates candidates, logs runs to MLflow, and saves the selected model.
#
# **Important:** Running the training cell retrains the models and overwrites
# the generated files under `artifacts/`. Do not run it unless you intend to
# refresh those artifacts.

# %%
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ARTIFACTS = ROOT / "artifacts"

print("Project:", ROOT)
print("Canonical training script:", ROOT / "scripts" / "train.py")

# %% [markdown]
# ## 1. Run the canonical training pipeline
#
# This is the same script used by the project command line and deployment
# workflow. It keeps preprocessing, cross-validation, model selection, and
# metadata generation consistent with the API.

# %%
run_training = False  # Change to True only when you intend to retrain.

if run_training:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py")],
        cwd=ROOT,
        check=True,
    )
else:
    print("Training skipped to preserve existing model artifacts.")
    print("Set run_training = True and rerun this cell to retrain.")

# %% [markdown]
# ## 2. Inspect the selected model metadata
#
# This reads the currently saved metadata. If training was skipped, it reports
# the existing model rather than creating a new one.

# %%
metadata_path = ARTIFACTS / "model_metadata.json"
if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    print(json.dumps(metadata, indent=2))
else:
    print("Model metadata not found. Run the training cell first.")

# %% [markdown]
# ## 3. Compare candidate model metrics
#
# The canonical script writes `model_comparison.csv` after a successful run.

# %%
import pandas as pd

comparison_path = ARTIFACTS / "model_comparison.csv"
if comparison_path.exists():
    comparison = pd.read_csv(comparison_path)
    print(comparison.to_string(index=False))
else:
    print("Model comparison file not found. Run training to generate it.")

# %% [markdown]
# ## 4. MLflow tracking
#
# The canonical training script records its experiment in `mlflow.db` under
# the `heart-disease-classification` experiment. From the project directory,
# launch the UI with:
#
# `mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000`
#
# Open port 5000 through your lab environment if it supports port forwarding.

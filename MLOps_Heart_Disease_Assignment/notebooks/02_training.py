# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 02 — Feature Engineering & Model Development
#
# Mirrors `scripts/train.py`: builds the preprocessing pipeline, tunes
# Logistic Regression and Random Forest with GridSearchCV (5-fold stratified CV,
# ROC-AUC selection), evaluates on a held-out test set, and logs everything to MLflow.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import joblib
import json
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay,
    accuracy_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from heartml.data import load_processed_dataframe, make_preprocessor

SEED = 42
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Load the processed dataset

# %%
df = load_processed_dataframe()   # reads data/processed/heart_disease_processed.csv
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# %% [markdown]
# ## 2. Preprocessing pipeline
# Same transformer as production: numeric → median impute + standard scale;
# categorical → most-frequent impute + one-hot. Lives inside the sklearn Pipeline
# so inference applies *exactly* the same transformations.

# %%
preprocessor = make_preprocessor()
preprocessor

# %% [markdown]
# ## 3. Hyperparameter tuning — GridSearchCV

# %%
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

tuned_specs = {
    "logistic_regression": {
        "model": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "grid": {"model__C": [0.01, 0.1, 1.0, 10.0],
                 "model__solver": ["lbfgs", "liblinear"]},
    },
    "random_forest": {
        "model": RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
        "grid": {"model__n_estimators": [100, 250],
                 "model__max_depth": [None, 8, 12],
                 "model__min_samples_leaf": [1, 2, 4]},
    },
}

# %%
best_models = {}
for name, spec in tuned_specs.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", spec["model"])])
    search = GridSearchCV(pipe, spec["grid"], cv=cv, scoring="roc_auc", n_jobs=1, refit=True)
    search.fit(X_train, y_train)
    best_models[name] = search
    print(f"{name}: best params={search.best_params_}  CV ROC-AUC={search.best_score_:.4f}")

# %% [markdown]
# ## 4. MLflow experiment tracking
# Every candidate is logged with its tuned params, CV score, test metrics, and plots.

# %%
mlflow.set_experiment("heart_disease_notebook")
results = []

with mlflow.start_run(run_name="notebook_model_comparison") as parent:
    for name, search in best_models.items():
        with mlflow.start_run(run_name=name, nested=True):
            model = search.best_estimator_
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }
            mlflow.log_params({f"tuned_{k.replace('model__', '')}": str(v)
                               for k, v in search.best_params_.items()})
            mlflow.log_param("model_name", name)
            mlflow.log_metric("cv_roc_auc_mean", search.best_score_)
            mlflow.log_metrics(metrics)

            # Plots as artifacts
            fig, ax = plt.subplots(figsize=(4, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            fig.savefig(FIGURES / f"nb_confusion_{name}.png", dpi=150)
            plt.close(fig)
            RocCurveDisplay.from_predictions(y_test, y_proba).plot_
            fig = plt.gcf(); fig.savefig(FIGURES / f"nb_roc_{name}.png", dpi=150); plt.close(fig)
            mlflow.log_artifact(str(FIGURES / f"nb_confusion_{name}.png"))
            mlflow.log_artifact(str(FIGURES / f"nb_roc_{name}.png"))

            results.append({"model": name, "cv_roc_auc": search.best_score_, **metrics})

# %% [markdown]
# ## 5. Model selection

# %%
results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
results_df

# %%
winner_name = results_df.iloc[0]["model"]
winner = best_models[winner_name].best_estimator_
print(f"Selected: {winner_name} (test ROC-AUC {results_df.iloc[0]['roc_auc']:.4f})")

# %% [markdown]
# ## 6. Save the final model + metadata

# %%
model_path = ARTIFACTS / "model.joblib"
joblib.dump(winner, model_path)

metadata = {
    "model_name": winner_name,
    "best_params": {k.replace("model__", ""): str(v)
                    for k, v in best_models[winner_name].best_params_.items()},
    "test_metrics": {k: float(v) for k, v in results_df.iloc[0].items() if k != "model"},
    "random_seed": SEED,
    "training_data": "data/processed/heart_disease_processed.csv",
}
(ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
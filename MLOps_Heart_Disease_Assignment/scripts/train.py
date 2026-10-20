from pathlib import Path
import sys
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score,
                              classification_report, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.heartml.data import load_dataset, make_preprocessor
from src.heartml.schema import FEATURES, TARGET
DATA = ROOT / "data/processed/heart_disease_processed.csv"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "reports/figures"
SEED = 42


def main():
    if not DATA.exists():
        raise SystemExit("Dataset not found. Run first: python scripts/download_data.py")
    ARTIFACTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA)
    X, y = df[FEATURES], df[TARGET]
    if y.value_counts().min() < 5:
        raise SystemExit("At least five rows per target class are required for 5-fold stratified CV.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
    }
    db_path = (ROOT / "mlflow.db").resolve().as_posix()
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("heart-disease-classification")
    results, fitted = [], []
    scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "roc_auc": "roc_auc"}
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        print(f"Training candidate: {name}", flush=True)
        cv_result = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1,
                                   error_score="raise")
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            f"cv_{metric}_mean": float(np.mean(cv_result[f"test_{metric}"]))
            for metric in scoring
        }
        metrics.update({
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_precision": float(precision_score(y_test, pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
        })
        report_path = ARTIFACTS / f"{name}_classification_report.json"
        report_path.write_text(json.dumps(classification_report(y_test, pred, output_dict=True,
            zero_division=0), indent=2), encoding="utf-8")
        model_path = ARTIFACTS / f"candidate_{name}.joblib"
        joblib.dump(pipeline, model_path)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params({"model_name": name, "random_seed": SEED, "cv_folds": 5,
                               "test_size": 0.2, "selection_metric": "cv_roc_auc_mean"})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(report_path), artifact_path="reports")
            mlflow.log_artifact(str(model_path), artifact_path="candidate_model")
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
            ax.set_title(f"{name}: test confusion matrix"); fig.tight_layout()
            path = FIGURES / f"{name}_confusion_matrix.png"
            fig.savefig(path, dpi=150); plt.close(fig); mlflow.log_artifact(str(path), artifact_path="plots")
            fig, ax = plt.subplots(figsize=(5, 4))
            RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
            ax.set_title(f"{name}: test ROC curve"); fig.tight_layout()
            path = FIGURES / f"{name}_roc_curve.png"
            fig.savefig(path, dpi=150); plt.close(fig); mlflow.log_artifact(str(path), artifact_path="plots")
            results.append({"model": name, **metrics, "run_id": run.info.run_id})
            fitted.append((metrics["cv_roc_auc_mean"], name, pipeline, metrics, run.info.run_id))
        print(f"{name} metrics: {json.dumps(metrics)}", flush=True)
    best_score, best_name, best_pipeline, best_metrics, best_run_id = max(fitted, key=lambda item: item[0])
    final_path = ARTIFACTS / "model.joblib"
    joblib.dump(best_pipeline, final_path)
    metadata = {"selected_model": best_name, "selection_metric": "mean 5-fold CV ROC-AUC",
        "selected_cv_roc_auc": best_score, "best_run_id": best_run_id, "features": FEATURES,
        "target_definition": "0 = source target 0; 1 = source target > 0", "random_seed": SEED,
        "test_metrics": {key: value for key, value in best_metrics.items() if key.startswith("test_")}}
    (ARTIFACTS / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    # Log the selected deployable artifact in a separate MLflow run, avoiding fragile sklearn flavor serialization.
    with mlflow.start_run(run_name="selected-final-model"):
        mlflow.log_params({"selected_model": best_name, "source_run_id": best_run_id})
        mlflow.log_metric("selected_cv_roc_auc", float(best_score))
        mlflow.log_artifact(str(final_path), artifact_path="final_model")
        mlflow.log_artifact(str(ARTIFACTS / "model_metadata.json"), artifact_path="metadata")
        mlflow.log_artifact(str(ARTIFACTS / "model_comparison.csv"), artifact_path="comparison")
    print("\nTraining complete. Selected model:")
    print(json.dumps(metadata, indent=2))
    print(f"Model saved to: {final_path}")
    print(f"MLflow tracking database: {ROOT / 'mlflow.db'}")

if __name__ == "__main__": main()

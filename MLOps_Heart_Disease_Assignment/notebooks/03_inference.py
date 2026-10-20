# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03 — Inference
#
# Three modes, all using the same committed `artifacts/model.joblib`:
# 1. Single prediction from a raw JSON request (like the API does).
# 2. Batch predictions over the held-out style data.
# 3. Live call against the FastAPI `/predict` endpoint (Docker/K8s).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import json
import joblib
import pandas as pd
import requests

from heartml.data import validate_features

model = joblib.load(ROOT / "artifacts" / "model.joblib")
metadata = json.loads((ROOT / "artifacts" / "model_metadata.json").read_text())
print("Loaded model:", metadata["model_name"], "| test metrics:", metadata["test_metrics"])

# %% [markdown]
# ## 1. Single prediction — sample request

# %%
request = json.loads((ROOT / "sample_request.json").read_text())
features = validate_features(request)      # same validation the API performs
X = pd.DataFrame([features])
proba = model.predict_proba(X)[0, 1]
prediction = int(proba >= 0.5)
print(f"Prediction: {prediction} | Confidence (P(disease)): {proba:.4f}")

# %% [markdown]
# ## 2. Batch inference over the processed dataset

# %%
df = pd.read_csv(ROOT / "data" / "processed" / "heart_disease_processed.csv")
X_all = df.drop(columns=["target"])
probas = model.predict_proba(X_all)[:, 1]
preds = (probas >= 0.5).astype(int)

out = X_all.copy()
out["prediction"] = preds
out["confidence"] = probas.round(4)
out["actual"] = df["target"]
out["correct"] = out["prediction"] == out["actual"]
out.head(10)

# %%
print(f"Batch accuracy: {out['correct'].mean():.4f} on {len(out)} rows")
out.to_csv(ROOT / "reports" / "batch_inference_results.csv", index=False)
print("Saved reports/batch_inference_results.csv")

# %% [markdown]
# ## 3. Live inference against the deployed API
# Start locally with `make api` (or `docker compose up`), then run this cell.
# Change BASE_URL to your NodePort / LoadBalancer URL when deployed to K8s.

# %%
BASE_URL = "http://localhost:8000"   # e.g. "http://192.168.49.2:30080" for Minikube NodePort

response = requests.post(
    f"{BASE_URL}/predict", json=request, timeout=10
)
print("Status:", response.status_code)
print("Body:", json.dumps(response.json(), indent=2))

# %%
health = requests.get(f"{BASE_URL}/health", timeout=10)
print("Health:", health.status_code, health.json())

# %% [markdown]
# ### Notes
# - `validate_features` guarantees the JSON has exactly the 13 expected features
#   with valid types/ranges — the same guard as the API.
# - Confidence is the model's positive-class probability, **not** a calibrated
#   clinical risk (educational prototype only).
# - After running section 3, hit `/metrics` and confirm `api_requests_total`
#   increased — that's your monitoring evidence.
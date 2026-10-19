# MLOps Assignment 01 — Heart Disease UCI

**Course:** Machine Learning Operations (MLOps), AIMLCZG523  
**Scope:** data acquisition, EDA, feature preprocessing, model comparison, MLflow experiment tracking, automated tests/CI, FastAPI inference, Docker, local Kubernetes, and Prometheus metrics/logging.

> **Educational prototype only.** This classifier is not a medical device and must not be used for diagnosis or treatment. Its positive-class probability is not a calibrated clinical risk.

## 1. Project structure

- `scripts/download_data.py` — downloads UCI Heart Disease dataset ID 45.
- `scripts/eda.py` — creates summary statistics and visualizations.
- `scripts/train.py` — trains Logistic Regression and Random Forest, performs stratified 5-fold CV, evaluates a held-out test set, logs runs/artifacts to MLflow, and saves the selected model.
- `src/heartml/data.py` — data validation, preprocessing, and feature validation.
- `src/heartml/api.py` — FastAPI `/predict`, `/health`, `/schema`, and `/metrics` endpoints; request logs and Prometheus metrics.
- `tests/` — pytest unit/smoke tests.
- `.github/workflows/ci.yml` — GitHub Actions lint, tests, download, EDA, training, and artifact upload.
- `Dockerfile`, `docker-compose.yml` — container build/run.
- `k8s/` — two-replica Kubernetes Deployment, NodePort Service, and Prometheus scrape example.
- `report/assignment_report.docx` — report template. Fill in your identity, repository URL, actual metrics, observations, screenshots, and deployment evidence before submission.

## 2. Clean setup — Windows PowerShell

Open PowerShell in the extracted project root (the directory containing `requirements.txt`). Python 3.11 is recommended.

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use a new PowerShell session and follow your institution's activation policy; do not delete or reset Docker data to solve a Python environment issue.

## 3. Clean setup — Linux/Ubuntu student environment

Run from the project root. Ensure Python 3.11 and `python3-venv` are available.

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If your student environment does not permit Docker or Kubernetes, run Sections 4–6 first and use the environment's approved container/cluster workflow for Section 8. Do not assume a local Docker Desktop context exists on Linux.

## 4. Run the full local ML pipeline

### Windows PowerShell — one command

With `.venv` activated and the current directory at the project root:

```powershell
.\scripts\run_pipeline.ps1
```

### Linux/Ubuntu — one command

```bash
bash scripts/run_pipeline.sh
```

Both scripts stop on the first failed step and run, in order:

1. Dataset download.
2. EDA summaries and plots.
3. Model comparison, cross-validation, held-out metrics, MLflow logging, and final model packaging.
4. Ruff checks.
5. Unit tests.

Alternatively, execute each step individually:

```text
python scripts/download_data.py
python scripts/eda.py
python scripts/train.py
ruff check .
python -m pytest -q
```

The downloader needs internet access. It intentionally stops with an error if the UCI source is unavailable; it does not fabricate data. The original UCI severity target is binarized as `0 -> 0` and `1–4 -> 1`. Missing feature values are imputed inside the scikit-learn pipeline to avoid fitting preprocessing before data splitting.

### Expected outputs

- `data/raw/heart_disease.csv`
- `reports/figures/class_balance.png`
- `reports/figures/feature_histograms.png`
- `reports/figures/correlation_heatmap.png`
- model ROC/confusion-matrix plots and `reports/figures/eda_summary.json`
- `artifacts/model.joblib`
- `artifacts/model_metadata.json`
- `artifacts/model_comparison.csv`
- candidate classification reports and candidate `.joblib` models in `artifacts/`
- `mlflow.db` — local MLflow SQLite tracking store

## 5. Experiment tracking — MLflow UI

Training creates/updates the local SQLite database. In a second terminal from the project root, activate `.venv` and run:

```powershell
mlflow ui --backend-store-uri sqlite:///./mlflow.db --host 127.0.0.1 --port 5000
```

On Linux, use the same command after activating `.venv`. Open `http://127.0.0.1:5000`. The two candidate runs log parameters, cross-validation and test metrics, classification reports, model files, and plots; a final run logs the selected deployable model and metadata.

## 6. Run and test the API locally

Training must have completed successfully first.

```powershell
uvicorn src.heartml.api:app --host 127.0.0.1 --port 8000
```

The command is the same on Linux. Open `http://127.0.0.1:8000/docs` and test `POST /predict` using the JSON in `sample_request.json`. The API also exposes:

- `GET /health` — confirms model loaded.
- `GET /schema` — input feature names.
- `GET /metrics` — Prometheus counters and latency histogram.

PowerShell smoke tests (run in a second terminal):

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
(Invoke-WebRequest http://127.0.0.1:8000/metrics).Content
Get-Content .\sample_request.json -Raw | Set-Content .\request-body.json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -InFile .\request-body.json
```

For Linux, use `curl`:

```bash
curl -sS http://127.0.0.1:8000/health
curl -sS http://127.0.0.1:8000/metrics
curl -sS -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' --data @sample_request.json
```

## 7. Docker containerization

Docker Engine/Desktop must be running, and the Docker CLI must connect to the correct engine.

```powershell
docker version
docker build -t heart-disease-api:local .
docker run --rm --name heart-disease-api-local -p 8000:8000 heart-disease-api:local
```

If port 8000 is already used by a local API or another container, stop that process or use a different host port such as `-p 8001:8000`, then use `http://localhost:8001`.

In a second terminal, test `/health`, `/metrics`, and `/docs`. The current Dockerfile downloads the dataset and trains during image build, so a network connection is required at build time. This is suitable for the coursework demo; a production pipeline would train in CI and package a versioned model artifact into the image.

Alternative using Compose:

```powershell
docker compose up --build
```

Stop the foreground process with `Ctrl+C`. If a detached Compose service was started, use `docker compose down`.

## 8. Local Kubernetes deployment — Docker Desktop (verified workflow)

The manifests were successfully applied to Docker Desktop Kubernetes with two replicas and a NodePort service. This route is appropriate if your student environment provides Docker Desktop Kubernetes. If the dedicated student environment is a remote Linux virtual lab, first check with its instructions whether Kubernetes is preconfigured; do not run Docker Desktop-specific context commands there.

### 8.1 Verify the cluster and image

```powershell
kubectl config get-contexts
kubectl config current-context
kubectl cluster-info
kubectl get nodes

docker context show
docker images heart-disease-api
```

On the tested Docker Desktop setup, the context was `docker-desktop`, the node was `Ready`, and the image `heart-disease-api:local` was present. The local cluster could use this image without pulling from a registry.

### 8.2 Apply the manifests

```powershell
kubectl apply -f .\k8s\deployment.yaml
kubectl apply -f .\k8s\service.yaml
kubectl rollout status deployment/heart-disease-api --timeout=180s
kubectl get deployments,pods,services
```

Expected deployment readiness: `2/2`; both pods should be `Running` with no restarts. The service should show `8000:30080/TCP`.

### 8.3 Test the deployed API

```powershell
Invoke-RestMethod http://localhost:30080/health
(Invoke-WebRequest http://localhost:30080/metrics).Content
```

Open `http://localhost:30080/docs` and test `POST /predict` with `sample_request.json`. If the NodePort is unavailable, inspect `kubectl get pods,svc` and `kubectl logs deployment/heart-disease-api`; do not delete the cluster or reset Docker data as an initial troubleshooting step.

The file `k8s/prometheus-scrape.yml` is an example scrape job to merge into an existing Prometheus configuration; it is not a complete Prometheus deployment and should not be applied with `kubectl apply` by itself.

### 8.4 Cleanup (only after capturing evidence)

```powershell
kubectl delete -f .\k8s\service.yaml
kubectl delete -f .\k8s\deployment.yaml
```

Do not clean up until you have captured required screenshots and tested the endpoints.

## 9. CI/CD — GitHub Actions

Push the project to your GitHub repository. The workflow at `.github/workflows/ci.yml` runs on push, pull request, and manual dispatch. It installs dependencies, runs Ruff and pytest, downloads data, creates EDA outputs, trains/logs models, and uploads run artifacts. Inspect the Actions run and save a screenshot of a successful execution. A failing lint/test/training step fails the workflow.

## 10. Monitoring, logging, and report deliverables

The API logs each request as a JSON line and exposes Prometheus counters and a latency histogram at `/metrics`. A complete Prometheus + Grafana dashboard is not bundled; either show the metrics/log output for the assignment's simple monitoring requirement or deploy Prometheus/Grafana if explicitly required by your instructor.

Before submitting:

1. Add your actual name/student ID, date, and repository URL to `report/assignment_report.docx`.
2. Replace template placeholders with actual metric values and observations from your run. Do not invent performance results.
3. Include real screenshots in `screenshots/`: EDA plots, MLflow runs, successful CI workflow, Docker build/run, Swagger prediction, health/metrics, and Kubernetes `2/2` replicas.
4. Render/export a final report of at least 10 pages if required by your instructor.
5. Record the requested short video showing data download, EDA, training/MLflow, tests/CI, Docker, Kubernetes, and API/metrics. Do not claim a public API URL; this project uses local deployment unless you separately deploy it to a public cloud.
6. Commit the code and artifacts appropriate for submission. Do not commit secrets or the local `.venv`; `mlflow.db` is ignored by `.gitignore`.

## Model-selection notes

The pipeline compares class-weighted Logistic Regression and Random Forest. It uses a stratified 80/20 train/test split and selects the model using mean stratified 5-fold cross-validation ROC-AUC on the training partition. It also reports accuracy, precision, recall, and ROC-AUC on the held-out test set. The full preprocessing pipeline is fitted within each cross-validation fold.

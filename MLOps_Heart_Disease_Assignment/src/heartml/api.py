from pathlib import Path
import json
import logging
import os
import time
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ROOT / "artifacts" / "model.joblib")))
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("heartml-api")
REQUESTS = Counter("api_requests_total", "API requests", ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "API request latency", ["endpoint"])
PREDICTIONS = Counter("model_predictions_total", "Predictions returned", ["prediction"])
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if MODEL_PATH.is_file():
        model = joblib.load(MODEL_PATH)
        logger.info(json.dumps({"event": "model_loaded", "path": str(MODEL_PATH)}))
    else:
        model = None
        logger.error(json.dumps({"event": "model_missing", "path": str(MODEL_PATH)}))
    yield

app = FastAPI(title="Heart Disease Classifier API", version="1.0.1",
              description="Educational demo only; not for clinical use.", lifespan=lifespan)

class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    age: float
    sex: str | float
    cp: str | float
    trestbps: float
    chol: float
    fbs: str | float
    restecg: str | float
    thalach: float
    exang: str | float
    oldpeak: float
    slope: str | float
    ca: str | float
    thal: str | float

@app.middleware("http")
async def log_requests(request: Request, call_next):
    started = time.perf_counter(); status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - started
        endpoint = request.url.path
        REQUESTS.labels(request.method, endpoint, str(status)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        logger.info(json.dumps({"event": "http_request", "method": request.method,
            "path": endpoint, "status": status, "latency_seconds": round(elapsed, 6)}))

@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ok", "model_loaded": True}

@app.get("/schema")
def schema():
    return {"required_features": list(PatientFeatures.model_fields),
            "target": "binary educational label; 0=absent, 1=present in source data"}

@app.post("/predict")
def predict(patient: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Run training first.")
    try:
        from src.heartml.data import validate_features
        row = validate_features(patient.model_dump())
        X = pd.DataFrame([row])
        prediction = int(model.predict(X)[0])
        probabilities = model.predict_proba(X)[0]
        positive_probability = float(probabilities[list(model.classes_).index(1)])
        confidence = float(max(probabilities))
        PREDICTIONS.labels(str(prediction)).inc()
        return {"prediction": prediction,
            "label": "disease_present" if prediction == 1 else "disease_absent",
            "confidence": round(confidence, 4),
            "positive_class_probability": round(positive_probability, 4),
            "disclaimer": "Educational output only; not a diagnosis or calibrated clinical risk."}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

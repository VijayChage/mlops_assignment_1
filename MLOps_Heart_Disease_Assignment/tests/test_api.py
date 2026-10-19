import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.heartml.data import make_preprocessor
from src.heartml.schema import FEATURES, NUMERIC_FEATURES
from src.heartml import api


def make_fitted_model():
    rng = np.random.default_rng(19)
    rows = []
    labels = []
    for i in range(100):
        row = {}
        for col in FEATURES:
            row[col] = float(rng.normal(50, 10)) if col in NUMERIC_FEATURES else str(rng.integers(0, 3))
        rows.append(row)
        labels.append(i % 2)
    pipe = Pipeline([("preprocessor", make_preprocessor()),
                     ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(pd.DataFrame(rows, columns=FEATURES), np.asarray(labels))
    return pipe


def test_health_and_prediction():
    api.model = make_fitted_model()
    client = TestClient(api.app)
    assert client.get("/health").json()["model_loaded"] is True
    payload = {"age": 63, "sex": 1, "cp": 2, "trestbps": 145, "chol": 233,
               "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
               "oldpeak": 2.3, "slope": 1, "ca": 0, "thal": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in (0, 1)
    assert 0 <= body["positive_class_probability"] <= 1
    assert client.get("/metrics").status_code == 200

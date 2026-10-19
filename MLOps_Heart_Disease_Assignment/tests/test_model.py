import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.heartml.data import make_preprocessor
from src.heartml.schema import FEATURES, NUMERIC_FEATURES


def test_pipeline_fit_predict_smoke():
    rng = np.random.default_rng(7)
    rows = []
    for i in range(100):
        row = {}
        for col in FEATURES:
            row[col] = float(rng.normal(50, 10)) if col in NUMERIC_FEATURES else str(rng.integers(0, 3))
        rows.append(row)
    X = pd.DataFrame(rows, columns=FEATURES)
    y = np.array([0, 1] * 50)
    pipe = Pipeline([("preprocessor", make_preprocessor()),
                     ("model", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    assert pipe.predict(X.iloc[:3]).shape == (3,)
    assert pipe.predict_proba(X.iloc[:3]).shape == (3, 2)

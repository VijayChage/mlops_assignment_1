FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = [name for name in FEATURES if name not in NUMERIC_FEATURES]
TARGET = "target"
ALIASES = {
    "thalch": "thalach",
    "restingbp": "trestbps",
    "resting_blood_pressure": "trestbps",
    "fastingbs": "fbs",
    "fasting_blood_sugar": "fbs",
    "restingecg": "restecg",
    "maxheartrate": "thalach",
}

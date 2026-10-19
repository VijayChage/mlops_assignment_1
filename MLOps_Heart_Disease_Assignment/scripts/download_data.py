from pathlib import Path
import sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.heartml.data import normalize_column_names

OUT = ROOT / "data" / "raw" / "heart_disease.csv"


def main():
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise SystemExit("Missing dependency ucimlrepo. Run: python -m pip install -r requirements.txt") from exc
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading UCI Heart Disease dataset (repository dataset ID 45)...")
    try:
        dataset = fetch_ucirepo(id=45)
        features = dataset.data.features.copy()
        targets = dataset.data.targets.copy()
    except Exception as exc:
        raise SystemExit(
            "UCI download failed. Check internet access and retry: python scripts/download_data.py\n"
            f"Underlying error: {exc}"
        ) from exc
    features.columns = normalize_column_names(features.columns)
    target_name = next((c for c in targets.columns if str(c).strip().lower() in {"num", "target"}), targets.columns[0])
    frame = features.copy()
    frame["target"] = pd.to_numeric(targets[target_name], errors="coerce").to_numpy()
    # UCI uses thalch in some versions; normalize_column_names maps it to thalach.
    aliases = {"target": "target"}
    frame = frame.rename(columns=aliases)
    expected = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                "exang", "oldpeak", "slope", "ca", "thal", "target"]
    missing = set(expected) - set(frame.columns)
    if missing:
        raise SystemExit(f"UCI schema changed. Missing {sorted(missing)}; received columns: {list(frame.columns)}")
    frame = frame[expected]
    frame.to_csv(OUT, index=False)
    print(f"Saved {len(frame)} rows and {len(frame.columns)} columns to: {OUT}")
    print("Raw target values:", sorted(frame["target"].dropna().unique().tolist()))
    print("The training loader converts target 0 to class 0 and target values >0 to class 1.")

if __name__ == "__main__":
    main()

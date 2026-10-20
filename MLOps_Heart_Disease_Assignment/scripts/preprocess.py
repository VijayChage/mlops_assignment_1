from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.heartml.data import load_dataset
from src.heartml.schema import FEATURES, TARGET

RAW_DATA = ROOT / "data" / "raw" / "heart_disease.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DATA = PROCESSED_DIR / "heart_disease_processed.csv"


def main():
    if not RAW_DATA.is_file():
        raise SystemExit(
            f"Raw dataset not found: {RAW_DATA}\n"
            "Run: ./.venv/bin/python scripts/download_data.py"
        )

    print(f"Reading raw dataset: {RAW_DATA}")
    raw = pd.read_csv(RAW_DATA)
    print(f"Raw shape: {raw.shape}")

    # Standardize the schema, validate required columns,
    # normalize categories and binarize the target.
    processed = load_dataset(RAW_DATA)

    if processed.empty:
        raise SystemExit("Preprocessing failed: no usable rows remain.")

    if not set(processed[TARGET].unique()).issubset({0, 1}):
        raise SystemExit("Preprocessing failed: target is not binary.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(PROCESSED_DATA, index=False)

    summary = {
        "raw_file": str(RAW_DATA.relative_to(ROOT)),
        "processed_file": str(PROCESSED_DATA.relative_to(ROOT)),
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(processed)),
        "feature_count": len(FEATURES),
        "target_column": TARGET,
        "target_distribution": {
            str(k): int(v)
            for k, v in processed[TARGET].value_counts().sort_index().items()
        },
        "missing_values_by_column": {
            str(k): int(v)
            for k, v in processed.isna().sum().items()
            if v > 0
        },
        "duplicate_rows": int(processed.duplicated().sum()),
    }

    summary_path = PROCESSED_DIR / "preprocessing_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nPreprocessing completed successfully.")
    print(json.dumps(summary, indent=2))
    print(f"\nProcessed dataset: {PROCESSED_DATA}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

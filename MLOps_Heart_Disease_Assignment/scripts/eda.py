from pathlib import Path
import sys
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.heartml.data import load_dataset
from src.heartml.schema import FEATURES, TARGET
DATA = ROOT / "data/processed/heart_disease_processed.csv"
OUT = ROOT / "reports/figures"


def main():
    if not DATA.exists():
        raise SystemExit("Dataset not found. Run first: python scripts/download_data.py")
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_dataset(DATA)
    df.describe(include="all").to_csv(OUT / "descriptive_statistics.csv")
    summary = {"rows": int(len(df)), "columns": int(df.shape[1]),
               "missing_values": {str(k): int(v) for k, v in df.isna().sum().items()},
               "target_balance": {str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()}}
    (OUT / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=TARGET, ax=ax)
    ax.set_title("Target balance (0 = absent, 1 = present)")
    fig.tight_layout(); fig.savefig(OUT / "class_balance.png", dpi=160); plt.close(fig)
    numeric = [c for c in FEATURES if pd.api.types.is_numeric_dtype(df[c])]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, numeric):
        sns.histplot(data=df, x=col, kde=True, ax=ax)
        ax.set_title(col)
    for ax in axes.flat[len(numeric):]: ax.axis("off")
    fig.suptitle("Numeric feature distributions")
    fig.tight_layout(); fig.savefig(OUT / "feature_histograms.png", dpi=160); plt.close(fig)
    numeric_df = df[FEATURES + [TARGET]].copy()
    for col in numeric_df.columns:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric_df.corr(numeric_only=True), cmap="vlag", center=0, ax=ax)
    ax.set_title("Numeric-coded feature correlation heatmap")
    fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=160); plt.close(fig)
    print(f"EDA complete. Outputs saved to {OUT}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()

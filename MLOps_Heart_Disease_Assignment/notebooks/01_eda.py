# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01 — Data Acquisition & Exploratory Data Analysis
#
# Uses the project dataset and canonical preprocessing/validation.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")  # safe for headless/CI runs; use "inline" for interactive work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.heartml.data import load_dataset

FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Download the dataset
# If the dataset is missing, run `python scripts/download_data.py` first.

# %%
DATA = ROOT / "data" / "processed" / "heart_disease_processed.csv"
if not DATA.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA}. Run `python scripts/download_data.py` first."
    )

df = load_dataset(DATA)
df.head()

# %% [markdown]
# ## 2. Validate the schema
# Uses the same load_dataset validation as the production training pipeline.

# %%
print(f"Loaded and validated: {df.shape[0]} rows x {df.shape[1]} columns")
df.dtypes

# %% [markdown]
# ## 3. Missing values & basic statistics

# %%
print("Missing values per column:")
print(df.isna().sum())
df.describe().T

# %% [markdown]
# ## 4. Class balance

# %%
counts = df["target"].value_counts().sort_index()
print(counts)
print(f"Positive rate: {df['target'].mean():.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
counts.plot(kind="bar", ax=ax, color=["#4c72b0", "#c44e52"])
ax.set_xticklabels(["No disease (0)", "Disease (1)"], rotation=0)
ax.set_ylabel("Count")
ax.set_title("Class balance — heart disease target")
fig.tight_layout()
fig.savefig(FIGURES / "nb_class_balance.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. Feature distributions (histograms)

# %%
feature_cols = [c for c in df.columns if c != "target"]
df[feature_cols].hist(bins=20, figsize=(14, 10), edgecolor="white")
plt.suptitle("Feature distributions")
plt.savefig(FIGURES / "nb_histograms.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Correlation heatmap

# %%
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", ax=ax)
ax.set_title("Correlation matrix — Heart Disease UCI")
fig.tight_layout()
fig.savefig(FIGURES / "nb_correlation_heatmap.png", dpi=150)
plt.show()

# %% [markdown]
# ## 7. Target-vs-feature relationships (boxplots for key numeric features)

# %%
key_numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
fig, axes = plt.subplots(1, len(key_numeric), figsize=(20, 4))
for col, ax in zip(key_numeric, axes):
    sns.boxplot(data=df, x="target", y=col, ax=ax)
    ax.set_title(f"{col} by target")
fig.tight_layout()
fig.savefig(FIGURES / "nb_target_boxplots.png", dpi=150)
plt.show()

# %% [markdown]
# ### EDA takeaways
# - Class balance: roughly balanced after binarization (~46% / ~54%).
# - Strongest correlates with disease: `cp`, `thalach`, `exang`, `oldpeak`.
# - `ca`/`thal` arrive as categorical codes with missing markers; handled by the
#   preprocessing pipeline (imputation + one-hot inside the sklearn Pipeline).
# - Features on very different scales → scaling required for Logistic Regression.

# %%
print("EDA notebook complete. Figures saved to reports/figures/.")
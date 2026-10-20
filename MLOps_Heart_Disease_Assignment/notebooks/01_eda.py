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
# Heart Disease UCI dataset (UCI ML Repository id 45, via `ucimlrepo`).
# Steps: download → validate → inspect → visualize (class balance, histograms, correlation heatmap).

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")  # safe for headless/CI runs; use "inline" for interactive work
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo

from heartml.data import validate_dataframe

FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Download the dataset
# `scripts/download_data.py` performs the same step and caches to `data/raw/`.
# The notebook fetches directly so every cell is self-contained.

# %%
heart = fetch_ucirepo(id=45)
df_raw = pd.concat([heart.data.features, heart.data.targets], axis=1)

# UCI's full target `num` ranges 0-4; binarize to presence/absence (0 vs >0).
if "num" in df_raw.columns:
    df_raw["target"] = (df_raw["num"] > 0).astype(int)
    df_raw = df_raw.drop(columns=["num"])

df_raw.head()

# %% [markdown]
# ## 2. Validate the schema
# Uses the same validation used by the pipeline and unit tests.

# %%
df = validate_dataframe(df_raw)
print(f"Validated: {df.shape[0]} rows x {df.shape[1]} columns")
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
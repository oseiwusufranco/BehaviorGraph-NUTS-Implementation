# Supplementary implementation: BehaviorGraph-NUTS
# Exported from BehaviorGraph_NUTS_Supplementary_Implementation_Clean.ipynb
# This script is provided as an optional plain-Python companion to the no-output notebook.


# ==============================================================================
# # Supplementary Implementation: BehaviorGraph-NUTS
# 
# This notebook provides the complete reproducibility implementation for **BehaviorGraph-NUTS: A Graph-Derived Bayesian Posterior Fusion Framework for Uncertainty-Aware Anti-Money Laundering Detection in IoT-Enabled Financial Systems**.
# 
# The notebook is distributed as supporting material without executed outputs. It contains the full pipeline for dataset loading, leakage-controlled partitioning, graph-derived behavioural feature construction, calibrated evidence modelling, deterministic fusion, NUTS-based Bayesian posterior fusion, calibration analysis, posterior diagnostics, alert-prioritization evaluation, five-seed robustness, statistical comparisons, runtime/memory transparency, and artifact export.
# 
# Before running the notebook, set `CSV_PATH` in Cell 2 or define the environment variable `BEHAVIORGRAPH_NUTS_CSV_PATH` to point to the HI-Small AML transaction file. Outputs are written to `OUTPUT_DIR`, which can also be overridden using `BEHAVIORGRAPH_NUTS_OUTPUT_DIR`.
# ==============================================================================


# %% Cell 1
# ============================================================
# CELL 1: Environment Setup and Imports
# ============================================================
# Supplementary implementation for:
# BehaviorGraph-NUTS: A Graph-Derived Bayesian Posterior Fusion Framework
# for Uncertainty-Aware Anti-Money Laundering Detection in IoT-Enabled Financial Systems.
#
# This notebook is intentionally distributed without executed outputs.
# Set INSTALL_MISSING_PACKAGES=0 in the runtime environment if package installation
# is not permitted by the submission/reproduction platform.

import os
import sys
import json
import math
import time
import gc
import random
import pickle
import warnings
import subprocess
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")

INSTALL_MISSING_PACKAGES = os.environ.get("INSTALL_MISSING_PACKAGES", "1") == "1"

def ensure_package(import_name, pip_name=None):
    """Install a package only when it is missing and installation is enabled."""
    if importlib.util.find_spec(import_name) is None:
        package = pip_name or import_name
        if not INSTALL_MISSING_PACKAGES:
            raise ImportError(
                f"Missing package '{package}'. Install it manually or set INSTALL_MISSING_PACKAGES=1."
            )
        print(f"Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for import_name, pip_name in [
    ("imblearn", "imbalanced-learn"),
    ("xgboost", "xgboost"),
    ("pymc", "pymc"),
    ("arviz", "arviz"),
    ("psutil", "psutil"),
]:
    ensure_package(import_name, pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil

from IPython.display import display
from scipy.special import logit, expit
from scipy.stats import ttest_rel, wilcoxon
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import IsolationForest
from imblearn.under_sampling import RandomUnderSampler

import xgboost as xgb
import pymc as pm
import arviz as az

pd.set_option("display.max_columns", 250)
pd.set_option("display.width", 200)

print("Python:", sys.version.replace("\n", " "))
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("Scikit-learn imported successfully.")
print("XGBoost:", xgb.__version__)
print("PyMC:", pm.__version__)
print("ArviZ:", az.__version__)



# %% Cell 2
# ============================================================
# CELL 2: Global Configuration and Reproducibility Controls
# ============================================================

FRAMEWORK_NAME = "BehaviorGraph-NUTS"
PAPER_TITLE = (
    "BehaviorGraph-NUTS: A Graph-Derived Bayesian Posterior Fusion Framework "
    "for Uncertainty-Aware Anti-Money Laundering Detection in IoT-Enabled Financial Systems"
)

SEED = 42
RANDOM_SEEDS = [42, 123, 202, 777, 999]

# ------------------------------------------------------------------
# Dataset and output paths
# ------------------------------------------------------------------
# For reproducibility, set BEHAVIORGRAPH_NUTS_CSV_PATH and
# BEHAVIORGRAPH_NUTS_OUTPUT_DIR in the runtime environment when the
# dataset is located somewhere other than the default Google Drive path.
CSV_PATH = os.environ.get(
    "BEHAVIORGRAPH_NUTS_CSV_PATH",
    "/content/drive/MyDrive/Dataset/AML Dataset/HI-Small_Trans.csv"
)

OUTPUT_DIR = os.environ.get(
    "BEHAVIORGRAPH_NUTS_OUTPUT_DIR",
    "/content/drive/MyDrive/Dataset/AML Dataset/BehaviorGraph_NUTS_Results_TUNED"
)

FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
for d in [OUTPUT_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# Use None for the full HI-Small transaction file.
ROW_LIMIT = None

# ------------------------------------------------------------------
# Partitioning
# ------------------------------------------------------------------
SPLIT_MODE = "stratified_random"   # Options: "stratified_random" or "chronological"
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

# Validation is split internally:
# - validation-fit: evidence calibration, deterministic fusion, and NUTS posterior fitting
# - validation-select: posterior score selection and threshold-policy tuning
VAL_SELECT_SIZE = 0.50

# ------------------------------------------------------------------
# Evidence-model tuning controls
# ------------------------------------------------------------------
RUN_FAST_XGB_TUNING = True
TUNING_MAX_ROWS = 650000
TUNING_NEG_POS_RATIO = 80

# ------------------------------------------------------------------
# NUTS posterior-fusion controls
# ------------------------------------------------------------------
NUTS_MAX_FUSION_ROWS = 60000
NUTS_DRAWS = 1000
NUTS_TUNE = 1500
NUTS_CHAINS = 4
NUTS_CORES = 1
NUTS_TARGET_ACCEPT = 0.97

# Empirical-Bayes anchoring uses deterministic logistic fusion estimates as weak priors.
USE_ANCHORED_NUTS_PRIOR = True
ANCHOR_ALPHA_SIGMA = 0.75
ANCHOR_BETA_SIGMA = 0.55

# ------------------------------------------------------------------
# Robustness controls
# ------------------------------------------------------------------
RUN_FIVE_SEED_DETERMINISTIC_EVAL = True
RUN_FIVE_SEED_NUTS = True

# ------------------------------------------------------------------
# Posterior score-selection controls
# ------------------------------------------------------------------
SCORE_SELECTION_MODE = "composite_tiebreak"
SCORE_AUPRC_TIE_TOL = 0.002
SCORE_BRIER_TIE_TOL = 2e-5
RANK_BLEND_VARIANTS_ENABLED = True
MAX_NUTS_FUSION_FEATURES = 8
MIN_SCREEN_ABS_COEF = 0.03

# ------------------------------------------------------------------
# Diagnostic controls
# ------------------------------------------------------------------
REPRESENTATIVE_PARAM_COUNT = 5
RUN_SUBSET_SIZE_CONVERGENCE_DIAGNOSTIC = True
SUBSET_CONVERGENCE_SIZES = [10000, 30000, 60000]
SUBSET_CONVERGENCE_MAX_SIZES = 3
RUN_SHORT_NUTS_SUBSET_DIAGNOSTIC = False
SHORT_NUTS_DRAWS = 300
SHORT_NUTS_TUNE = 500
SHORT_NUTS_CHAINS = 2

# ------------------------------------------------------------------
# Operational alert-policy controls
# ------------------------------------------------------------------
ALERT_RATES = [0.001, 0.005, 0.01, 0.02, 0.05]
MAX_MANUSCRIPT_ALERT_RATE = 0.01
EPS = 1e-7

def set_global_seed(seed: int = 42):
    """Set Python and NumPy seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)

set_global_seed(SEED)

print(PAPER_TITLE)
print("CSV path:", CSV_PATH)
print("Output directory:", OUTPUT_DIR)
print("Split mode:", SPLIT_MODE)
print("Primary seed:", SEED)
print("Five-seed list:", RANDOM_SEEDS)
print("Five-seed NUTS enabled:", RUN_FIVE_SEED_NUTS)



# %% Cell 3
# ============================================================
# CELL 3: Load Dataset
# ============================================================

read_kwargs = {}
if ROW_LIMIT is not None:
    read_kwargs["nrows"] = ROW_LIMIT

df_raw = pd.read_csv(CSV_PATH, **read_kwargs)

print("Dataset shape:", df_raw.shape)
print("Columns:", df_raw.columns.tolist())

if "Is Laundering" in df_raw.columns:
    label_counts = df_raw["Is Laundering"].value_counts().sort_index()
    label_rates = (df_raw["Is Laundering"].value_counts(normalize=True).sort_index() * 100).round(6)
    print("\\nLabel distribution:")
    print(label_counts)
    print("\\nLabel distribution (%):")
    print(label_rates)

display(df_raw.head())



# %% Cell 4
# ============================================================
# CELL 4: Canonical Column Mapping and Validation
# ============================================================

required_cols = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering"
]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df_raw.rename(columns={
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Is Laundering": "label"
}).copy()

for col in ["amount_received", "amount_paid"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
df = df.replace([np.inf, -np.inf], np.nan)

if not set(df["label"].unique()).issubset({0, 1}):
    raise ValueError("The label column must be binary after canonical mapping.")

print("Canonical dataset shape:", df.shape)
print("Positive class rate:", df["label"].mean())



# %% Cell 5
# ============================================================
# CELL 5: Transaction, Temporal, and Graph Node Features
# ============================================================

df["timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
df["day"] = df["timestamp"].dt.day.fillna(0).astype(int)
df["dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)
df["month"] = df["timestamp"].dt.month.fillna(0).astype(int)
df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5]).astype(int)

df["amount_diff"] = df["amount_paid"] - df["amount_received"]
df["abs_amount_diff"] = df["amount_diff"].abs()
df["amount_ratio"] = df["amount_paid"] / (df["amount_received"] + EPS)
df["log_amount_paid"] = np.log1p(df["amount_paid"].clip(lower=0))
df["log_amount_received"] = np.log1p(df["amount_received"].clip(lower=0))

df["same_bank"] = (df["from_bank"].astype(str) == df["to_bank"].astype(str)).astype(int)
df["same_account"] = (df["from_account"].astype(str) == df["to_account"].astype(str)).astype(int)
df["same_currency"] = (df["payment_currency"].astype(str) == df["receiving_currency"].astype(str)).astype(int)
df["cross_bank"] = 1 - df["same_bank"]
df["cross_currency"] = 1 - df["same_currency"]

df["sender_id"] = df["from_bank"].astype(str) + "_" + df["from_account"].astype(str)
df["receiver_id"] = df["to_bank"].astype(str) + "_" + df["to_account"].astype(str)
df["edge_id"] = df["sender_id"] + "->" + df["receiver_id"]

df = df.replace([np.inf, -np.inf], np.nan)
print("Feature-engineered dataset shape:", df.shape)
display(df.head())



# %% Cell 6
# ============================================================
# CELL 6: Leakage-Controlled Train/Validation/Test Split
# ============================================================

def make_splits(data: pd.DataFrame, split_mode: str = "stratified_random"):
    data = data.copy()
    if split_mode == "chronological":
        data = data.sort_values("timestamp").reset_index(drop=True)
        n = len(data)
        n_train = int(TRAIN_SIZE * n)
        n_val = int(VAL_SIZE * n)
        train_part = data.iloc[:n_train].copy()
        val_part = data.iloc[n_train:n_train + n_val].copy()
        test_part = data.iloc[n_train + n_val:].copy()
        for name, part in [("Train", train_part), ("Validation", val_part), ("Test", test_part)]:
            if part["label"].sum() == 0:
                raise ValueError(f"Chronological split produced zero laundering cases in {name}. Use stratified_random.")
        return train_part, val_part, test_part

    if split_mode == "stratified_random":
        train_part, temp_part = train_test_split(
            data, test_size=(1 - TRAIN_SIZE), random_state=SEED, stratify=data["label"]
        )
        relative_test = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
        val_part, test_part = train_test_split(
            temp_part, test_size=relative_test, random_state=SEED, stratify=temp_part["label"]
        )
        return train_part.copy(), val_part.copy(), test_part.copy()

    raise ValueError(f"Unknown split_mode: {split_mode}")

train_df, val_df, test_df = make_splits(df, SPLIT_MODE)

for name, part in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
    print(f"{name} shape: {part.shape}")
    print(part["label"].value_counts().sort_index())
    print((part["label"].value_counts(normalize=True).sort_index() * 100).round(6))
    print("-" * 60)



# %% Cell 7
# ============================================================
# CELL 7: Build Training-Only Graph-Derived Behavioural Profiles
# ============================================================

def build_train_only_behavior_graph_profiles(train_data: pd.DataFrame):
    sender_profile = train_data.groupby("sender_id").agg(
        sender_txn_count=("amount_paid", "count"),
        sender_total_paid=("amount_paid", "sum"),
        sender_mean_paid=("amount_paid", "mean"),
        sender_std_paid=("amount_paid", "std"),
        sender_max_paid=("amount_paid", "max"),
        sender_min_paid=("amount_paid", "min"),
        sender_unique_receivers=("receiver_id", "nunique"),
        sender_unique_edges=("edge_id", "nunique"),
        sender_unique_payment_formats=("payment_format", "nunique"),
        sender_unique_payment_currencies=("payment_currency", "nunique"),
        sender_cross_bank_rate=("cross_bank", "mean"),
        sender_cross_currency_rate=("cross_currency", "mean"),
        sender_night_rate=("is_night", "mean")
    ).reset_index()

    receiver_profile = train_data.groupby("receiver_id").agg(
        receiver_txn_count=("amount_received", "count"),
        receiver_total_received=("amount_received", "sum"),
        receiver_mean_received=("amount_received", "mean"),
        receiver_std_received=("amount_received", "std"),
        receiver_max_received=("amount_received", "max"),
        receiver_min_received=("amount_received", "min"),
        receiver_unique_senders=("sender_id", "nunique"),
        receiver_unique_edges=("edge_id", "nunique"),
        receiver_unique_payment_formats=("payment_format", "nunique"),
        receiver_unique_receiving_currencies=("receiving_currency", "nunique"),
        receiver_cross_bank_rate=("cross_bank", "mean"),
        receiver_cross_currency_rate=("cross_currency", "mean"),
        receiver_night_rate=("is_night", "mean")
    ).reset_index()

    sender_graph = train_data.groupby("sender_id").agg(
        out_degree=("receiver_id", "nunique"),
        out_txn_count=("receiver_id", "count"),
        out_total_amount=("amount_paid", "sum"),
        out_mean_amount=("amount_paid", "mean"),
        out_max_amount=("amount_paid", "max")
    ).reset_index()

    receiver_graph = train_data.groupby("receiver_id").agg(
        in_degree=("sender_id", "nunique"),
        in_txn_count=("sender_id", "count"),
        in_total_amount=("amount_received", "sum"),
        in_mean_amount=("amount_received", "mean"),
        in_max_amount=("amount_received", "max")
    ).reset_index()

    edge_profile = train_data.groupby("edge_id").agg(
        edge_txn_count=("amount_paid", "count"),
        edge_total_paid=("amount_paid", "sum"),
        edge_mean_paid=("amount_paid", "mean"),
        edge_std_paid=("amount_paid", "std"),
        edge_max_paid=("amount_paid", "max"),
        edge_cross_currency_rate=("cross_currency", "mean"),
        edge_night_rate=("is_night", "mean")
    ).reset_index()

    return sender_profile, receiver_profile, sender_graph, receiver_graph, edge_profile

sender_profile, receiver_profile, sender_graph, receiver_graph, edge_profile = build_train_only_behavior_graph_profiles(train_df)

print("Sender profiles:", sender_profile.shape)
print("Receiver profiles:", receiver_profile.shape)
print("Sender graph:", sender_graph.shape)
print("Receiver graph:", receiver_graph.shape)
print("Edge profiles:", edge_profile.shape)



# %% Cell 8
# ============================================================
# CELL 8: Merge Training-Derived Graph Behavioural Features
# ============================================================

def merge_behavior_graph_features(data, sender_profile, receiver_profile, sender_graph, receiver_graph, edge_profile):
    merged = data.copy()
    merged = merged.merge(sender_profile, on="sender_id", how="left")
    merged = merged.merge(receiver_profile, on="receiver_id", how="left")
    merged = merged.merge(sender_graph, on="sender_id", how="left")
    merged = merged.merge(receiver_graph, on="receiver_id", how="left")
    merged = merged.merge(edge_profile, on="edge_id", how="left")
    return merged

train_df_f = merge_behavior_graph_features(train_df, sender_profile, receiver_profile, sender_graph, receiver_graph, edge_profile)
val_df_f = merge_behavior_graph_features(val_df, sender_profile, receiver_profile, sender_graph, receiver_graph, edge_profile)
test_df_f = merge_behavior_graph_features(test_df, sender_profile, receiver_profile, sender_graph, receiver_graph, edge_profile)

profile_feature_cols = [
    c for c in train_df_f.columns
    if c.startswith(("sender_", "receiver_", "out_", "in_", "edge_"))
    and c not in ["sender_id", "receiver_id", "edge_id"]
]

fill_values = train_df_f[profile_feature_cols].median(numeric_only=True).to_dict()
for data in [train_df_f, val_df_f, test_df_f]:
    for col in profile_feature_cols:
        data[col] = data[col].fillna(fill_values.get(col, 0))
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)

print("Number of train-only graph behavioural features:", len(profile_feature_cols))
print(profile_feature_cols)



# %% Cell 9
# ============================================================
# CELL 9: Higher-Order Graph Risk Features
# ============================================================

def add_higher_order_graph_risk_features(data: pd.DataFrame):
    data = data.copy()
    eps = 1e-9

    data["degree_ratio"] = data["out_degree"] / (data["in_degree"] + 1)
    data["amount_flow_ratio"] = data["out_total_amount"] / (data["in_total_amount"] + 1)
    data["txn_flow_ratio"] = data["out_txn_count"] / (data["in_txn_count"] + 1)

    data["sender_receiver_degree_sum"] = data["out_degree"] + data["in_degree"]
    data["sender_receiver_txn_sum"] = data["sender_txn_count"] + data["receiver_txn_count"]
    data["sender_receiver_amount_sum"] = data["sender_total_paid"] + data["receiver_total_received"]

    data["amount_vs_sender_mean"] = data["amount_paid"] / (data["sender_mean_paid"] + eps)
    data["amount_vs_receiver_mean"] = data["amount_received"] / (data["receiver_mean_received"] + eps)
    data["amount_vs_edge_mean"] = data["amount_paid"] / (data["edge_mean_paid"] + eps)

    data["receiver_concentration"] = data["sender_unique_receivers"] / (data["sender_txn_count"] + 1)
    data["sender_concentration"] = data["receiver_unique_senders"] / (data["receiver_txn_count"] + 1)

    data["structural_risk_raw"] = (
        np.log1p(data["degree_ratio"].clip(lower=0)) +
        np.log1p(data["amount_flow_ratio"].clip(lower=0)) +
        np.log1p(data["txn_flow_ratio"].clip(lower=0)) +
        data["cross_bank"] +
        data["cross_currency"] +
        data["is_night"]
    )

    return data.replace([np.inf, -np.inf], np.nan).fillna(0)

train_df_f = add_higher_order_graph_risk_features(train_df_f)
val_df_f = add_higher_order_graph_risk_features(val_df_f)
test_df_f = add_higher_order_graph_risk_features(test_df_f)

graph_risk_cols = [
    "degree_ratio", "amount_flow_ratio", "txn_flow_ratio",
    "sender_receiver_degree_sum", "sender_receiver_txn_sum", "sender_receiver_amount_sum",
    "amount_vs_sender_mean", "amount_vs_receiver_mean", "amount_vs_edge_mean",
    "receiver_concentration", "sender_concentration", "structural_risk_raw"
]
print("Higher-order graph risk features added:", graph_risk_cols)



# %% Cell 10
# ============================================================
# CELL 10: Train-Only Categorical Encoding
# ============================================================

categorical_cols = ["receiving_currency", "payment_currency", "payment_format"]
category_maps = {}
category_frequency_maps = {}

for col in categorical_cols:
    train_values = train_df_f[col].astype(str)
    uniques = pd.Series(train_values.unique()).sort_values().tolist()
    category_maps[col] = {v: i for i, v in enumerate(uniques)}
    freq_map = train_values.value_counts(normalize=True).to_dict()
    category_frequency_maps[col] = freq_map

    for data in [train_df_f, val_df_f, test_df_f]:
        data[col + "_enc"] = data[col].astype(str).map(category_maps[col]).fillna(-1).astype(int)
        data[col + "_freq"] = data[col].astype(str).map(freq_map).fillna(0.0).astype(float)

encoded_cat_cols = [col + "_enc" for col in categorical_cols]
freq_cat_cols = [col + "_freq" for col in categorical_cols]
print("Encoded categorical columns:", encoded_cat_cols)
print("Frequency categorical columns:", freq_cat_cols)



# %% Cell 11
# ============================================================
# CELL 11: Define Feature Sets for Ablation and Evidence Models
# ============================================================

base_feature_cols = [
    "from_bank", "to_bank", "amount_received", "amount_paid",
    "amount_diff", "abs_amount_diff", "amount_ratio",
    "log_amount_paid", "log_amount_received",
    "hour", "day", "dayofweek", "month", "is_weekend", "is_night",
    "same_bank", "same_account", "same_currency", "cross_bank", "cross_currency"
] + encoded_cat_cols + freq_cat_cols

behavior_graph_feature_cols = profile_feature_cols + graph_risk_cols
all_feature_cols = base_feature_cols + behavior_graph_feature_cols

all_feature_cols = [
    c for c in all_feature_cols
    if c in train_df_f.columns and c in val_df_f.columns and c in test_df_f.columns
]
base_feature_cols = [c for c in base_feature_cols if c in all_feature_cols]
behavior_graph_feature_cols = [c for c in behavior_graph_feature_cols if c in all_feature_cols]

print("Base tabular feature count:", len(base_feature_cols))
print("Graph-derived behavioural/profile/risk feature count:", len(behavior_graph_feature_cols))
print("Total BehaviorGraph feature count:", len(all_feature_cols))



# %% Cell 12
# ============================================================
# CELL 12: Prepare Feature Matrices
# ============================================================

def prepare_matrices(feature_cols):
    X_train_raw = train_df_f[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_val_raw = val_df_f[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_test_raw = test_df_f[feature_cols].replace([np.inf, -np.inf], np.nan)

    imputer_local = SimpleImputer(strategy="median")
    X_train_local = pd.DataFrame(imputer_local.fit_transform(X_train_raw), columns=feature_cols, index=train_df_f.index)
    X_val_local = pd.DataFrame(imputer_local.transform(X_val_raw), columns=feature_cols, index=val_df_f.index)
    X_test_local = pd.DataFrame(imputer_local.transform(X_test_raw), columns=feature_cols, index=test_df_f.index)

    scaler_local = StandardScaler()
    X_train_scaled_local = scaler_local.fit_transform(X_train_local)
    X_val_scaled_local = scaler_local.transform(X_val_local)
    X_test_scaled_local = scaler_local.transform(X_test_local)

    return X_train_local, X_val_local, X_test_local, X_train_scaled_local, X_val_scaled_local, X_test_scaled_local, imputer_local, scaler_local

y_train = train_df_f["label"].astype(int).values
y_val = val_df_f["label"].astype(int).values
y_test = test_df_f["label"].astype(int).values

X_train_base, X_val_base, X_test_base, X_train_base_scaled, X_val_base_scaled, X_test_base_scaled, base_imputer, base_scaler = prepare_matrices(base_feature_cols)
X_train, X_val, X_test, X_train_scaled, X_val_scaled, X_test_scaled, imputer, scaler = prepare_matrices(all_feature_cols)

print("Base matrices:", X_train_base.shape, X_val_base.shape, X_test_base.shape)
print("BehaviorGraph matrices:", X_train.shape, X_val.shape, X_test.shape)
print("Train positive rate:", y_train.mean())



# %% Cell 13
# ============================================================
# CELL 13A: Validation-Fit / Validation-Selection Split
# ============================================================
# This prevents the same validation rows from doing every job.
# validation-fit: calibration, deterministic fusion fitting, and NUTS fitting.
# validation-select: score-variant selection and threshold selection.

val_positions = np.arange(len(y_val))
val_fit_pos, val_select_pos = train_test_split(
    val_positions,
    test_size=VAL_SELECT_SIZE,
    random_state=SEED,
    stratify=y_val
)

val_fit_pos = np.asarray(val_fit_pos)
val_select_pos = np.asarray(val_select_pos)

y_val_fit = y_val[val_fit_pos]
y_val_select = y_val[val_select_pos]

val_fit_df_f = val_df_f.iloc[val_fit_pos].copy()
val_select_df_f = val_df_f.iloc[val_select_pos].copy()

X_val_base_fit = X_val_base.iloc[val_fit_pos]
X_val_base_select = X_val_base.iloc[val_select_pos]
X_val_fit = X_val.iloc[val_fit_pos]
X_val_select = X_val.iloc[val_select_pos]
X_val_scaled_fit = X_val_scaled[val_fit_pos]
X_val_scaled_select = X_val_scaled[val_select_pos]

print("Validation-fit size:", len(y_val_fit), "positive rate:", y_val_fit.mean())
print("Validation-select size:", len(y_val_select), "positive rate:", y_val_select.mean())



# %% Cell 14
# ============================================================
# CELL 13: Evaluation Utilities
# ============================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), EPS, 1 - EPS)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_rows = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & ((y_prob < hi) if i < n_bins - 1 else (y_prob <= hi))
        if mask.sum() == 0:
            bin_rows.append({"bin": i, "lower": lo, "upper": hi, "count": 0, "confidence": np.nan, "empirical_rate": np.nan, "gap": np.nan})
            continue
        conf = y_prob[mask].mean()
        emp = y_true[mask].mean()
        gap = abs(emp - conf)
        ece += (mask.sum() / len(y_true)) * gap
        bin_rows.append({"bin": i, "lower": lo, "upper": hi, "count": int(mask.sum()), "confidence": conf, "empirical_rate": emp, "gap": gap})
    return float(ece), pd.DataFrame(bin_rows)

def adaptive_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob).astype(float), EPS, 1 - EPS)
    order = np.argsort(y_prob)
    splits = np.array_split(order, n_bins)
    ace = 0.0
    rows = []
    for i, idx in enumerate(splits):
        if len(idx) == 0:
            continue
        conf = y_prob[idx].mean()
        emp = y_true[idx].mean()
        gap = abs(emp - conf)
        ace += (len(idx) / len(y_true)) * gap
        rows.append({"bin": i, "count": len(idx), "min_score": y_prob[idx].min(), "max_score": y_prob[idx].max(), "confidence": conf, "empirical_rate": emp, "gap": gap})
    return float(ace), pd.DataFrame(rows)

def maximum_calibration_error(y_true, y_prob, n_bins=10):
    _, bins_df = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    gaps = bins_df["gap"].dropna()
    return float(gaps.max()) if len(gaps) else np.nan

def safe_auc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

def safe_auprc(y_true, y_prob):
    return average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan

def safe_log_loss(y_true, y_prob):
    y_prob = np.clip(y_prob, EPS, 1 - EPS)
    return log_loss(y_true, y_prob, labels=[0, 1])

def evaluate_predictions(model_name, split_name, y_true, y_prob, threshold=0.5, n_bins=10):
    y_prob = np.clip(np.asarray(y_prob).astype(float), EPS, 1 - EPS)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    ece, _ = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    ace, _ = adaptive_calibration_error(y_true, y_prob, n_bins=n_bins)
    mce = maximum_calibration_error(y_true, y_prob, n_bins=n_bins)
    return {
        "Model": model_name, "Split": split_name, "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "F2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Cohen Kappa": cohen_kappa_score(y_true, y_pred),
        "AUROC": safe_auc(y_true, y_prob),
        "AUPRC": safe_auprc(y_true, y_prob),
        "Brier": brier_score_loss(y_true, y_prob),
        "Log Loss": safe_log_loss(y_true, y_prob),
        "ECE": ece, "ACE": ace, "MCE": mce,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Alert Rate": float(y_pred.mean())
    }

def top_k_alert_metrics(y_true, score, model_name, k_values=None):
    if k_values is None:
        k_values = ALERT_RATES
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score).astype(float)
    n = len(y_true)
    total_pos = max(y_true.sum(), 1)
    order = np.argsort(-score)
    rows = []
    for k in k_values:
        alerts = max(1, int(round(k * n)))
        idx = order[:alerts]
        tp = int(y_true[idx].sum())
        fp = int(alerts - tp)
        precision = tp / alerts
        recall = tp / total_pos
        lift = precision / max(y_true.mean(), EPS)
        rows.append({"Model": model_name, "Alert Rate": k, "Alerts": alerts, "TP@k": tp, "FP@k": fp, "Precision@k": precision, "Recall@k": recall, "Lift@k": lift})
    return pd.DataFrame(rows)

def threshold_for_alert_rate(scores, alert_rate):
    return float(np.quantile(np.asarray(scores).astype(float), 1.0 - alert_rate))

def threshold_search_constrained(y_true, score, max_alert_rate=0.01, metric="F2"):
    thresholds = np.unique(np.quantile(score, np.linspace(0.50, 0.9999, 500)))
    rows = []
    for th in thresholds:
        row = evaluate_predictions("policy", "Validation", y_true, score, threshold=float(th))
        if row["Alert Rate"] <= max_alert_rate:
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No threshold satisfies the alert-rate constraint.")
    return out.sort_values(metric, ascending=False).iloc[0], out

print("Evaluation utilities ready.")



# %% Cell 15
# ============================================================
# CELL 14: Class Imbalance Summary
# ============================================================

imbalance_summary = pd.DataFrame({
    "Split": ["Train", "Validation", "Test"],
    "Samples": [len(y_train), len(y_val), len(y_test)],
    "Laundering Cases": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())],
    "Non-Laundering Cases": [int(len(y_train)-y_train.sum()), int(len(y_val)-y_val.sum()), int(len(y_test)-y_test.sum())],
    "Laundering Rate": [y_train.mean(), y_val.mean(), y_test.mean()],
    "Imbalance Ratio (Non-Laundering:Laundering)": [
        (len(y_train)-y_train.sum()) / max(y_train.sum(), 1),
        (len(y_val)-y_val.sum()) / max(y_val.sum(), 1),
        (len(y_test)-y_test.sum()) / max(y_test.sum(), 1)
    ]
})
display(imbalance_summary)
imbalance_summary.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_class_imbalance_summary.csv"), index=False)



# %% Cell 16
# ============================================================
# CELL 15: Tune and Train Strong Tabular XGBoost Baseline
# ============================================================

scale_pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

def make_xgb_tuning_subset(y, max_rows=TUNING_MAX_ROWS, neg_pos_ratio=TUNING_NEG_POS_RATIO, seed=42):
    """Use all positives and a controlled number of negatives for fast PR-AUC-oriented tuning."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    max_neg = min(len(neg_idx), max_rows - len(pos_idx), len(pos_idx) * neg_pos_ratio)
    max_neg = max(max_neg, min(len(neg_idx), max_rows - len(pos_idx)))
    if len(pos_idx) + max_neg > max_rows:
        max_neg = max_rows - len(pos_idx)
    neg_sub = rng.choice(neg_idx, size=max_neg, replace=False)
    idx = np.concatenate([pos_idx, neg_sub])
    rng.shuffle(idx)
    return idx

def xgb_candidate_params(scale_pos_weight, seed=42):
    base = dict(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
    candidates = [
        dict(n_estimators=700, max_depth=4, learning_rate=0.035, subsample=0.90, colsample_bytree=0.85, min_child_weight=2, reg_lambda=4.0, reg_alpha=0.25, max_delta_step=1),
        dict(n_estimators=900, max_depth=5, learning_rate=0.030, subsample=0.88, colsample_bytree=0.88, min_child_weight=3, reg_lambda=5.0, reg_alpha=0.50, max_delta_step=1),
        dict(n_estimators=1100, max_depth=4, learning_rate=0.025, subsample=0.92, colsample_bytree=0.80, min_child_weight=2, reg_lambda=7.0, reg_alpha=0.75, max_delta_step=2),
        dict(n_estimators=800, max_depth=6, learning_rate=0.030, subsample=0.85, colsample_bytree=0.85, min_child_weight=4, reg_lambda=8.0, reg_alpha=1.00, max_delta_step=1),
        dict(n_estimators=1200, max_depth=3, learning_rate=0.025, subsample=0.95, colsample_bytree=0.90, min_child_weight=1, reg_lambda=3.0, reg_alpha=0.25, max_delta_step=2),
        dict(n_estimators=1000, max_depth=5, learning_rate=0.020, subsample=0.90, colsample_bytree=0.75, min_child_weight=2, reg_lambda=10.0, reg_alpha=0.50, max_delta_step=2),
    ]
    return [{**base, **p} for p in candidates]

def fit_xgb_with_params(X_tr, y_tr, params):
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr)
    return model

def tune_xgb_by_val_auprc(X_tr_full, y_tr, X_val_metric, y_val_metric, label, seed=42):
    if not RUN_FAST_XGB_TUNING:
        params = xgb_candidate_params(scale_pos_weight, seed=seed)[1]
        model = fit_xgb_with_params(X_tr_full, y_tr, params)
        return model, params, pd.DataFrame([{"candidate": 0, "AUPRC": np.nan, "AUROC": np.nan, "label": label}])

    tune_idx = make_xgb_tuning_subset(y_tr, seed=seed)
    X_tune = X_tr_full.iloc[tune_idx] if hasattr(X_tr_full, "iloc") else X_tr_full[tune_idx]
    y_tune = y_tr[tune_idx]

    rows = []
    best_model = None
    best_params = None
    best_score = -np.inf

    for i, params in enumerate(xgb_candidate_params(scale_pos_weight, seed=seed)):
        model = fit_xgb_with_params(X_tune, y_tune, params)
        val_score = model.predict_proba(X_val_metric)[:, 1]
        auprc = safe_auprc(y_val_metric, val_score)
        auroc = safe_auc(y_val_metric, val_score)
        top1 = top_k_alert_metrics(y_val_metric, val_score, label, k_values=[0.01]).iloc[0]["Recall@k"]
        rows.append({"label": label, "candidate": i, "AUPRC": auprc, "AUROC": auroc, "Recall@1%": top1, **{k: params[k] for k in params if k not in ["objective","eval_metric","tree_method","random_state","n_jobs"]}})
        # Primary: AUPRC; tie breaker: Recall@1%
        score = auprc + 0.02 * top1
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model

    tuning_df = pd.DataFrame(rows).sort_values(["AUPRC", "Recall@1%"], ascending=False)
    print(f"Best tuning candidate for {label}:")
    display(tuning_df.head(3).round(6))

    # Refit on full training set with the selected parameters.
    final_model = fit_xgb_with_params(X_tr_full, y_tr, best_params)
    return final_model, best_params, tuning_df

tabular_xgb_model, tabular_xgb_params, tabular_tuning_df = tune_xgb_by_val_auprc(
    X_train_base, y_train, X_val_base_fit, y_val_fit, "Tabular XGBoost", seed=SEED
)

tabular_xgb_val_score = tabular_xgb_model.predict_proba(X_val_base)[:, 1]
tabular_xgb_val_fit_score = tabular_xgb_val_score[val_fit_pos]
tabular_xgb_val_select_score = tabular_xgb_val_score[val_select_pos]
tabular_xgb_test_score = tabular_xgb_model.predict_proba(X_test_base)[:, 1]

tabular_tuning_df.to_csv(os.path.join(TABLE_DIR, "tabular_xgboost_tuning_results.csv"), index=False)
print("Tuned Tabular XGBoost trained.")




# %% Cell 17
# ============================================================
# CELL 16: Graph Feature Selection and Tuned BehaviorGraph-XGBoost
# ============================================================

# Step 1: quick probe model to identify useful graph-derived features.
probe_idx = make_xgb_tuning_subset(y_train, max_rows=min(TUNING_MAX_ROWS, 450000), seed=SEED + 11)
probe_params = xgb_candidate_params(scale_pos_weight, seed=SEED)[1]
probe_params = {**probe_params, "n_estimators": 500, "max_depth": 4, "learning_rate": 0.04}
bg_probe = fit_xgb_with_params(X_train.iloc[probe_idx], y_train[probe_idx], probe_params)

probe_importance = pd.DataFrame({
    "Feature": all_feature_cols,
    "Importance": bg_probe.feature_importances_
}).sort_values("Importance", ascending=False)

ranked_graph_features = probe_importance[
    probe_importance["Feature"].isin(behavior_graph_feature_cols)
]["Feature"].tolist()

risk_graph_features = [c for c in graph_risk_cols if c in behavior_graph_feature_cols]
top10_graph = ranked_graph_features[:10]
top20_graph = ranked_graph_features[:20]
top35_graph = ranked_graph_features[:35]

graph_feature_candidates = {
    "Base + GraphRisk": list(dict.fromkeys(base_feature_cols + risk_graph_features)),
    "Base + Top10Graph": list(dict.fromkeys(base_feature_cols + top10_graph)),
    "Base + Top20Graph": list(dict.fromkeys(base_feature_cols + top20_graph)),
    "Base + Top35Graph": list(dict.fromkeys(base_feature_cols + top35_graph)),
    "Base + AllGraph": all_feature_cols,
}

feature_candidate_rows = []
feature_candidate_models = {}

for fs_name, cols in graph_feature_candidates.items():
    X_tr_c, X_val_c, X_test_c, *_ = prepare_matrices(cols)
    tune_idx = make_xgb_tuning_subset(y_train, max_rows=min(TUNING_MAX_ROWS, 500000), seed=SEED + 17)
    # Use the strongest tabular parameters as a starting point to avoid a very expensive nested search.
    params = dict(tabular_xgb_params)
    params["random_state"] = SEED + 17
    model_c = fit_xgb_with_params(X_tr_c.iloc[tune_idx], y_train[tune_idx], params)
    val_fit_score_c = model_c.predict_proba(X_val_c.iloc[val_fit_pos])[:, 1]
    auprc_c = safe_auprc(y_val_fit, val_fit_score_c)
    auroc_c = safe_auc(y_val_fit, val_fit_score_c)
    top1_c = top_k_alert_metrics(y_val_fit, val_fit_score_c, fs_name, k_values=[0.01]).iloc[0]["Recall@k"]
    feature_candidate_rows.append({
        "Feature Set": fs_name,
        "Feature Count": len(cols),
        "Graph Feature Count": len([c for c in cols if c in behavior_graph_feature_cols]),
        "Validation-Fit AUPRC": auprc_c,
        "Validation-Fit AUROC": auroc_c,
        "Validation-Fit Recall@1%": top1_c
    })
    feature_candidate_models[fs_name] = cols

feature_selection_df = pd.DataFrame(feature_candidate_rows).sort_values(
    ["Validation-Fit AUPRC", "Validation-Fit Recall@1%"], ascending=False
)
display(feature_selection_df.round(6))
feature_selection_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_feature_set_selection.csv"), index=False)

selected_behaviorgraph_feature_set = feature_selection_df.iloc[0]["Feature Set"]
selected_behaviorgraph_feature_cols = feature_candidate_models[selected_behaviorgraph_feature_set]

print("Selected BehaviorGraph feature set:", selected_behaviorgraph_feature_set)
print("Selected feature count:", len(selected_behaviorgraph_feature_cols))

X_train_bg_selected, X_val_bg_selected, X_test_bg_selected, X_train_bg_selected_scaled, X_val_bg_selected_scaled, X_test_bg_selected_scaled, bg_selected_imputer, bg_selected_scaler = prepare_matrices(selected_behaviorgraph_feature_cols)
X_val_bg_selected_fit = X_val_bg_selected.iloc[val_fit_pos]
X_val_bg_selected_select = X_val_bg_selected.iloc[val_select_pos]
X_val_bg_selected_scaled_fit = X_val_bg_selected_scaled[val_fit_pos]
X_val_bg_selected_scaled_select = X_val_bg_selected_scaled[val_select_pos]

# Step 2: tune BehaviorGraph-XGBoost on the selected feature set.
behaviorgraph_xgb_model, behaviorgraph_xgb_params, behaviorgraph_tuning_df = tune_xgb_by_val_auprc(
    X_train_bg_selected, y_train, X_val_bg_selected_fit, y_val_fit, "BehaviorGraph-XGBoost", seed=SEED + 23
)

bg_xgb_val_score = behaviorgraph_xgb_model.predict_proba(X_val_bg_selected)[:, 1]
bg_xgb_val_fit_score = bg_xgb_val_score[val_fit_pos]
bg_xgb_val_select_score = bg_xgb_val_score[val_select_pos]
bg_xgb_test_score = behaviorgraph_xgb_model.predict_proba(X_test_bg_selected)[:, 1]

behaviorgraph_tuning_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_xgboost_tuning_results.csv"), index=False)
print("Tuned BehaviorGraph-XGBoost evidence model trained.")



# %% Cell 18
# ============================================================
# CELL 17: Tune RUS-XGBoost Imbalance-Aware Evidence Model
# ============================================================

rus_candidates = [
    {"sampling_strategy": 0.05, "max_depth": 4, "learning_rate": 0.035, "n_estimators": 700, "reg_lambda": 5.0, "reg_alpha": 0.5},
    {"sampling_strategy": 0.10, "max_depth": 4, "learning_rate": 0.030, "n_estimators": 900, "reg_lambda": 6.0, "reg_alpha": 0.75},
    {"sampling_strategy": 0.15, "max_depth": 5, "learning_rate": 0.030, "n_estimators": 800, "reg_lambda": 7.0, "reg_alpha": 0.75},
    {"sampling_strategy": 0.20, "max_depth": 5, "learning_rate": 0.025, "n_estimators": 900, "reg_lambda": 8.0, "reg_alpha": 1.0},
    {"sampling_strategy": 0.30, "max_depth": 4, "learning_rate": 0.025, "n_estimators": 1000, "reg_lambda": 9.0, "reg_alpha": 1.0},
]

rus_rows = []
best_rus_model = None
best_rus_score = -np.inf
best_rus_cfg = None

for i, cfg in enumerate(rus_candidates):
    rus = RandomUnderSampler(random_state=SEED + i, sampling_strategy=cfg["sampling_strategy"])
    X_train_rus, y_train_rus = rus.fit_resample(X_train_bg_selected, y_train)

    rus_model_i = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=0.90,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=cfg["reg_lambda"],
        reg_alpha=cfg["reg_alpha"],
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED + i,
        n_jobs=-1,
        max_delta_step=1
    )
    rus_model_i.fit(X_train_rus, y_train_rus)
    score_fit = rus_model_i.predict_proba(X_val_bg_selected_fit)[:, 1]
    auprc_i = safe_auprc(y_val_fit, score_fit)
    auroc_i = safe_auc(y_val_fit, score_fit)
    top1_i = top_k_alert_metrics(y_val_fit, score_fit, "RUS-XGBoost", k_values=[0.01]).iloc[0]["Recall@k"]
    rows = {"candidate": i, "Validation-Fit AUPRC": auprc_i, "Validation-Fit AUROC": auroc_i, "Validation-Fit Recall@1%": top1_i, **cfg}
    rus_rows.append(rows)
    score_i = auprc_i + 0.02 * top1_i
    if score_i > best_rus_score:
        best_rus_score = score_i
        best_rus_cfg = cfg
        best_rus_model = rus_model_i

rus_tuning_df = pd.DataFrame(rus_rows).sort_values(["Validation-Fit AUPRC", "Validation-Fit Recall@1%"], ascending=False)
display(rus_tuning_df.round(6))
rus_tuning_df.to_csv(os.path.join(TABLE_DIR, "rus_xgboost_tuning_results.csv"), index=False)

# Refit best RUS configuration.
rus = RandomUnderSampler(random_state=SEED, sampling_strategy=best_rus_cfg["sampling_strategy"])
X_train_rus, y_train_rus = rus.fit_resample(X_train_bg_selected, y_train)

rus_xgb_model = xgb.XGBClassifier(
    n_estimators=best_rus_cfg["n_estimators"],
    max_depth=best_rus_cfg["max_depth"],
    learning_rate=best_rus_cfg["learning_rate"],
    subsample=0.90,
    colsample_bytree=0.85,
    min_child_weight=2,
    reg_lambda=best_rus_cfg["reg_lambda"],
    reg_alpha=best_rus_cfg["reg_alpha"],
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
    max_delta_step=1
)
rus_xgb_model.fit(X_train_rus, y_train_rus)

rus_val_score = rus_xgb_model.predict_proba(X_val_bg_selected)[:, 1]
rus_val_fit_score = rus_val_score[val_fit_pos]
rus_val_select_score = rus_val_score[val_select_pos]
rus_test_score = rus_xgb_model.predict_proba(X_test_bg_selected)[:, 1]

print("Best RUS-XGBoost configuration:", best_rus_cfg)
print("RUS-XGBoost trained on resampled data:", X_train_rus.shape)



# %% Cell 19
rus_candidates = [
    {"sampling_strategy": 0.05, "max_depth": 4, "learning_rate": 0.035, "n_estimators": 700, "reg_lambda": 5.0, "reg_alpha": 0.5},
    {"sampling_strategy": 0.10, "max_depth": 4, "learning_rate": 0.030, "n_estimators": 900, "reg_lambda": 6.0, "reg_alpha": 0.75},
    {"sampling_strategy": 0.15, "max_depth": 5, "learning_rate": 0.030, "n_estimators": 800, "reg_lambda": 7.0, "reg_alpha": 0.75},
    {"sampling_strategy": 0.20, "max_depth": 5, "learning_rate": 0.025, "n_estimators": 900, "reg_lambda": 8.0, "reg_alpha": 1.0},
    {"sampling_strategy": 0.30, "max_depth": 4, "learning_rate": 0.025, "n_estimators": 1000, "reg_lambda": 9.0, "reg_alpha": 1.0},
]

rus_rows = []
best_rus_model = None
best_rus_score = -np.inf
best_rus_cfg = None

for i, cfg in enumerate(rus_candidates):
    rus = RandomUnderSampler(random_state=SEED + i, sampling_strategy=cfg["sampling_strategy"])
    X_train_rus, y_train_rus = rus.fit_resample(X_train_bg_selected, y_train)

    rus_model_i = xgb.XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=0.90,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=cfg["reg_lambda"],
        reg_alpha=cfg["reg_alpha"],
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=SEED + i,
        n_jobs=-1,
        max_delta_step=1
    )
    rus_model_i.fit(X_train_rus, y_train_rus)
    score_fit = rus_model_i.predict_proba(X_val_bg_selected_fit)[:, 1]
    auprc_i = safe_auprc(y_val_fit, score_fit)
    top1_i = top_k_alert_metrics(y_val_fit, score_fit, "RUS-XGBoost", k_values=[0.01]).iloc[0]["Recall@k"]

    score_i = auprc_i + 0.02 * top1_i
    if score_i > best_rus_score:
        best_rus_score = score_i
        best_rus_cfg = cfg
        best_rus_model = rus_model_i

# Refit best RUS configuration
rus = RandomUnderSampler(random_state=SEED, sampling_strategy=best_rus_cfg["sampling_strategy"])
X_train_rus, y_train_rus = rus.fit_resample(X_train_bg_selected, y_train)

rus_xgb_model = xgb.XGBClassifier(
    n_estimators=best_rus_cfg["n_estimators"],
    max_depth=best_rus_cfg["max_depth"],
    learning_rate=best_rus_cfg["learning_rate"],
    subsample=0.90,
    colsample_bytree=0.85,
    min_child_weight=2,
    reg_lambda=best_rus_cfg["reg_lambda"],
    reg_alpha=best_rus_cfg["reg_alpha"],
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    random_state=SEED,
    n_jobs=-1,
    max_delta_step=1
)
rus_xgb_model.fit(X_train_rus, y_train_rus)

# Define the missing variables globally
rus_val_score = rus_xgb_model.predict_proba(X_val_bg_selected)[:, 1]
rus_val_fit_score = rus_val_score[val_fit_pos]
rus_val_select_score = rus_val_score[val_select_pos]
rus_test_score = rus_xgb_model.predict_proba(X_test_bg_selected)[:, 1]

print("RUS-XGBoost variables defined and ready.")


# %% Cell 20
# ============================================================
# CELL 18: Tuned Isolation Forest Anomaly Evidence Score
# ============================================================

benign_mask = (y_train == 0)
X_iforest_fit_full = X_train_bg_selected_scaled[benign_mask]

IFOREST_MAX_ROWS = 300000
if len(X_iforest_fit_full) > IFOREST_MAX_ROWS:
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(X_iforest_fit_full), size=IFOREST_MAX_ROWS, replace=False)
    X_iforest_fit = X_iforest_fit_full[idx]
else:
    X_iforest_fit = X_iforest_fit_full

iforest_candidates = [
    {"n_estimators": 250, "max_samples": "auto", "contamination": max(y_train.mean(), 1e-4)},
    {"n_estimators": 350, "max_samples": 0.70, "contamination": max(y_train.mean() * 2, 1e-4)},
    {"n_estimators": 450, "max_samples": 0.85, "contamination": max(y_train.mean() * 5, 1e-4)},
]

if_rows = []
best_iforest = None
best_if_scaler = None
best_if_score = -np.inf

for i, cfg in enumerate(iforest_candidates):
    model_i = IsolationForest(
        n_estimators=cfg["n_estimators"],
        max_samples=cfg["max_samples"],
        contamination=cfg["contamination"],
        random_state=SEED + i,
        n_jobs=-1
    )
    model_i.fit(X_iforest_fit)
    raw_fit_i = -model_i.decision_function(X_val_bg_selected_scaled_fit)
    scaler_i = MinMaxScaler()
    score_fit_i = scaler_i.fit_transform(raw_fit_i.reshape(-1, 1)).ravel()
    auprc_i = safe_auprc(y_val_fit, score_fit_i)
    auroc_i = safe_auc(y_val_fit, score_fit_i)
    top1_i = top_k_alert_metrics(y_val_fit, score_fit_i, "Isolation Forest", k_values=[0.01]).iloc[0]["Recall@k"]
    if_rows.append({"candidate": i, "Validation-Fit AUPRC": auprc_i, "Validation-Fit AUROC": auroc_i, "Validation-Fit Recall@1%": top1_i, **cfg})
    score_i = auprc_i + 0.02 * top1_i
    if score_i > best_if_score:
        best_if_score = score_i
        best_iforest = model_i
        best_if_scaler = scaler_i

iforest_tuning_df = pd.DataFrame(if_rows).sort_values(["Validation-Fit AUPRC", "Validation-Fit Recall@1%"], ascending=False)
display(iforest_tuning_df.round(6))
iforest_tuning_df.to_csv(os.path.join(TABLE_DIR, "iforest_tuning_results.csv"), index=False)

iforest = best_iforest
if_val_raw = -iforest.decision_function(X_val_bg_selected_scaled)
if_test_raw = -iforest.decision_function(X_test_bg_selected_scaled)

# Refit scaler on validation-fit anomaly scores only.
if_scaler = MinMaxScaler()
if_scaler.fit(if_val_raw[val_fit_pos].reshape(-1, 1))
if_val_score = np.clip(if_scaler.transform(if_val_raw.reshape(-1, 1)).ravel(), 0, 1)
if_val_fit_score = if_val_score[val_fit_pos]
if_val_select_score = if_val_score[val_select_pos]
if_test_score = np.clip(if_scaler.transform(if_test_raw.reshape(-1, 1)).ravel(), 0, 1)

print("Tuned Isolation Forest anomaly evidence generated.")



# %% Cell 21
# ============================================================
# CELL 19: Transparent Rule-Based AML Risk Evidence Score
# ============================================================

amount_q95 = train_df_f["amount_paid"].quantile(0.95)
amount_q99 = train_df_f["amount_paid"].quantile(0.99)
degree_ratio_q95 = train_df_f["degree_ratio"].quantile(0.95)
flow_ratio_q95 = train_df_f["amount_flow_ratio"].quantile(0.95)
edge_txn_q95 = train_df_f["edge_txn_count"].quantile(0.95)
sender_txn_q95 = train_df_f["sender_txn_count"].quantile(0.95)
receiver_txn_q95 = train_df_f["receiver_txn_count"].quantile(0.95)
structural_risk_q95 = train_df_f["structural_risk_raw"].quantile(0.95)

def rule_based_risk_score(data):
    score = np.zeros(len(data), dtype=float)
    score += 0.16 * (data["amount_paid"] >= amount_q95).astype(float)
    score += 0.20 * (data["amount_paid"] >= amount_q99).astype(float)
    score += 0.10 * data["cross_bank"].astype(float)
    score += 0.10 * data["cross_currency"].astype(float)
    score += 0.06 * data["is_night"].astype(float)
    score += 0.10 * (data["degree_ratio"] >= degree_ratio_q95).astype(float)
    score += 0.08 * (data["amount_flow_ratio"] >= flow_ratio_q95).astype(float)
    score += 0.06 * (data["edge_txn_count"] >= edge_txn_q95).astype(float)
    score += 0.06 * (data["sender_txn_count"] >= sender_txn_q95).astype(float)
    score += 0.04 * (data["receiver_txn_count"] >= receiver_txn_q95).astype(float)
    score += 0.04 * (data["structural_risk_raw"] >= structural_risk_q95).astype(float)
    return np.clip(score, 0, 1)

rule_val_score = rule_based_risk_score(val_df_f)
rule_val_fit_score = rule_val_score[val_fit_pos]
rule_val_select_score = rule_val_score[val_select_pos]
rule_test_score = rule_based_risk_score(test_df_f)
print("Rule-based AML risk evidence generated.")



# %% Cell 22
# ============================================================
# CELL 20: Probability Calibration for Evidence Scores
# ============================================================

def score_to_logit(score):
    return logit(np.clip(np.asarray(score).astype(float), EPS, 1 - EPS))

def fit_platt_calibrator(fit_score, y_fit):
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=3000)
    lr.fit(score_to_logit(fit_score).reshape(-1, 1), y_fit)
    return lr

def apply_platt_calibrator(calibrator, score):
    return calibrator.predict_proba(score_to_logit(score).reshape(-1, 1))[:, 1]

evidence_raw = {
    "Tabular XGBoost": (tabular_xgb_val_score, tabular_xgb_test_score),
    "BehaviorGraph-XGBoost": (bg_xgb_val_score, bg_xgb_test_score),
    "RUS-XGBoost": (rus_val_score, rus_test_score),
    "Isolation Forest": (if_val_score, if_test_score),
    "Rule-Based AML Risk": (rule_val_score, rule_test_score),
}

evidence_raw_fit_select = {
    "Tabular XGBoost": (tabular_xgb_val_fit_score, tabular_xgb_val_select_score, tabular_xgb_test_score),
    "BehaviorGraph-XGBoost": (bg_xgb_val_fit_score, bg_xgb_val_select_score, bg_xgb_test_score),
    "RUS-XGBoost": (rus_val_fit_score, rus_val_select_score, rus_test_score),
    "Isolation Forest": (if_val_fit_score, if_val_select_score, if_test_score),
    "Rule-Based AML Risk": (rule_val_fit_score, rule_val_select_score, rule_test_score),
}

calibrators = {}
evidence_calibrated = {}
evidence_calibrated_fit_select = {}

for name, (fit_score, select_score, test_score) in evidence_raw_fit_select.items():
    cal = fit_platt_calibrator(fit_score, y_val_fit)
    calibrators[name] = cal
    full_val_cal = apply_platt_calibrator(cal, evidence_raw[name][0])
    fit_cal = full_val_cal[val_fit_pos]
    select_cal = full_val_cal[val_select_pos]
    test_cal = apply_platt_calibrator(cal, test_score)
    evidence_calibrated[name] = (full_val_cal, test_cal)
    evidence_calibrated_fit_select[name] = (fit_cal, select_cal, test_cal)

print("Platt calibration completed using validation-fit only.")



# %% Cell 23
# ============================================================
# CELL 21: Evaluate Raw and Calibrated Evidence Streams
# ============================================================

rows = []
for name, (val_score, test_score) in evidence_raw.items():
    rows.append(evaluate_predictions(name + " Raw", "Validation-Fit", y_val_fit, val_score[val_fit_pos], threshold=0.5))
    rows.append(evaluate_predictions(name + " Raw", "Validation-Select", y_val_select, val_score[val_select_pos], threshold=0.5))
    rows.append(evaluate_predictions(name + " Raw", "Test", y_test, test_score, threshold=0.5))

for name, (val_score, test_score) in evidence_calibrated.items():
    rows.append(evaluate_predictions(name + " Calibrated", "Validation-Fit", y_val_fit, val_score[val_fit_pos], threshold=0.5))
    rows.append(evaluate_predictions(name + " Calibrated", "Validation-Select", y_val_select, val_score[val_select_pos], threshold=0.5))
    rows.append(evaluate_predictions(name + " Calibrated", "Test", y_test, test_score, threshold=0.5))

evidence_metrics_df = pd.DataFrame(rows)
display(evidence_metrics_df.round(5))
evidence_metrics_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_evidence_metrics_raw_calibrated.csv"), index=False)

topk_rows = []
for name, (_, test_score) in evidence_raw.items():
    topk_rows.append(top_k_alert_metrics(y_test, test_score, name + " Raw"))
for name, (_, test_score) in evidence_calibrated.items():
    topk_rows.append(top_k_alert_metrics(y_test, test_score, name + " Calibrated"))

evidence_topk_df = pd.concat(topk_rows, ignore_index=True)
display(evidence_topk_df.round(5))
evidence_topk_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_evidence_topk.csv"), index=False)



# %% Cell 24
# ============================================================
# CELL 22: Feature Importance and Graph Contribution Audit
# ============================================================

importance_df = pd.DataFrame({
    "Feature": selected_behaviorgraph_feature_cols,
    "Importance": behaviorgraph_xgb_model.feature_importances_
}).sort_values("Importance", ascending=False)
importance_df["Feature Group"] = np.where(importance_df["Feature"].isin(behavior_graph_feature_cols), "Graph-derived", "Base tabular")

display(importance_df.head(30))
importance_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_xgboost_feature_importance.csv"), index=False)

group_importance = importance_df.groupby("Feature Group")["Importance"].sum().reset_index()
display(group_importance)
group_importance.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_feature_group_importance.csv"), index=False)

plt.figure(figsize=(8, 7))
top_imp = importance_df.head(20).iloc[::-1]
plt.barh(top_imp["Feature"], top_imp["Importance"])
plt.title("Top 20 Feature Importances: Tuned BehaviorGraph-XGBoost")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "feature_importance_behaviorgraph_xgboost_top20.png"), dpi=300)
plt.show()



# %% Cell 25
# ============================================================
# CELL 23: Construct Compact Bayesian Fusion Meta-Features
# ============================================================

tab_fit_cal, tab_select_cal, tab_test_cal = evidence_calibrated_fit_select["Tabular XGBoost"]
bg_fit_cal, bg_select_cal, bg_test_cal = evidence_calibrated_fit_select["BehaviorGraph-XGBoost"]
rus_fit_cal, rus_select_cal, rus_test_cal = evidence_calibrated_fit_select["RUS-XGBoost"]
if_fit_cal, if_select_cal, if_test_cal = evidence_calibrated_fit_select["Isolation Forest"]
rule_fit_cal, rule_select_cal, rule_test_cal = evidence_calibrated_fit_select["Rule-Based AML Risk"]

def build_fusion_frame(data, tab_cal, bg_cal, rus_cal, if_cal, rule_cal):
    frame = pd.DataFrame(index=data.index)
    frame["tabular_xgb_logit"] = score_to_logit(tab_cal)
    frame["behaviorgraph_xgb_logit"] = score_to_logit(bg_cal)
    frame["rus_xgb_logit"] = score_to_logit(rus_cal)
    frame["iforest_logit"] = score_to_logit(if_cal)
    frame["rule_risk_logit"] = score_to_logit(rule_cal)
    frame["structural_risk_log"] = np.log1p(data["structural_risk_raw"].clip(lower=0).values)
    frame["degree_ratio_log"] = np.log1p(data["degree_ratio"].clip(lower=0).values)
    frame["amount_flow_ratio_log"] = np.log1p(data["amount_flow_ratio"].clip(lower=0).values)
    frame["txn_flow_ratio_log"] = np.log1p(data["txn_flow_ratio"].clip(lower=0).values)
    frame["sender_receiver_degree_log"] = np.log1p(data["sender_receiver_degree_sum"].clip(lower=0).values)
    frame["sender_receiver_txn_log"] = np.log1p(data["sender_receiver_txn_sum"].clip(lower=0).values)
    frame["log_amount_paid"] = data["log_amount_paid"].values
    frame["cross_bank"] = data["cross_bank"].values
    frame["cross_currency"] = data["cross_currency"].values
    frame["same_account"] = data["same_account"].values
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)

fusion_fit_df = build_fusion_frame(val_fit_df_f, tab_fit_cal, bg_fit_cal, rus_fit_cal, if_fit_cal, rule_fit_cal)
fusion_select_df = build_fusion_frame(val_select_df_f, tab_select_cal, bg_select_cal, rus_select_cal, if_select_cal, rule_select_cal)
fusion_test_df = build_fusion_frame(test_df_f, tab_test_cal, bg_test_cal, rus_test_cal, if_test_cal, rule_test_cal)

fusion_feature_names_initial = fusion_fit_df.columns.tolist()
print("Initial fusion features:", fusion_feature_names_initial)
display(fusion_fit_df.head())



# %% Cell 26
# ============================================================
# CELL 24: Fusion Feature Pruning, Selection, and Standardization
# ============================================================

# This cell keeps the NUTS layer compact and stable. It protects the strongest
# evidence streams, removes highly correlated variables, and limits the final
# Bayesian fusion space to the most useful meta-features.

def correlation_prune(df, threshold=0.95, protected=("tabular_xgb_logit", "rus_xgb_logit", "iforest_logit", "structural_risk_log", "degree_ratio_log")):
    corr = df.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = []
    for col in upper.columns:
        if col in protected:
            continue
        if any(upper[col] > threshold):
            drop.append(col)
    keep = [c for c in df.columns if c not in drop]
    return keep, drop, corr

fusion_feature_names_corr, dropped_fusion_features, fusion_corr = correlation_prune(fusion_fit_df, threshold=0.95)

# Deterministic screening model: used only to rank candidate fusion features before NUTS.
screen_imputer = SimpleImputer(strategy="median")
screen_scaler = StandardScaler()
X_screen_fit = screen_scaler.fit_transform(screen_imputer.fit_transform(fusion_fit_df[fusion_feature_names_corr]))
X_screen_select = screen_scaler.transform(screen_imputer.transform(fusion_select_df[fusion_feature_names_corr]))

screen_lr = LogisticRegression(C=0.5, penalty="l2", solver="lbfgs", max_iter=3000)
screen_lr.fit(X_screen_fit, y_val_fit)

screen_coef = pd.DataFrame({
    "Feature": fusion_feature_names_corr,
    "AbsCoef": np.abs(screen_lr.coef_[0]),
    "Coef": screen_lr.coef_[0]
}).sort_values("AbsCoef", ascending=False)

# Univariate validation-select ranking signal for candidate features.
univar_rows = []
for feat in fusion_feature_names_corr:
    raw = fusion_select_df[feat].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    # Use direction-free AUPRC: if a feature is negatively associated, reverse it.
    ap_pos = safe_auprc(y_val_select, raw)
    ap_neg = safe_auprc(y_val_select, -raw)
    univar_rows.append({"Feature": feat, "UnivarAUPRC": max(ap_pos, ap_neg), "BestDirection": "positive" if ap_pos >= ap_neg else "negative"})
univar_df = pd.DataFrame(univar_rows)

screen_coef = screen_coef.merge(univar_df, on="Feature", how="left")
# Composite rank: coefficient magnitude is primary; univariate AUPRC helps keep rank-useful features.
screen_coef["CoefRank"] = screen_coef["AbsCoef"].rank(ascending=False, method="min")
screen_coef["AUPRCRank"] = screen_coef["UnivarAUPRC"].rank(ascending=False, method="min")
screen_coef["CompositeRank"] = 0.70 * screen_coef["CoefRank"] + 0.30 * screen_coef["AUPRCRank"]
screen_coef = screen_coef.sort_values(["CompositeRank", "CoefRank"]).reset_index(drop=True)

protected_features = ["tabular_xgb_logit", "rus_xgb_logit", "iforest_logit", "structural_risk_log", "degree_ratio_log"]
protected_features = [c for c in protected_features if c in fusion_feature_names_corr]

candidate_ranked = screen_coef[
    (screen_coef["AbsCoef"] >= MIN_SCREEN_ABS_COEF) | (screen_coef["Feature"].isin(protected_features))
]["Feature"].tolist()

fusion_feature_names = []
for feat in protected_features + candidate_ranked:
    if feat in fusion_feature_names_corr and feat not in fusion_feature_names:
        fusion_feature_names.append(feat)
    if len(fusion_feature_names) >= MAX_NUTS_FUSION_FEATURES:
        break

# Safety: never run an empty or too-small fusion model.
if len(fusion_feature_names) < min(5, len(fusion_feature_names_corr)):
    for feat in screen_coef["Feature"].tolist():
        if feat not in fusion_feature_names:
            fusion_feature_names.append(feat)
        if len(fusion_feature_names) >= min(5, len(fusion_feature_names_corr)):
            break

print("Dropped highly correlated fusion features:", dropped_fusion_features)
print("Selected NUTS fusion features:", fusion_feature_names)
display(screen_coef.round(6))

fusion_imputer = SimpleImputer(strategy="median")
X_fusion_fit_imp = fusion_imputer.fit_transform(fusion_fit_df[fusion_feature_names])
X_fusion_select_imp = fusion_imputer.transform(fusion_select_df[fusion_feature_names])
X_fusion_test_imp = fusion_imputer.transform(fusion_test_df[fusion_feature_names])

fusion_scaler = StandardScaler()
X_fusion_fit = fusion_scaler.fit_transform(X_fusion_fit_imp)
X_fusion_select = fusion_scaler.transform(X_fusion_select_imp)
X_fusion_test = fusion_scaler.transform(X_fusion_test_imp)

pd.DataFrame(fusion_corr).to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_fusion_feature_correlation.csv"))
screen_coef.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_fusion_feature_screening.csv"), index=False)
print("Fusion matrices:", X_fusion_fit.shape, X_fusion_select.shape, X_fusion_test.shape)



# %% Cell 27
# ============================================================
# CELL 25: Stratified Case-Control Subsample for NUTS with Likelihood Weights
# ============================================================

def stratified_weighted_case_control_sample(X, y, max_rows=60000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_pos_full = len(pos_idx)
    n_neg_full = len(neg_idx)

    if len(y) <= max_rows:
        idx = np.arange(len(y))
    else:
        # Use all/most positives available in validation-fit and enough negatives for stable posterior geometry.
        n_pos_sub = min(n_pos_full, max(1, int(max_rows * 0.35)))
        pos_sub = rng.choice(pos_idx, size=n_pos_sub, replace=False) if n_pos_full > n_pos_sub else pos_idx
        n_neg_sub = max_rows - len(pos_sub)
        n_neg_sub = min(n_neg_sub, n_neg_full)
        neg_sub = rng.choice(neg_idx, size=n_neg_sub, replace=False)
        idx = np.concatenate([pos_sub, neg_sub])
        rng.shuffle(idx)

    X_sub = X[idx]
    y_sub = y[idx]
    pos_sub_count = max((y_sub == 1).sum(), 1)
    neg_sub_count = max((y_sub == 0).sum(), 1)
    w = np.where(y_sub == 1, n_pos_full / pos_sub_count, n_neg_full / neg_sub_count).astype(float)
    w = w / w.mean()
    return X_sub, y_sub, w, idx

X_nuts, y_nuts, w_nuts, nuts_idx = stratified_weighted_case_control_sample(
    X_fusion_fit, y_val_fit, max_rows=NUTS_MAX_FUSION_ROWS, seed=SEED
)

print("NUTS fusion training subset:", X_nuts.shape)
print("Full validation-fit positive rate:", y_val_fit.mean())
print("Subset positive rate:", y_nuts.mean())
print("Weight range:", w_nuts.min(), w_nuts.max(), "mean:", w_nuts.mean())



# %% Cell 28
# ============================================================
# CELL 26: Deterministic Logistic Fusion Baseline and NUTS Initialization
# ============================================================

# Use stronger regularized logistic fusion and tune C on validation-select.
meta_c_values = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
meta_rows = []
best_meta_lr = None
best_meta_score = -np.inf
best_meta_c = None

for cval in meta_c_values:
    lr_i = LogisticRegression(C=cval, penalty="l2", solver="lbfgs", max_iter=4000, class_weight=None)
    lr_i.fit(X_nuts, y_nuts, sample_weight=w_nuts)
    select_score_i = lr_i.predict_proba(X_fusion_select)[:, 1]
    auprc_i = safe_auprc(y_val_select, select_score_i)
    auroc_i = safe_auc(y_val_select, select_score_i)
    top1_i = top_k_alert_metrics(y_val_select, select_score_i, "Deterministic Fusion", k_values=[0.01]).iloc[0]["Recall@k"]
    meta_rows.append({"C": cval, "Validation-Select AUPRC": auprc_i, "Validation-Select AUROC": auroc_i, "Validation-Select Recall@1%": top1_i})
    score_i = auprc_i + 0.02 * top1_i
    if score_i > best_meta_score:
        best_meta_score = score_i
        best_meta_lr = lr_i
        best_meta_c = cval

meta_tuning_df = pd.DataFrame(meta_rows).sort_values(["Validation-Select AUPRC", "Validation-Select Recall@1%"], ascending=False)
display(meta_tuning_df.round(6))
meta_tuning_df.to_csv(os.path.join(TABLE_DIR, "deterministic_fusion_tuning_results.csv"), index=False)

meta_lr = best_meta_lr
meta_fit_score = meta_lr.predict_proba(X_fusion_fit)[:, 1]
meta_select_score = meta_lr.predict_proba(X_fusion_select)[:, 1]
meta_test_score = meta_lr.predict_proba(X_fusion_test)[:, 1]

init_alpha = float(meta_lr.intercept_[0])
init_beta = meta_lr.coef_[0].astype(float)

print("Best deterministic logistic fusion C:", best_meta_c)
print("Initial alpha:", init_alpha)
print("Initial beta:", init_beta)

meta_lr_rows = [
    evaluate_predictions("Deterministic Logistic Fusion", "Validation-Fit", y_val_fit, meta_fit_score, threshold=0.5),
    evaluate_predictions("Deterministic Logistic Fusion", "Validation-Select", y_val_select, meta_select_score, threshold=0.5),
    evaluate_predictions("Deterministic Logistic Fusion", "Test", y_test, meta_test_score, threshold=0.5)
]
display(pd.DataFrame(meta_lr_rows).round(5))



# %% Cell 29
# ============================================================
# CELL 27: Bayesian Posterior Fusion with NUTS
# ============================================================

coords = {"feature": fusion_feature_names}
prior_alpha_mu = init_alpha if USE_ANCHORED_NUTS_PRIOR else float(logit(np.clip(y_val_fit.mean(), EPS, 1 - EPS)))
prior_alpha_sigma = ANCHOR_ALPHA_SIGMA if USE_ANCHORED_NUTS_PRIOR else 1.5
prior_beta_mu = init_beta if USE_ANCHORED_NUTS_PRIOR else np.zeros(len(fusion_feature_names))
prior_beta_sigma = ANCHOR_BETA_SIGMA if USE_ANCHORED_NUTS_PRIOR else 0.75

with pm.Model(coords=coords) as behaviorgraph_nuts_model:
    X_data = pm.Data("X_data", X_nuts.astype("float64"))
    y_data = pm.Data("y_data", y_nuts.astype("int64"))
    w_data = pm.Data("w_data", w_nuts.astype("float64"))

    alpha = pm.Normal("alpha", mu=prior_alpha_mu, sigma=prior_alpha_sigma)
    beta = pm.Normal("beta", mu=prior_beta_mu, sigma=prior_beta_sigma, dims="feature")

    eta = alpha + pm.math.dot(X_data, beta)
    log_lik_vec = pm.logp(pm.Bernoulli.dist(logit_p=eta), y_data)
    pm.Potential("weighted_likelihood", pm.math.sum(w_data * log_lik_vec))

    idata = pm.sample(
        draws=NUTS_DRAWS,
        tune=NUTS_TUNE,
        chains=NUTS_CHAINS,
        cores=NUTS_CORES,
        target_accept=NUTS_TARGET_ACCEPT,
        init="jitter+adapt_diag",
        initvals={"alpha": init_alpha, "beta": init_beta},
        random_seed=SEED,
        return_inferencedata=True
    )

print("NUTS posterior sampling complete.")
print("Anchored NUTS prior used:", USE_ANCHORED_NUTS_PRIOR)



# %% Cell 30
# ============================================================
# CELL 28: NUTS Diagnostic Summary and Quality Gate
# ============================================================

nuts_summary = az.summary(idata, var_names=["alpha", "beta"], round_to=4)
display(nuts_summary)
nuts_summary.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_diagnostic_summary.csv"))

divergences = int(idata.sample_stats["diverging"].sum().values)
max_rhat = float(nuts_summary["r_hat"].max()) if "r_hat" in nuts_summary.columns else np.nan
min_ess_bulk = float(nuts_summary["ess_bulk"].min()) if "ess_bulk" in nuts_summary.columns else np.nan
min_ess_tail = float(nuts_summary["ess_tail"].min()) if "ess_tail" in nuts_summary.columns else np.nan

diagnostic_gate = pd.DataFrame([{
    "Divergences": divergences,
    "Max Rhat": max_rhat,
    "Min Bulk ESS": min_ess_bulk,
    "Min Tail ESS": min_ess_tail,
    "Pass Divergence Gate": divergences == 0,
    "Pass Rhat Gate": max_rhat <= 1.01,
    "Pass ESS Gate": min_ess_bulk >= 100 and min_ess_tail >= 100
}])
display(diagnostic_gate)
diagnostic_gate.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_diagnostic_quality_gate.csv"), index=False)

if divergences > 0 or max_rhat > 1.01:
    print("WARNING: NUTS diagnostics are not yet manuscript-ready.")
    print("Recommended next action: increase NUTS_TUNE, raise target_accept to 0.99, or further reduce fusion feature collinearity.")
else:
    print("NUTS diagnostics passed the main manuscript-readiness gate.")



# %% Cell 31
# ============================================================
# CELL 29: Posterior Coefficient Interpretation
# ============================================================

posterior = idata.posterior
alpha_samples = posterior["alpha"].stack(sample=("chain", "draw")).values
beta_samples = posterior["beta"].stack(sample=("chain", "draw")).values

coef_rows = []
for j, feat in enumerate(fusion_feature_names):
    vals = beta_samples[j, :]
    coef_rows.append({
        "Feature": feat,
        "Posterior Mean": float(np.mean(vals)),
        "Posterior SD": float(np.std(vals)),
        "Lower 95%": float(np.quantile(vals, 0.025)),
        "Upper 95%": float(np.quantile(vals, 0.975)),
        "P(beta > 0)": float((vals > 0).mean()),
        "P(beta < 0)": float((vals < 0).mean())
    })

coef_df = pd.DataFrame(coef_rows).sort_values("Posterior Mean", ascending=False)
display(coef_df.round(5))
coef_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_posterior_coefficients.csv"), index=False)



# %% Cell 32
# ============================================================
# CELL 30: Posterior Predictive Function and NUTS Score Variants
# ============================================================

def posterior_predict_fusion(X, idata, batch_size=100000):
    posterior = idata.posterior
    alpha = posterior["alpha"].stack(sample=("chain", "draw")).values.astype(float)
    beta = posterior["beta"].stack(sample=("chain", "draw")).values.astype(float)

    n = X.shape[0]
    means, medians, stds, lowers, uppers, eta_means = [], [], [], [], [], []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        eta = alpha[None, :] + X[start:end].dot(beta)
        prob = expit(eta)
        means.append(prob.mean(axis=1))
        medians.append(np.median(prob, axis=1))
        stds.append(prob.std(axis=1))
        lowers.append(np.quantile(prob, 0.025, axis=1))
        uppers.append(np.quantile(prob, 0.975, axis=1))
        eta_means.append(expit(eta.mean(axis=1)))

    return {
        "mean": np.concatenate(means),
        "median": np.concatenate(medians),
        "std": np.concatenate(stds),
        "lower": np.concatenate(lowers),
        "upper": np.concatenate(uppers),
        "eta_mean_prob": np.concatenate(eta_means)
    }

def rank_normalize(x):
    x = np.asarray(x, dtype=float)
    r = pd.Series(x).rank(method="average").values
    if len(r) <= 1:
        return np.zeros_like(r, dtype=float)
    return (r - 1) / (len(r) - 1)

bayes_select_pred = posterior_predict_fusion(X_fusion_select, idata)
bayes_test_pred = posterior_predict_fusion(X_fusion_test, idata)

def build_bayes_score_variants(pred):
    mean = np.clip(pred["mean"], EPS, 1 - EPS)
    median = np.clip(pred["median"], EPS, 1 - EPS)
    std = pred["std"]
    lower = np.clip(pred["lower"], EPS, 1 - EPS)
    upper = np.clip(pred["upper"], EPS, 1 - EPS)
    eta_mean = np.clip(pred["eta_mean_prob"], EPS, 1 - EPS)

    variants = {
        "posterior_mean": mean,
        "posterior_median": median,
        "posterior_eta_mean_prob": eta_mean,
        "posterior_lower_95": lower,
        "posterior_upper_95": upper,
        "mean_plus_0.25sd": np.clip(mean + 0.25 * std, EPS, 1 - EPS),
        "mean_plus_0.50sd": np.clip(mean + 0.50 * std, EPS, 1 - EPS),
        "mean_minus_0.25sd": np.clip(mean - 0.25 * std, EPS, 1 - EPS),
        "mean_minus_0.50sd": np.clip(mean - 0.50 * std, EPS, 1 - EPS),
    }

    if RANK_BLEND_VARIANTS_ENABLED:
        # Rank blends preserve ordering signals from conservative and central posterior summaries.
        variants["rank_blend_lower_eta"] = 0.50 * rank_normalize(lower) + 0.50 * rank_normalize(eta_mean)
        variants["rank_blend_lower_mean"] = 0.50 * rank_normalize(lower) + 0.50 * rank_normalize(mean)
        variants["rank_blend_conservative"] = (
            0.40 * rank_normalize(lower)
            + 0.35 * rank_normalize(np.clip(mean - 0.25 * std, EPS, 1 - EPS))
            + 0.25 * rank_normalize(eta_mean)
        )
        variants["rank_blend_alert"] = (
            0.45 * rank_normalize(upper)
            + 0.35 * rank_normalize(mean)
            + 0.20 * rank_normalize(std)
        )
    return variants

bayes_select_variants = build_bayes_score_variants(bayes_select_pred)
bayes_test_variants = build_bayes_score_variants(bayes_test_pred)

variant_rows = []
for name, score in bayes_select_variants.items():
    score = np.clip(score, EPS, 1 - EPS)
    row = evaluate_predictions(f"BehaviorGraph-NUTS {name}", "Validation-Select", y_val_select, score, threshold=0.5)
    row["Variant"] = name
    for ar in [0.001, 0.005, 0.01, 0.02]:
        top = top_k_alert_metrics(y_val_select, score, name, k_values=[ar]).iloc[0]
        row[f"Recall@{ar*100:.1f}%"] = top["Recall@k"]
        row[f"Precision@{ar*100:.1f}%"] = top["Precision@k"]
    variant_rows.append(row)

variant_selection_df = pd.DataFrame(variant_rows)

# Composite score variant selection: AUPRC remains primary, but near-ties are resolved using calibration
# and low-alert recall rather than selecting a variant on a tiny AUPRC advantage alone.
best_auprc = variant_selection_df["AUPRC"].max()
near_best = variant_selection_df[variant_selection_df["AUPRC"] >= best_auprc - SCORE_AUPRC_TIE_TOL].copy()
near_best["AUPRC_rank"] = near_best["AUPRC"].rank(ascending=False, method="min")
near_best["Brier_rank"] = near_best["Brier"].rank(ascending=True, method="min")
near_best["ECE_rank"] = near_best["ECE"].rank(ascending=True, method="min")
near_best["Recall1_rank"] = near_best["Recall@1.0%"].rank(ascending=False, method="min")
near_best["CompositeSelectionRank"] = (
    0.45 * near_best["AUPRC_rank"]
    + 0.25 * near_best["Brier_rank"]
    + 0.15 * near_best["ECE_rank"]
    + 0.15 * near_best["Recall1_rank"]
)

variant_selection_df = variant_selection_df.merge(
    near_best[["Variant", "CompositeSelectionRank"]], on="Variant", how="left"
)
variant_selection_df["CompositeSelectionRank"] = variant_selection_df["CompositeSelectionRank"].fillna(9999)
variant_selection_df = variant_selection_df.sort_values(["CompositeSelectionRank", "AUPRC", "Recall@1.0%"], ascending=[True, False, False])

display_cols = ["Variant", "AUROC", "AUPRC", "Brier", "ECE", "MCE", "Recall@0.1%", "Recall@0.5%", "Recall@1.0%", "CompositeSelectionRank"]
display(variant_selection_df[display_cols].round(6))
variant_selection_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_score_variant_selection.csv"), index=False)

selected_bayes_variant = variant_selection_df.iloc[0]["Variant"]
bayes_select_score_uncal = np.clip(bayes_select_variants[selected_bayes_variant], EPS, 1 - EPS)
bayes_test_score_uncal = np.clip(bayes_test_variants[selected_bayes_variant], EPS, 1 - EPS)

# Final calibration for the selected Bayesian score using validation-select only.
bayes_final_calibrator = fit_platt_calibrator(bayes_select_score_uncal, y_val_select)
bayes_select_score = apply_platt_calibrator(bayes_final_calibrator, bayes_select_score_uncal)
bayes_test_score = apply_platt_calibrator(bayes_final_calibrator, bayes_test_score_uncal)

# Preserve posterior mean outputs for uncertainty tables even if selected ranking score differs.
bayes_test_posterior_mean = np.clip(bayes_test_pred["mean"], EPS, 1 - EPS)
bayes_select_posterior_mean = np.clip(bayes_select_pred["mean"], EPS, 1 - EPS)

print("Selected BehaviorGraph-NUTS score variant:", selected_bayes_variant)
print("Score selection mode:", SCORE_SELECTION_MODE)
print("Validation-select selected score range:", bayes_select_score.min(), bayes_select_score.max())
print("Test selected score range:", bayes_test_score.min(), bayes_test_score.max())



# %% Cell 33
# ============================================================
# CELL 31: Threshold Search for Operationally Constrained Bayesian Policies
# ============================================================

best_f2_policy, f2_policy_grid = threshold_search_constrained(y_val_select, bayes_select_score, max_alert_rate=MAX_MANUSCRIPT_ALERT_RATE, metric="F2")
best_mcc_policy, mcc_policy_grid = threshold_search_constrained(y_val_select, bayes_select_score, max_alert_rate=MAX_MANUSCRIPT_ALERT_RATE, metric="MCC")

print("Best F2 policy under alert-rate constraint:")
display(pd.DataFrame([best_f2_policy]).round(5))
print("Best MCC policy under alert-rate constraint:")
display(pd.DataFrame([best_mcc_policy]).round(5))

policy_grid = f2_policy_grid.copy()
policy_grid.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_threshold_policy_grid_constrained.csv"), index=False)

plt.figure(figsize=(8, 5))
plt.plot(policy_grid["Threshold"], policy_grid["F2"], label="F2")
plt.plot(policy_grid["Threshold"], policy_grid["MCC"], label="MCC")
plt.plot(policy_grid["Threshold"], policy_grid["Alert Rate"], label="Alert Rate")
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.title("Constrained Threshold Search for BehaviorGraph-NUTS")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_constrained_threshold_search.png"), dpi=300)
plt.show()



# %% Cell 34
# ============================================================
# CELL 32: Test Evaluation Under Bayesian Threshold Policies
# ============================================================

bayes_policy_specs = [
    ("BehaviorGraph-NUTS F2-Constrained Policy", float(best_f2_policy["Threshold"])),
    ("BehaviorGraph-NUTS MCC-Constrained Policy", float(best_mcc_policy["Threshold"]))
]

bayes_policy_rows = []
for name, threshold in bayes_policy_specs:
    row = evaluate_predictions(name, "Test", y_test, bayes_test_score, threshold=threshold)
    bayes_policy_rows.append(row)
    print("=" * 80)
    print(name, "threshold=", threshold)
    print(classification_report(y_test, (bayes_test_score >= threshold).astype(int), digits=4, zero_division=0))

bayes_policy_df = pd.DataFrame(bayes_policy_rows)
display(bayes_policy_df.round(5))
bayes_policy_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_policy_test_metrics.csv"), index=False)



# %% Cell 35
# ============================================================
# CELL 33: Alert-Rate-Constrained Top-k AML Policies
# ============================================================

alert_policy_rows = []
for ar in ALERT_RATES:
    threshold = threshold_for_alert_rate(bayes_select_score, ar)
    row = evaluate_predictions(f"BehaviorGraph-NUTS Top {ar*100:.1f}% Alert Policy", "Test", y_test, bayes_test_score, threshold=threshold)
    row["Validation-Select-Derived Alert Rate"] = ar
    alert_policy_rows.append(row)

alert_policy_df = pd.DataFrame(alert_policy_rows)
display(alert_policy_df.round(5))
alert_policy_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_alert_rate_policy_metrics.csv"), index=False)

# Fair top-k comparison across all current models.
topk_model_scores = {
    "Tabular XGBoost": tab_test_cal,
    "BehaviorGraph-XGBoost": bg_test_cal,
    "RUS-XGBoost": rus_test_cal,
    "Isolation Forest": if_test_cal,
    "Rule-Based AML Risk": rule_test_cal,
    "Deterministic Logistic Fusion": meta_test_score,
    "BehaviorGraph-NUTS": bayes_test_score,
}

all_topk_rows = []
for model_name, score in topk_model_scores.items():
    all_topk_rows.append(top_k_alert_metrics(y_test, score, model_name, ALERT_RATES))

all_topk_df = pd.concat(all_topk_rows, ignore_index=True)
display(all_topk_df.round(5))
all_topk_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_all_models_topk_alert_metrics.csv"), index=False)

bayes_topk_df = all_topk_df[all_topk_df["Model"] == "BehaviorGraph-NUTS"].copy()
bayes_topk_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_topk_alert_metrics.csv"), index=False)



# %% Cell 36
# ============================================================
# CELL 34: Credible-Interval and Uncertainty Triage Outputs
# ============================================================

def make_uncertainty_frame(base_df, pred_dict, selected_score=None):
    keep_cols = ["Timestamp", "sender_id", "receiver_id", "amount_paid", "payment_format", "label"]
    out = base_df[keep_cols].copy()
    out["posterior_mean"] = np.clip(pred_dict["mean"], EPS, 1 - EPS)
    out["posterior_median"] = np.clip(pred_dict["median"], EPS, 1 - EPS)
    out["posterior_eta_mean_prob"] = np.clip(pred_dict["eta_mean_prob"], EPS, 1 - EPS)
    out["posterior_sd"] = pred_dict["std"]
    out["posterior_lower_95"] = pred_dict["lower"]
    out["posterior_upper_95"] = pred_dict["upper"]
    out["credible_interval_width"] = out["posterior_upper_95"] - out["posterior_lower_95"]
    if selected_score is not None:
        out["selected_decision_score"] = selected_score
    sd_q75 = out["posterior_sd"].quantile(0.75)
    sd_q90 = out["posterior_sd"].quantile(0.90)
    out["uncertainty_band"] = np.select(
        [out["posterior_sd"] >= sd_q90, out["posterior_sd"] >= sd_q75],
        ["High", "Moderate"],
        default="Low"
    )
    out["priority_score"] = out.get("selected_decision_score", out["posterior_mean"]) + 0.15 * out["posterior_sd"]
    return out

val_select_uncertainty_df = make_uncertainty_frame(val_select_df_f, bayes_select_pred, selected_score=bayes_select_score)
test_uncertainty_df = make_uncertainty_frame(test_df_f, bayes_test_pred, selected_score=bayes_test_score)

display(test_uncertainty_df.sort_values("priority_score", ascending=False).head(20))
test_uncertainty_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_test_posterior_uncertainty_outputs.csv"), index=False)

uncertainty_summary = test_uncertainty_df.groupby(["label", "uncertainty_band"]).agg(
    count=("posterior_mean", "count"),
    mean_posterior_risk=("posterior_mean", "mean"),
    mean_selected_score=("selected_decision_score", "mean"),
    mean_posterior_sd=("posterior_sd", "mean"),
    mean_ci_width=("credible_interval_width", "mean")
).reset_index()
display(uncertainty_summary)
uncertainty_summary.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_uncertainty_band_summary.csv"), index=False)



# %% Cell 37
# ============================================================
# CELL 35: Final Model Comparison Table
# ============================================================

final_rows = []
comparison_specs = [
    ("Tabular XGBoost Calibrated", tab_test_cal, 0.5),
    ("BehaviorGraph-XGBoost Calibrated", bg_test_cal, 0.5),
    ("RUS-XGBoost Calibrated", rus_test_cal, 0.5),
    ("Isolation Forest Calibrated", if_test_cal, 0.5),
    ("Rule-Based AML Risk Calibrated", rule_test_cal, 0.5),
    ("Deterministic Logistic Fusion", meta_test_score, 0.5),
    ("BehaviorGraph-NUTS Selected Score", bayes_test_score, 0.5),
    ("BehaviorGraph-NUTS F2-Constrained Policy", bayes_test_score, float(best_f2_policy["Threshold"])),
    ("BehaviorGraph-NUTS MCC-Constrained Policy", bayes_test_score, float(best_mcc_policy["Threshold"]))
]

for name, score, threshold in comparison_specs:
    row = evaluate_predictions(name, "Test", y_test, score, threshold=threshold)
    for ar, label in [(0.001, "0.1%"), (0.005, "0.5%"), (0.01, "1%"), (0.02, "2%"), (0.05, "5%")]:
        top = top_k_alert_metrics(y_test, score, name, k_values=[ar]).iloc[0]
        row[f"Precision@{label}"] = top["Precision@k"]
        row[f"Recall@{label}"] = top["Recall@k"]
        row[f"TP@{label}"] = top["TP@k"]
    final_rows.append(row)

final_comparison_df = pd.DataFrame(final_rows)
display(final_comparison_df.round(5))
final_comparison_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_final_model_comparison.csv"), index=False)



# %% Cell 38
# ============================================================
# CELL 36: Calibration Reliability Diagram and Calibration Tables
# ============================================================

calibration_models = {
    "Tabular XGBoost": tab_test_cal,
    "BehaviorGraph-XGBoost": bg_test_cal,
    "Deterministic Fusion": meta_test_score,
    "BehaviorGraph-NUTS": bayes_test_score
}

plt.figure(figsize=(6, 6))
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
for name, score in calibration_models.items():
    _, bins_df = expected_calibration_error(y_test, score, n_bins=10)
    bins_df = bins_df.dropna(subset=["confidence", "empirical_rate"])
    plt.plot(bins_df["confidence"], bins_df["empirical_rate"], marker="o", label=name)
plt.xlabel("Mean predicted laundering risk")
plt.ylabel("Empirical laundering rate")
plt.title("Reliability Diagram")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_reliability_diagram.png"), dpi=300)
plt.show()

calibration_rows = []
for name, score in calibration_models.items():
    row = evaluate_predictions(name, "Test", y_test, score, threshold=0.5)
    calibration_rows.append({"Model": name, "Brier": row["Brier"], "Log Loss": row["Log Loss"], "ECE": row["ECE"], "ACE": row["ACE"], "MCE": row["MCE"], "AUPRC": row["AUPRC"], "AUROC": row["AUROC"]})

calibration_df = pd.DataFrame(calibration_rows)
display(calibration_df.round(6))
calibration_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_calibration_summary.csv"), index=False)


# Rare-class-focused calibration in the highest-risk alert bands.
def top_risk_band_calibration(y_true, score, model_name, rates=ALERT_RATES):
    rows = []
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)
    n = len(score)
    order = np.argsort(-score)
    total_pos = max(y_true.sum(), 1)
    for ar in rates:
        k = max(1, int(np.ceil(n * ar)))
        idx = order[:k]
        rows.append({
            "Model": model_name,
            "Alert Rate": ar,
            "Alerts": k,
            "Mean Predicted Risk": float(score[idx].mean()),
            "Observed Laundering Rate": float(y_true[idx].mean()),
            "Absolute Calibration Gap": float(abs(score[idx].mean() - y_true[idx].mean())),
            "Captured Laundering Cases": int(y_true[idx].sum()),
            "Recall@k": float(y_true[idx].sum() / total_pos),
        })
    return pd.DataFrame(rows)

top_band_calibration_df = pd.concat([
    top_risk_band_calibration(y_test, score, name, ALERT_RATES)
    for name, score in calibration_models.items()
], ignore_index=True)
display(top_band_calibration_df.round(6))
top_band_calibration_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_top_risk_band_calibration.csv"), index=False)

plt.figure(figsize=(8, 5))
for name in top_band_calibration_df["Model"].unique():
    sub = top_band_calibration_df[top_band_calibration_df["Model"] == name]
    plt.plot(sub["Alert Rate"] * 100, sub["Absolute Calibration Gap"], marker="o", label=name)
plt.xlabel("Top alert budget (%)")
plt.ylabel("Absolute calibration gap")
plt.title("Top-Risk-Band Calibration")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_top_risk_band_calibration.png"), dpi=300)
plt.show()



# %% Cell 39
# ============================================================
# CELL 37: ROC and Precision-Recall Curves
# ============================================================

curve_specs = {
    "Tabular XGBoost": tab_test_cal,
    "BehaviorGraph-XGBoost": bg_test_cal,
    "Deterministic Fusion": meta_test_score,
    "BehaviorGraph-NUTS": bayes_test_score
}

plt.figure(figsize=(7, 6))
for name, score in curve_specs.items():
    fpr, tpr, _ = roc_curve(y_test, score)
    auc = roc_auc_score(y_test, score)
    plt.plot(fpr, tpr, label=f"{name} (AUROC={auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_roc_curves.png"), dpi=300)
plt.show()

plt.figure(figsize=(7, 6))
for name, score in curve_specs.items():
    prec, rec, _ = precision_recall_curve(y_test, score)
    auprc = average_precision_score(y_test, score)
    plt.plot(rec, prec, label=f"{name} (AUPRC={auprc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curves")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_precision_recall_curves.png"), dpi=300)
plt.show()



# %% Cell 40
# ============================================================
# CELL 38: Posterior Risk, Uncertainty, and Top-k Figures
# ============================================================

plt.figure(figsize=(8, 5))
plt.hist(bayes_test_score[y_test == 0], bins=50, alpha=0.6, density=True, label="Non-laundering")
plt.hist(bayes_test_score[y_test == 1], bins=50, alpha=0.6, density=True, label="Laundering")
plt.xlabel("Selected BehaviorGraph-NUTS decision score")
plt.ylabel("Density")
plt.title("BehaviorGraph-NUTS Decision Score Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_decision_score_distribution.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(bayes_test_pred["std"][y_test == 0], bins=50, alpha=0.6, density=True, label="Non-laundering")
plt.hist(bayes_test_pred["std"][y_test == 1], bins=50, alpha=0.6, density=True, label="Laundering")
plt.xlabel("Posterior standard deviation")
plt.ylabel("Density")
plt.title("BehaviorGraph-NUTS Posterior Uncertainty Distribution")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_posterior_uncertainty_distribution.png"), dpi=300)
plt.show()

plt.figure(figsize=(8, 5))
topk_plot = bayes_topk_df.copy()
plt.plot(topk_plot["Alert Rate"] * 100, topk_plot["Precision@k"], marker="o", label="Precision@k")
plt.plot(topk_plot["Alert Rate"] * 100, topk_plot["Recall@k"], marker="o", label="Recall@k")
plt.xlabel("Alert budget (%)")
plt.ylabel("Metric value")
plt.title("BehaviorGraph-NUTS Top-k Alert Performance")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_topk_alert_performance.png"), dpi=300)
plt.show()



# %% Cell 41
# ============================================================
# CELL 39: NUTS Trace and Posterior Density Plots for Representative Parameters
# ============================================================

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# Select representative parameters from the largest absolute posterior means.
coef_for_rep = coef_df.copy()
coef_for_rep["AbsPosteriorMean"] = coef_for_rep["Posterior Mean"].abs()

representative_features = (
    coef_for_rep
    .sort_values("AbsPosteriorMean", ascending=False)["Feature"]
    .head(REPRESENTATIVE_PARAM_COUNT)
    .tolist()
)

print("Representative NUTS beta parameters:", representative_features)


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def _flatten_axes(ax_obj):
    """
    Converts ArviZ/matplotlib axis output into a flat list of axes.
    Handles single Axes, numpy arrays of Axes, and nested axis structures.
    """
    if ax_obj is None:
        return []
    if isinstance(ax_obj, np.ndarray):
        return list(ax_obj.ravel())
    if isinstance(ax_obj, (list, tuple)):
        flat = []
        for item in ax_obj:
            flat.extend(_flatten_axes(item))
        return flat
    return [ax_obj]


def get_num_chains(idata_obj):
    """
    Safely retrieves number of NUTS chains from an ArviZ InferenceData object.
    """
    try:
        return int(idata_obj.posterior.sizes["chain"])
    except Exception:
        return 1


def make_chain_legend(n_chains):
    """
    Creates a clear chain legend explaining that each color corresponds
    to one independent NUTS chain.
    """
    # Use matplotlib default cycle so legend colors align with ArviZ/matplotlib defaults.
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    chain_handles = []
    for c in range(n_chains):
        chain_handles.append(
            Line2D(
                [0], [0],
                color=default_colors[c % len(default_colors)],
                lw=2,
                label=f"Chain {c + 1}"
            )
        )

    return chain_handles


def add_chain_legend(ax_obj, n_chains, loc="upper right"):
    """
    Adds chain legend to trace plot axes.
    """
    chain_handles = make_chain_legend(n_chains)

    for ax in _flatten_axes(ax_obj):
        ax.legend(
            handles=chain_handles,
            title="NUTS chains\n(color = independent chain)",
            loc=loc,
            fontsize=8,
            title_fontsize=8,
            frameon=True
        )


def add_density_legend(ax_obj, loc="upper right"):
    """
    Adds explanatory legend for posterior density plots.
    """
    density_handles = [
        Patch(
            facecolor="gray",
            edgecolor="gray",
            alpha=0.35,
            label="Posterior density / 95% HDI region"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            lw=1.5,
            label="Zero reference"
        )
    ]

    for ax in _flatten_axes(ax_obj):
        ax.legend(
            handles=density_handles,
            title="Legend",
            loc=loc,
            fontsize=8,
            title_fontsize=8,
            frameon=True
        )


def add_forest_legend(ax_obj, loc="lower right"):
    """
    Adds explanatory legend for forest plots.
    """
    forest_handles = [
        Line2D(
            [0], [0],
            color="black",
            lw=2,
            label="Posterior mean"
        ),
        Line2D(
            [0], [0],
            color="black",
            lw=1,
            marker="|",
            markersize=10,
            label="95% HDI interval"
        ),
        Line2D(
            [0], [0],
            color="black",
            linestyle="--",
            lw=1.5,
            label="Zero reference"
        )
    ]

    for ax in _flatten_axes(ax_obj):
        ax.legend(
            handles=forest_handles,
            title="Posterior interval legend",
            loc=loc,
            fontsize=8,
            title_fontsize=8,
            frameon=True
        )


n_chains = get_num_chains(idata)
print(f"Detected NUTS chains: {n_chains}")


# ------------------------------------------------------------
# 1. Trace plot for alpha
# ------------------------------------------------------------
trace_alpha_axes = az.plot_trace(
    idata,
    var_names=["alpha"],
    compact=False
)

for ax in _flatten_axes(trace_alpha_axes):
    ax.set_title("Trace diagnostics for intercept (alpha)")

add_chain_legend(trace_alpha_axes, n_chains=n_chains, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_trace_alpha.png"), dpi=300)
plt.show()


# ------------------------------------------------------------
# 2. Trace plots for representative beta coefficients
# ------------------------------------------------------------
if len(representative_features) > 0:
    trace_beta_axes = az.plot_trace(
        idata,
        var_names=["beta"],
        coords={"feature": representative_features},
        compact=False
    )

    for ax in _flatten_axes(trace_beta_axes):
        ax.set_title("Trace diagnostics for representative beta coefficients")

    add_chain_legend(trace_beta_axes, n_chains=n_chains, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_trace_representative_betas.png"), dpi=300)
    plt.show()


# ------------------------------------------------------------
# 3. Posterior density plot for alpha
# ------------------------------------------------------------
density_alpha_axes = az.plot_posterior(
    idata,
    var_names=["alpha"],
    hdi_prob=0.95
)

for ax in _flatten_axes(density_alpha_axes):
    ax.set_title("Posterior density for intercept (alpha)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)

add_density_legend(density_alpha_axes, loc="upper right")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_density_alpha.png"), dpi=300)
plt.show()


# ------------------------------------------------------------
# 4. Posterior density plots for representative beta coefficients
# ------------------------------------------------------------
if len(representative_features) > 0:
    density_beta_axes = az.plot_posterior(
        idata,
        var_names=["beta"],
        coords={"feature": representative_features},
        hdi_prob=0.95
    )

    for ax, feature_name in zip(_flatten_axes(density_beta_axes), representative_features):
        ax.set_title(f"Posterior density: {feature_name}")
        ax.axvline(0, color="black", linestyle="--", linewidth=1.5)

    add_density_legend(density_beta_axes, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_density_representative_betas.png"), dpi=300)
    plt.show()


# ------------------------------------------------------------
# 5. Forest plot for all retained Bayesian fusion coefficients
# ------------------------------------------------------------
forest_axes = az.plot_forest(
    idata,
    var_names=["beta"],
    combined=True,
    hdi_prob=0.95
)

for ax in _flatten_axes(forest_axes):
    ax.axvline(0, color="black", linestyle="--", linewidth=1.5)

add_forest_legend(forest_axes, loc="lower right")

plt.title("Posterior Intervals for Bayesian Fusion Coefficients")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_beta_forest_plot.png"), dpi=300)
plt.show()


# %% Cell 42
# ============================================================
# CELL 40: R-hat, Bulk ESS, and Tail ESS Diagnostic Plots
# ============================================================

diag_df = nuts_summary.reset_index().rename(columns={"index": "Parameter"})
# Make parameter names cleaner for beta features.
diag_df["Parameter"] = diag_df["Parameter"].astype(str)

display(diag_df.round(4))

if "r_hat" in diag_df.columns:
    rhat_plot_df = diag_df.sort_values("r_hat", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(rhat_plot_df["Parameter"], rhat_plot_df["r_hat"])
    plt.axhline(1.01, linestyle="--", label="1.01 reference")
    plt.axhline(1.05, linestyle=":", label="1.05 warning")
    plt.xticks(rotation=90)
    plt.ylabel("R-hat")
    plt.title("NUTS R-hat Diagnostic Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_rhat_diagnostic_plot.png"), dpi=300)
    plt.show()

if "ess_bulk" in diag_df.columns and "ess_tail" in diag_df.columns:
    ess_plot_df = diag_df.sort_values("ess_bulk", ascending=True)
    x_pos = np.arange(len(ess_plot_df))
    plt.figure(figsize=(10, 5))
    plt.bar(x_pos - 0.2, ess_plot_df["ess_bulk"], width=0.4, label="Bulk ESS")
    plt.bar(x_pos + 0.2, ess_plot_df["ess_tail"], width=0.4, label="Tail ESS")
    plt.axhline(400, linestyle="--", label="ESS=400 reference")
    plt.axhline(100, linestyle=":", label="ESS=100 warning")
    plt.xticks(x_pos, ess_plot_df["Parameter"], rotation=90)
    plt.ylabel("Effective sample size")
    plt.title("NUTS Bulk ESS and Tail ESS Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_bulk_tail_ess_plot.png"), dpi=300)
    plt.show()



# %% Cell 43
# ============================================================
# CELL 40B: Subset-Size Convergence Diagnostics
# ============================================================

# This lightweight diagnostic checks whether the fusion-stage performance stabilizes as the
# case-control training subset increases. It uses the deterministic fusion path by default so
# it can run quickly. Optional short-NUTS subset diagnostics are available but disabled by default.

subset_convergence_rows = []
if RUN_SUBSET_SIZE_CONVERGENCE_DIAGNOSTIC:
    sizes = [s for s in SUBSET_CONVERGENCE_SIZES if s <= len(y_val_fit)]
    sizes = sizes[:SUBSET_CONVERGENCE_MAX_SIZES]
    print("Subset-size convergence sizes:", sizes)

    for size in sizes:
        X_sub_i, y_sub_i, w_sub_i, idx_i = stratified_weighted_case_control_sample(
            X_fusion_fit, y_val_fit, max_rows=size, seed=SEED + int(size)
        )
        lr_i = LogisticRegression(C=best_meta_c, penalty="l2", solver="lbfgs", max_iter=4000)
        lr_i.fit(X_sub_i, y_sub_i, sample_weight=w_sub_i)
        sel_i = lr_i.predict_proba(X_fusion_select)[:, 1]
        test_i = lr_i.predict_proba(X_fusion_test)[:, 1]
        sel_metrics = evaluate_predictions(f"SubsetFusion_{size}", "Validation-Select", y_val_select, sel_i, threshold=0.5)
        test_metrics = evaluate_predictions(f"SubsetFusion_{size}", "Test", y_test, test_i, threshold=0.5)
        top_sel = top_k_alert_metrics(y_val_select, sel_i, f"SubsetFusion_{size}", k_values=[0.01]).iloc[0]
        top_test = top_k_alert_metrics(y_test, test_i, f"SubsetFusion_{size}", k_values=[0.01]).iloc[0]
        subset_convergence_rows.append({
            "Subset Size": size,
            "Subset Positive Rate": float(y_sub_i.mean()),
            "Validation-Select AUROC": sel_metrics["AUROC"],
            "Validation-Select AUPRC": sel_metrics["AUPRC"],
            "Validation-Select Brier": sel_metrics["Brier"],
            "Validation-Select ECE": sel_metrics["ECE"],
            "Validation-Select Recall@1%": top_sel["Recall@k"],
            "Test AUROC": test_metrics["AUROC"],
            "Test AUPRC": test_metrics["AUPRC"],
            "Test Brier": test_metrics["Brier"],
            "Test ECE": test_metrics["ECE"],
            "Test Recall@1%": top_test["Recall@k"],
        })

    subset_convergence_df = pd.DataFrame(subset_convergence_rows)
    display(subset_convergence_df.round(6))
    subset_convergence_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_nuts_subset_size_convergence_diagnostics.csv"), index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(subset_convergence_df["Subset Size"], subset_convergence_df["Validation-Select AUPRC"], marker="o", label="Validation AUPRC")
    plt.plot(subset_convergence_df["Subset Size"], subset_convergence_df["Test AUPRC"], marker="o", label="Test AUPRC")
    plt.xlabel("Fusion training subset size")
    plt.ylabel("AUPRC")
    plt.title("Subset-Size Convergence Diagnostics: AUPRC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_subset_size_convergence_auprc.png"), dpi=300)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(subset_convergence_df["Subset Size"], subset_convergence_df["Validation-Select Recall@1%"], marker="o", label="Validation Recall@1%")
    plt.plot(subset_convergence_df["Subset Size"], subset_convergence_df["Test Recall@1%"], marker="o", label="Test Recall@1%")
    plt.xlabel("Fusion training subset size")
    plt.ylabel("Recall@1%")
    plt.title("Subset-Size Convergence Diagnostics: Recall@1%")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "behaviorgraph_nuts_subset_size_convergence_recall1.png"), dpi=300)
    plt.show()
else:
    print("RUN_SUBSET_SIZE_CONVERGENCE_DIAGNOSTIC is False.")

# Optional short-NUTS subset diagnostics are intentionally left off for speed.
if RUN_SHORT_NUTS_SUBSET_DIAGNOSTIC:
    print("Short-NUTS subset diagnostic is enabled. This may add considerable runtime.")
else:
    print("RUN_SHORT_NUTS_SUBSET_DIAGNOSTIC is False. Deterministic subset convergence diagnostic was used.")



# %% Cell 44
# ============================================================
# CELL 41: Five-Seed Deterministic Robustness Evaluation
# ============================================================

five_seed_rows = []
if RUN_FIVE_SEED_DETERMINISTIC_EVAL:
    for seed in RANDOM_SEEDS:
        set_global_seed(seed)
        spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        params = dict(behaviorgraph_xgb_params)
        params["random_state"] = seed
        params["scale_pos_weight"] = spw
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_bg_selected, y_train)
        score = model.predict_proba(X_test_bg_selected)[:, 1]
        row = evaluate_predictions("BehaviorGraph-XGBoost", "Test", y_test, score, threshold=0.5)
        row["Seed"] = seed
        five_seed_rows.append(row)

    five_seed_df = pd.DataFrame(five_seed_rows)
    display(five_seed_df.round(5))

    summary_cols = ["Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "F2", "MCC", "AUROC", "AUPRC", "Brier", "ECE", "ACE", "MCE"]
    five_seed_summary = five_seed_df[summary_cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "Metric"})
    display(five_seed_summary.round(6))
    five_seed_df.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_xgboost_five_seed_metrics.csv"), index=False)
    five_seed_summary.to_csv(os.path.join(TABLE_DIR, "behaviorgraph_xgboost_five_seed_summary.csv"), index=False)
else:
    print("RUN_FIVE_SEED_DETERMINISTIC_EVAL is False.")


# %% Cell 45
# ============================================================
# CELL 42: Five-Seed Evaluation Configuration and Helper Functions
# ============================================================

# Purpose:
# Run five-seed robustness evaluation across the main models:
# 1. Tabular XGBoost
# 2. BehaviorGraph-XGBoost
# 3. RUS-XGBoost
# 4. Deterministic Logistic Fusion
# 5. BehaviorGraph-NUTS
#
# The model design is now fixed. This section varies random seeds only.

RUN_FULL_FIVE_SEED_EVALUATION = True

FIVE_SEED_LIST = RANDOM_SEEDS if "RANDOM_SEEDS" in globals() else [42, 123, 202, 777, 999]

# Use the same NUTS settings as the current best single-run model.
FIVE_SEED_NUTS_DRAWS = NUTS_DRAWS
FIVE_SEED_NUTS_TUNE = NUTS_TUNE
FIVE_SEED_NUTS_CHAINS = NUTS_CHAINS
FIVE_SEED_NUTS_CORES = NUTS_CORES
FIVE_SEED_NUTS_TARGET_ACCEPT = NUTS_TARGET_ACCEPT

# Set this to False only for debugging. For final results, keep True.
RUN_NUTS_WITHIN_FIVE_SEED = True

# If runtime becomes excessive, set this to a smaller value temporarily.
FIVE_SEED_NUTS_MAX_FUSION_ROWS = NUTS_MAX_FUSION_ROWS

print("Five-seed evaluation seeds:", FIVE_SEED_LIST)
print("NUTS inside five-seed evaluation:", RUN_NUTS_WITHIN_FIVE_SEED)
print("Five-seed NUTS draws:", FIVE_SEED_NUTS_DRAWS)
print("Five-seed NUTS tune:", FIVE_SEED_NUTS_TUNE)
print("Five-seed NUTS chains:", FIVE_SEED_NUTS_CHAINS)
print("Five-seed NUTS target_accept:", FIVE_SEED_NUTS_TARGET_ACCEPT)


def clone_xgb_params(params, seed):
    """
    Copy XGBoost parameters and update the seed-related fields.
    """
    p = dict(params)
    p["random_state"] = seed
    p["n_jobs"] = -1
    if "tree_method" not in p:
        p["tree_method"] = "hist"
    if "objective" not in p:
        p["objective"] = "binary:logistic"
    if "eval_metric" not in p:
        p["eval_metric"] = "aucpr"
    return p


def fit_seed_tabular_xgb(seed):
    """
    Fit Tabular XGBoost using the already-selected best hyperparameters.
    """
    params = clone_xgb_params(tabular_xgb_params, seed)
    params["scale_pos_weight"] = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_base, y_train)

    val_score = model.predict_proba(X_val_base)[:, 1]
    test_score = model.predict_proba(X_test_base)[:, 1]

    return model, val_score, val_score[val_fit_pos], val_score[val_select_pos], test_score


def fit_seed_behaviorgraph_xgb(seed):
    """
    Fit BehaviorGraph-XGBoost using the selected BehaviorGraph feature set
    and fixed best hyperparameters.
    """
    params = clone_xgb_params(behaviorgraph_xgb_params, seed)
    params["scale_pos_weight"] = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_bg_selected, y_train)

    val_score = model.predict_proba(X_val_bg_selected)[:, 1]
    test_score = model.predict_proba(X_test_bg_selected)[:, 1]

    return model, val_score, val_score[val_fit_pos], val_score[val_select_pos], test_score


def fit_seed_rus_xgb(seed):
    """
    Fit RUS-XGBoost using the selected best RUS configuration.
    """
    rus_sampler = RandomUnderSampler(
        random_state=seed,
        sampling_strategy=best_rus_cfg["sampling_strategy"]
    )

    X_train_rus_seed, y_train_rus_seed = rus_sampler.fit_resample(
        X_train_bg_selected,
        y_train
    )

    model = xgb.XGBClassifier(
        n_estimators=best_rus_cfg["n_estimators"],
        max_depth=best_rus_cfg["max_depth"],
        learning_rate=best_rus_cfg["learning_rate"],
        subsample=0.90,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=best_rus_cfg["reg_lambda"],
        reg_alpha=best_rus_cfg["reg_alpha"],
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
        max_delta_step=1
    )

    model.fit(X_train_rus_seed, y_train_rus_seed)

    val_score = model.predict_proba(X_val_bg_selected)[:, 1]
    test_score = model.predict_proba(X_test_bg_selected)[:, 1]

    return model, val_score, val_score[val_fit_pos], val_score[val_select_pos], test_score


def fit_seed_iforest(seed):
    """
    Fit Isolation Forest for the seed using the selected/tuned iforest configuration.
    This remains an auxiliary evidence stream for fusion.
    """
    benign_mask_seed = (y_train == 0)
    X_iforest_full_seed = X_train_bg_selected_scaled[benign_mask_seed]

    if "IFOREST_MAX_ROWS" not in globals():
        max_rows = 300000
    else:
        max_rows = IFOREST_MAX_ROWS

    if len(X_iforest_full_seed) > max_rows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_iforest_full_seed), size=max_rows, replace=False)
        X_iforest_fit_seed = X_iforest_full_seed[idx]
    else:
        X_iforest_fit_seed = X_iforest_full_seed

    # Reuse best fitted iforest parameters if available.
    if "iforest" in globals():
        if_params = iforest.get_params()
        if_params["random_state"] = seed
        if_params["n_jobs"] = -1
        model = IsolationForest(**if_params)
    else:
        model = IsolationForest(
            n_estimators=350,
            max_samples=0.70,
            contamination=max(y_train.mean() * 2, 1e-4),
            random_state=seed,
            n_jobs=-1
        )

    model.fit(X_iforest_fit_seed)

    val_raw = -model.decision_function(X_val_bg_selected_scaled)
    test_raw = -model.decision_function(X_test_bg_selected_scaled)

    scaler_seed = MinMaxScaler()
    scaler_seed.fit(val_raw[val_fit_pos].reshape(-1, 1))

    val_score = np.clip(scaler_seed.transform(val_raw.reshape(-1, 1)).ravel(), 0, 1)
    test_score = np.clip(scaler_seed.transform(test_raw.reshape(-1, 1)).ravel(), 0, 1)

    return model, val_score, val_score[val_fit_pos], val_score[val_select_pos], test_score


def calibrate_seed_evidence(seed_scores):
    """
    Calibrate evidence streams using validation-fit only.
    """
    calibrated = {}
    calibrators_seed = {}

    for name, score_dict in seed_scores.items():
        cal = fit_platt_calibrator(score_dict["val_fit"], y_val_fit)
        calibrators_seed[name] = cal

        full_val_cal = apply_platt_calibrator(cal, score_dict["val"])
        test_cal = apply_platt_calibrator(cal, score_dict["test"])

        calibrated[name] = {
            "val": full_val_cal,
            "val_fit": full_val_cal[val_fit_pos],
            "val_select": full_val_cal[val_select_pos],
            "test": test_cal
        }

    return calibrated, calibrators_seed


def build_seed_fusion_matrices(calibrated_seed):
    """
    Build, select, impute, and scale fusion matrices using the already-fixed
    final fusion_feature_names from the current best model.
    """
    tab_fit = calibrated_seed["Tabular XGBoost"]["val_fit"]
    tab_select = calibrated_seed["Tabular XGBoost"]["val_select"]
    tab_test = calibrated_seed["Tabular XGBoost"]["test"]

    bg_fit = calibrated_seed["BehaviorGraph-XGBoost"]["val_fit"]
    bg_select = calibrated_seed["BehaviorGraph-XGBoost"]["val_select"]
    bg_test = calibrated_seed["BehaviorGraph-XGBoost"]["test"]

    rus_fit = calibrated_seed["RUS-XGBoost"]["val_fit"]
    rus_select = calibrated_seed["RUS-XGBoost"]["val_select"]
    rus_test = calibrated_seed["RUS-XGBoost"]["test"]

    if_fit = calibrated_seed["Isolation Forest"]["val_fit"]
    if_select = calibrated_seed["Isolation Forest"]["val_select"]
    if_test = calibrated_seed["Isolation Forest"]["test"]

    rule_fit = calibrated_seed["Rule-Based AML Risk"]["val_fit"]
    rule_select = calibrated_seed["Rule-Based AML Risk"]["val_select"]
    rule_test = calibrated_seed["Rule-Based AML Risk"]["test"]

    fit_df = build_fusion_frame(
        val_fit_df_f,
        tab_fit,
        bg_fit,
        rus_fit,
        if_fit,
        rule_fit
    )

    select_df = build_fusion_frame(
        val_select_df_f,
        tab_select,
        bg_select,
        rus_select,
        if_select,
        rule_select
    )

    test_df = build_fusion_frame(
        test_df_f,
        tab_test,
        bg_test,
        rus_test,
        if_test,
        rule_test
    )

    # Use fixed final fusion feature list from the current best model.
    fit_df = fit_df[fusion_feature_names].copy()
    select_df = select_df[fusion_feature_names].copy()
    test_df = test_df[fusion_feature_names].copy()

    imp = SimpleImputer(strategy="median")
    scaler_seed = StandardScaler()

    X_fit_imp = imp.fit_transform(fit_df)
    X_select_imp = imp.transform(select_df)
    X_test_imp = imp.transform(test_df)

    X_fit_scaled = scaler_seed.fit_transform(X_fit_imp)
    X_select_scaled = scaler_seed.transform(X_select_imp)
    X_test_scaled = scaler_seed.transform(X_test_imp)

    return X_fit_scaled, X_select_scaled, X_test_scaled, imp, scaler_seed


def fit_seed_deterministic_fusion(X_fit, X_select, X_test, seed):
    """
    Fit deterministic logistic fusion using the fixed best_meta_c if available.
    Uses the same weighted case-control fitting logic as the main pipeline.
    """
    meta_c_seed = best_meta_c if "best_meta_c" in globals() else 1.0

    X_sub, y_sub, w_sub, _ = stratified_weighted_case_control_sample(
        X_fit,
        y_val_fit,
        max_rows=NUTS_MAX_FUSION_ROWS,
        seed=seed + 250
    )

    model = LogisticRegression(
        C=meta_c_seed,
        penalty="l2",
        solver="lbfgs",
        max_iter=4000,
        random_state=seed
    )

    model.fit(X_sub, y_sub, sample_weight=w_sub)

    select_score = model.predict_proba(X_select)[:, 1]
    test_score = model.predict_proba(X_test)[:, 1]

    return model, select_score, test_score


def fit_seed_nuts_fusion(X_fit, X_select, X_test, meta_model, seed):
    """
    Fit full BehaviorGraph-NUTS for a given seed using the fixed full fusion feature set.
    Returns selected calibrated Bayesian score, posterior prediction objects, idata, and diagnostics.
    """
    X_nuts_seed, y_nuts_seed, w_nuts_seed, _ = stratified_weighted_case_control_sample(
        X_fit,
        y_val_fit,
        max_rows=FIVE_SEED_NUTS_MAX_FUSION_ROWS,
        seed=seed + 500
    )

    init_alpha_seed = float(meta_model.intercept_[0])
    init_beta_seed = meta_model.coef_.reshape(-1).astype(float)

    if USE_ANCHORED_NUTS_PRIOR:
        prior_alpha_mu_seed = init_alpha_seed
        prior_beta_mu_seed = init_beta_seed
        prior_alpha_sigma_seed = ANCHOR_ALPHA_SIGMA
        prior_beta_sigma_seed = ANCHOR_BETA_SIGMA
    else:
        prior_alpha_mu_seed = 0.0
        prior_beta_mu_seed = np.zeros(X_fit.shape[1])
        prior_alpha_sigma_seed = 2.0
        prior_beta_sigma_seed = 1.0

    with pm.Model(coords={"feature": fusion_feature_names}) as seed_nuts_model:
        X_data = pm.Data("X_data", X_nuts_seed.astype("float64"))
        y_data = pm.Data("y_data", y_nuts_seed.astype("int64"))
        w_data = pm.Data("w_data", w_nuts_seed.astype("float64"))

        alpha = pm.Normal(
            "alpha",
            mu=prior_alpha_mu_seed,
            sigma=prior_alpha_sigma_seed
        )

        beta = pm.Normal(
            "beta",
            mu=prior_beta_mu_seed,
            sigma=prior_beta_sigma_seed,
            dims="feature"
        )

        eta = alpha + pm.math.dot(X_data, beta)
        log_lik_vec = pm.logp(pm.Bernoulli.dist(logit_p=eta), y_data)
        pm.Potential("weighted_likelihood", pm.math.sum(w_data * log_lik_vec))

        idata_seed = pm.sample(
            draws=FIVE_SEED_NUTS_DRAWS,
            tune=FIVE_SEED_NUTS_TUNE,
            chains=FIVE_SEED_NUTS_CHAINS,
            cores=FIVE_SEED_NUTS_CORES,
            target_accept=FIVE_SEED_NUTS_TARGET_ACCEPT,
            init="jitter+adapt_diag",
            initvals={"alpha": init_alpha_seed, "beta": init_beta_seed},
            random_seed=seed,
            return_inferencedata=True,
            progressbar=True
        )

    # Diagnostics.
    summary_seed = az.summary(idata_seed, var_names=["alpha", "beta"], round_to=4)
    divergences_seed = int(idata_seed.sample_stats["diverging"].sum().values)
    max_rhat_seed = float(summary_seed["r_hat"].max()) if "r_hat" in summary_seed.columns else np.nan
    min_bulk_seed = float(summary_seed["ess_bulk"].min()) if "ess_bulk" in summary_seed.columns else np.nan
    min_tail_seed = float(summary_seed["ess_tail"].min()) if "ess_tail" in summary_seed.columns else np.nan

    # Posterior predictive variants.
    select_pred_seed = posterior_predict_fusion(X_select, idata_seed)
    test_pred_seed = posterior_predict_fusion(X_test, idata_seed)

    select_variants_seed = build_bayes_score_variants(select_pred_seed)
    test_variants_seed = build_bayes_score_variants(test_pred_seed)

    variant_rows_seed = []

    for variant_name, score_select_uncal in select_variants_seed.items():
        score_select_uncal = np.clip(score_select_uncal, EPS, 1 - EPS)

        row = evaluate_predictions(
            f"BehaviorGraph-NUTS {variant_name}",
            "Validation-Select",
            y_val_select,
            score_select_uncal,
            threshold=0.5
        )
        row["Variant"] = variant_name

        for ar in [0.001, 0.005, 0.01, 0.02]:
            top = top_k_alert_metrics(
                y_val_select,
                score_select_uncal,
                variant_name,
                k_values=[ar]
            ).iloc[0]
            row[f"Recall@{ar*100:.1f}%"] = top["Recall@k"]
            row[f"Precision@{ar*100:.1f}%"] = top["Precision@k"]

        variant_rows_seed.append(row)

    variant_df_seed = pd.DataFrame(variant_rows_seed)

    best_auprc_seed = variant_df_seed["AUPRC"].max()
    near_best_seed = variant_df_seed[
        variant_df_seed["AUPRC"] >= best_auprc_seed - SCORE_AUPRC_TIE_TOL
    ].copy()

    near_best_seed["AUPRC_rank"] = near_best_seed["AUPRC"].rank(ascending=False, method="min")
    near_best_seed["Brier_rank"] = near_best_seed["Brier"].rank(ascending=True, method="min")
    near_best_seed["ECE_rank"] = near_best_seed["ECE"].rank(ascending=True, method="min")
    near_best_seed["Recall1_rank"] = near_best_seed["Recall@1.0%"].rank(ascending=False, method="min")

    near_best_seed["CompositeSelectionRank"] = (
        0.45 * near_best_seed["AUPRC_rank"]
        + 0.25 * near_best_seed["Brier_rank"]
        + 0.15 * near_best_seed["ECE_rank"]
        + 0.15 * near_best_seed["Recall1_rank"]
    )

    variant_df_seed = variant_df_seed.merge(
        near_best_seed[["Variant", "CompositeSelectionRank"]],
        on="Variant",
        how="left"
    )

    variant_df_seed["CompositeSelectionRank"] = variant_df_seed["CompositeSelectionRank"].fillna(9999)

    variant_df_seed = variant_df_seed.sort_values(
        ["CompositeSelectionRank", "AUPRC", "Recall@1.0%"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    selected_variant_seed = variant_df_seed.loc[0, "Variant"]

    select_uncal_seed = np.clip(select_variants_seed[selected_variant_seed], EPS, 1 - EPS)
    test_uncal_seed = np.clip(test_variants_seed[selected_variant_seed], EPS, 1 - EPS)

    final_calibrator_seed = fit_platt_calibrator(select_uncal_seed, y_val_select)

    select_score_seed = apply_platt_calibrator(final_calibrator_seed, select_uncal_seed)
    test_score_seed = apply_platt_calibrator(final_calibrator_seed, test_uncal_seed)

    diagnostics_seed = {
        "Divergences": divergences_seed,
        "Max Rhat": max_rhat_seed,
        "Min Bulk ESS": min_bulk_seed,
        "Min Tail ESS": min_tail_seed,
        "Selected Variant": selected_variant_seed,
        "NUTS Subset Size": int(len(y_nuts_seed)),
        "NUTS Subset Positive Rate": float(y_nuts_seed.mean())
    }

    return test_score_seed, select_score_seed, idata_seed, diagnostics_seed, variant_df_seed


# %% Cell 46
# ============================================================
# CELL 43: Run Five-Seed Evaluation Across Main Models
# ============================================================

five_seed_metric_rows = []
five_seed_topk_rows = []
five_seed_nuts_diagnostic_rows = []
five_seed_variant_rows = []

if RUN_FULL_FIVE_SEED_EVALUATION:
    for seed in FIVE_SEED_LIST:
        print("\n" + "=" * 100)
        print(f"Starting five-seed evaluation for seed {seed}")
        print("=" * 100)

        set_global_seed(seed)

        # ----------------------------------------------------
        # 1. Fit seed-specific evidence models
        # ----------------------------------------------------
        tab_model_s, tab_val_s, tab_val_fit_s, tab_val_select_s, tab_test_s = fit_seed_tabular_xgb(seed)
        bg_model_s, bg_val_s, bg_val_fit_s, bg_val_select_s, bg_test_s = fit_seed_behaviorgraph_xgb(seed)
        rus_model_s, rus_val_s, rus_val_fit_s, rus_val_select_s, rus_test_s = fit_seed_rus_xgb(seed)
        if_model_s, if_val_s, if_val_fit_s, if_val_select_s, if_test_s = fit_seed_iforest(seed)

        rule_val_s = rule_val_score.copy()
        rule_val_fit_s = rule_val_s[val_fit_pos]
        rule_val_select_s = rule_val_s[val_select_pos]
        rule_test_s = rule_test_score.copy()

        seed_raw_scores = {
            "Tabular XGBoost": {
                "val": tab_val_s,
                "val_fit": tab_val_fit_s,
                "val_select": tab_val_select_s,
                "test": tab_test_s
            },
            "BehaviorGraph-XGBoost": {
                "val": bg_val_s,
                "val_fit": bg_val_fit_s,
                "val_select": bg_val_select_s,
                "test": bg_test_s
            },
            "RUS-XGBoost": {
                "val": rus_val_s,
                "val_fit": rus_val_fit_s,
                "val_select": rus_val_select_s,
                "test": rus_test_s
            },
            "Isolation Forest": {
                "val": if_val_s,
                "val_fit": if_val_fit_s,
                "val_select": if_val_select_s,
                "test": if_test_s
            },
            "Rule-Based AML Risk": {
                "val": rule_val_s,
                "val_fit": rule_val_fit_s,
                "val_select": rule_val_select_s,
                "test": rule_test_s
            }
        }

        # ----------------------------------------------------
        # 2. Calibrate evidence streams
        # ----------------------------------------------------
        seed_calibrated, seed_calibrators = calibrate_seed_evidence(seed_raw_scores)

        tab_test_cal_s = seed_calibrated["Tabular XGBoost"]["test"]
        bg_test_cal_s = seed_calibrated["BehaviorGraph-XGBoost"]["test"]
        rus_test_cal_s = seed_calibrated["RUS-XGBoost"]["test"]
        if_test_cal_s = seed_calibrated["Isolation Forest"]["test"]
        rule_test_cal_s = seed_calibrated["Rule-Based AML Risk"]["test"]

        # ----------------------------------------------------
        # 3. Build seed-specific fusion matrices
        # ----------------------------------------------------
        X_fit_s, X_select_s, X_test_s, seed_fusion_imputer, seed_fusion_scaler = build_seed_fusion_matrices(
            seed_calibrated
        )

        # ----------------------------------------------------
        # 4. Deterministic fusion
        # ----------------------------------------------------
        meta_model_s, meta_select_s, meta_test_s = fit_seed_deterministic_fusion(
            X_fit_s,
            X_select_s,
            X_test_s,
            seed
        )

        # ----------------------------------------------------
        # 5. Full BehaviorGraph-NUTS
        # ----------------------------------------------------
        if RUN_NUTS_WITHIN_FIVE_SEED:
            nuts_test_s, nuts_select_s, idata_s, nuts_diag_s, variant_df_s = fit_seed_nuts_fusion(
                X_fit_s,
                X_select_s,
                X_test_s,
                meta_model_s,
                seed
            )

            nuts_diag_s["Seed"] = seed
            five_seed_nuts_diagnostic_rows.append(nuts_diag_s)

            variant_df_s = variant_df_s.copy()
            variant_df_s["Seed"] = seed
            five_seed_variant_rows.append(variant_df_s)

        else:
            nuts_test_s = np.full_like(y_test, fill_value=np.nan, dtype=float)
            nuts_diag_s = {
                "Seed": seed,
                "Divergences": np.nan,
                "Max Rhat": np.nan,
                "Min Bulk ESS": np.nan,
                "Min Tail ESS": np.nan,
                "Selected Variant": "NUTS not run",
                "NUTS Subset Size": np.nan,
                "NUTS Subset Positive Rate": np.nan
            }
            five_seed_nuts_diagnostic_rows.append(nuts_diag_s)

        # ----------------------------------------------------
        # 6. Collect metrics
        # ----------------------------------------------------
        seed_model_scores = {
            "Tabular XGBoost": tab_test_cal_s,
            "BehaviorGraph-XGBoost": bg_test_cal_s,
            "RUS-XGBoost": rus_test_cal_s,
            "Deterministic Logistic Fusion": meta_test_s
        }

        if RUN_NUTS_WITHIN_FIVE_SEED:
            seed_model_scores["BehaviorGraph-NUTS"] = nuts_test_s

        for model_name, score in seed_model_scores.items():
            row = evaluate_predictions(
                model_name,
                "Test",
                y_test,
                score,
                threshold=0.5
            )
            row["Seed"] = seed
            five_seed_metric_rows.append(row)

            topk_df_s = top_k_alert_metrics(
                y_test,
                score,
                model_name,
                k_values=ALERT_RATES
            )
            topk_df_s["Seed"] = seed
            five_seed_topk_rows.append(topk_df_s)

        print(f"Completed seed {seed}")

    five_seed_metrics_df = pd.DataFrame(five_seed_metric_rows)
    five_seed_topk_df = pd.concat(five_seed_topk_rows, ignore_index=True)

    five_seed_nuts_diagnostics_df = pd.DataFrame(five_seed_nuts_diagnostic_rows)

    if len(five_seed_variant_rows) > 0:
        five_seed_variant_selection_df = pd.concat(five_seed_variant_rows, ignore_index=True)
    else:
        five_seed_variant_selection_df = pd.DataFrame()

    display(five_seed_metrics_df.round(6))
    display(five_seed_topk_df.round(6))
    display(five_seed_nuts_diagnostics_df.round(6))

    five_seed_metrics_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_all_model_metrics_raw.csv"),
        index=False
    )

    five_seed_topk_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_all_model_topk_raw.csv"),
        index=False
    )

    five_seed_nuts_diagnostics_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_nuts_diagnostics.csv"),
        index=False
    )

    if not five_seed_variant_selection_df.empty:
        five_seed_variant_selection_df.to_csv(
            os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_variant_selection.csv"),
            index=False
        )

else:
    print("RUN_FULL_FIVE_SEED_EVALUATION is False.")


# %% Cell 47
# ============================================================
# CELL 44: Five-Seed Mean ± Standard Deviation Summaries
# ============================================================

# Purpose:
# Summarize five-seed results using mean and standard deviation.

if "five_seed_metrics_df" in globals() and len(five_seed_metrics_df) > 0:

    metric_cols_for_summary = [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1",
        "F2",
        "MCC",
        "AUROC",
        "AUPRC",
        "Brier",
        "Log Loss",
        "ECE",
        "ACE",
        "MCE"
    ]

    metric_cols_for_summary = [
        c for c in metric_cols_for_summary if c in five_seed_metrics_df.columns
    ]

    five_seed_summary_df = (
        five_seed_metrics_df
        .groupby("Model")[metric_cols_for_summary]
        .agg(["mean", "std"])
    )

    # Flatten multi-index columns.
    five_seed_summary_df.columns = [
        f"{metric}_{stat}" for metric, stat in five_seed_summary_df.columns
    ]

    five_seed_summary_df = five_seed_summary_df.reset_index()

    display(five_seed_summary_df.round(6))

    five_seed_summary_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_all_model_metric_summary.csv"),
        index=False
    )

    # Create formatted mean ± std table.
    formatted_rows = []
    for _, row in five_seed_summary_df.iterrows():
        out = {"Model": row["Model"]}
        for metric in metric_cols_for_summary:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"
            out[metric] = f"{row[mean_col]:.6f} ± {row[std_col]:.6f}"
        formatted_rows.append(out)

    five_seed_formatted_summary_df = pd.DataFrame(formatted_rows)

    display(five_seed_formatted_summary_df)

    five_seed_formatted_summary_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_all_model_metric_summary_formatted.csv"),
        index=False
    )

else:
    print("five_seed_metrics_df not available. Run Cell 43 first.")


# ------------------------------------------------------------
# Top-k mean ± std summary
# ------------------------------------------------------------
if "five_seed_topk_df" in globals() and len(five_seed_topk_df) > 0:

    topk_summary_df = (
        five_seed_topk_df
        .groupby(["Model", "Top Fraction"])[["Precision@k", "Recall@k", "Lift"]]
        .agg(["mean", "std"])
    )

    topk_summary_df.columns = [
        f"{metric}_{stat}" for metric, stat in topk_summary_df.columns
    ]

    topk_summary_df = topk_summary_df.reset_index()

    display(topk_summary_df.round(6))

    topk_summary_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_topk_summary.csv"),
        index=False
    )

    # Formatted version.
    formatted_topk_rows = []
    for _, row in topk_summary_df.iterrows():
        out = {
            "Model": row["Model"],
            "Top Fraction": row["Top Fraction"],
            "Precision@k": f"{row['Precision@k_mean']:.6f} ± {row['Precision@k_std']:.6f}",
            "Recall@k": f"{row['Recall@k_mean']:.6f} ± {row['Recall@k_std']:.6f}",
            "Lift": f"{row['Lift_mean']:.6f} ± {row['Lift_std']:.6f}",
        }
        formatted_topk_rows.append(out)

    five_seed_topk_formatted_df = pd.DataFrame(formatted_topk_rows)

    display(five_seed_topk_formatted_df)

    five_seed_topk_formatted_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_topk_summary_formatted.csv"),
        index=False
    )

else:
    print("five_seed_topk_df not available. Run Cell 43 first.")


# %% Cell 48
# ============================================================
# CELL 45: Five-Seed Statistical Comparison and Diagnostic Summary
# ============================================================
# Purpose:
# Compute paired seed-level comparisons between BehaviorGraph-NUTS and each
# internal baseline across the same five random seeds.
#
# Interpretation:
# - For higher-is-better metrics, positive improvement means BehaviorGraph-NUTS is larger.
# - For lower-is-better metrics, positive improvement means BehaviorGraph-NUTS is smaller.

HIGHER_IS_BETTER_METRICS = [
    "Accuracy", "Balanced Accuracy", "Precision", "Recall", "F1", "F2",
    "MCC", "AUROC", "AUPRC", "Recall@1%"
]

LOWER_IS_BETTER_METRICS = [
    "Brier", "Log Loss", "ECE", "ACE", "MCE"
]

INTERNAL_BASELINES_FOR_STATS = [
    "Tabular XGBoost",
    "BehaviorGraph-XGBoost",
    "Deterministic Logistic Fusion"
]

if "five_seed_metrics_df" in globals() and "BehaviorGraph-NUTS" in five_seed_metrics_df["Model"].unique():

    proposed_model_name = "BehaviorGraph-NUTS"

    available_metrics = [
        m for m in HIGHER_IS_BETTER_METRICS + LOWER_IS_BETTER_METRICS
        if m in five_seed_metrics_df.columns
    ]

    stat_rows = []

    for reference_model_name in INTERNAL_BASELINES_FOR_STATS:
        if reference_model_name not in five_seed_metrics_df["Model"].unique():
            continue

        for metric in available_metrics:
            prop = (
                five_seed_metrics_df[five_seed_metrics_df["Model"] == proposed_model_name]
                .sort_values("Seed")[["Seed", metric]]
                .rename(columns={metric: "Proposed"})
            )

            ref = (
                five_seed_metrics_df[five_seed_metrics_df["Model"] == reference_model_name]
                .sort_values("Seed")[["Seed", metric]]
                .rename(columns={metric: "Baseline"})
            )

            merged = prop.merge(ref, on="Seed", how="inner")

            if len(merged) >= 2:
                raw_diff = merged["Proposed"] - merged["Baseline"]

                if metric in LOWER_IS_BETTER_METRICS:
                    improvement = merged["Baseline"] - merged["Proposed"]
                    better_direction = "Lower"
                else:
                    improvement = merged["Proposed"] - merged["Baseline"]
                    better_direction = "Higher"

                try:
                    t_stat, t_p = ttest_rel(merged["Proposed"], merged["Baseline"])
                except Exception:
                    t_stat, t_p = np.nan, np.nan

                try:
                    w_stat, w_p = wilcoxon(merged["Proposed"], merged["Baseline"])
                except Exception:
                    w_stat, w_p = np.nan, np.nan

                stat_rows.append({
                    "Metric": metric,
                    "Better Direction": better_direction,
                    "Proposed Model": proposed_model_name,
                    "Baseline Model": reference_model_name,
                    "Mean Proposed": merged["Proposed"].mean(),
                    "Mean Baseline": merged["Baseline"].mean(),
                    "Raw Difference (Proposed - Baseline)": raw_diff.mean(),
                    "Improvement": improvement.mean(),
                    "Improvement Std": improvement.std(),
                    "Paired t-test p-value": t_p,
                    "Wilcoxon p-value": w_p,
                    "Seeds Compared": len(merged)
                })

    five_seed_stat_tests_df = pd.DataFrame(stat_rows)

    display(five_seed_stat_tests_df.round(9))

    five_seed_stat_tests_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_statistical_tests_all_baselines.csv"),
        index=False
    )

else:
    print("BehaviorGraph-NUTS five-seed metrics not available. Run Cell 43 with RUN_NUTS_WITHIN_FIVE_SEED=True.")


# ------------------------------------------------------------
# Manuscript-ready p-value matrix
# ------------------------------------------------------------
if "five_seed_stat_tests_df" in globals() and len(five_seed_stat_tests_df) > 0:

    pvalue_table_df = (
        five_seed_stat_tests_df
        .pivot_table(
            index=["Metric", "Better Direction"],
            columns="Baseline Model",
            values="Paired t-test p-value",
            aggfunc="first"
        )
        .reset_index()
    )

    display(pvalue_table_df)

    pvalue_table_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_pvalue_table.csv"),
        index=False
    )


# ------------------------------------------------------------
# NUTS diagnostics summary across seeds
# ------------------------------------------------------------
if "five_seed_nuts_diagnostics_df" in globals() and len(five_seed_nuts_diagnostics_df) > 0:

    display(five_seed_nuts_diagnostics_df)

    diag_numeric_cols = [
        "Divergences",
        "Max Rhat",
        "Min Bulk ESS",
        "Min Tail ESS",
        "NUTS Subset Size",
        "NUTS Subset Positive Rate"
    ]

    diag_numeric_cols = [
        c for c in diag_numeric_cols if c in five_seed_nuts_diagnostics_df.columns
    ]

    five_seed_nuts_diag_summary_df = (
        five_seed_nuts_diagnostics_df[diag_numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .T
        .reset_index()
        .rename(columns={"index": "Diagnostic"})
    )

    display(five_seed_nuts_diag_summary_df.round(6))

    five_seed_nuts_diag_summary_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_nuts_diagnostic_summary.csv"),
        index=False
    )

    if "Selected Variant" in five_seed_nuts_diagnostics_df.columns:
        variant_count_df = (
            five_seed_nuts_diagnostics_df["Selected Variant"]
            .value_counts()
            .rename_axis("Selected Variant")
            .reset_index(name="Count")
        )
        display(variant_count_df)
        variant_count_df.to_csv(
            os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_selected_variant_counts.csv"),
            index=False
        )

else:
    print("five_seed_nuts_diagnostics_df not available. Run the full five-seed NUTS evaluation first.")



# %% Cell 49
# ============================================================
# CELL 46: Five-Seed Visualization with Error Bars
# ============================================================

# Purpose:
# Generate manuscript-ready five-seed comparison plots with mean ± std error bars.

if "five_seed_summary_df" in globals() and len(five_seed_summary_df) > 0:

    plot_metrics = ["AUROC", "AUPRC", "Brier", "ECE", "MCE"]
    plot_metrics = [m for m in plot_metrics if f"{m}_mean" in five_seed_summary_df.columns]

    for metric in plot_metrics:
        plot_df = five_seed_summary_df.copy()

        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        plt.figure(figsize=(10, 5))
        plt.bar(
            plot_df["Model"],
            plot_df[mean_col],
            yerr=plot_df[std_col],
            capsize=5
        )
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(metric)
        plt.title(f"Five-Seed Model Comparison: {metric} (mean ± std)")
        plt.tight_layout()

        file_metric = metric.lower().replace(" ", "_").replace("-", "_")
        plt.savefig(
            os.path.join(FIG_DIR, f"five_seed_model_comparison_{file_metric}.png"),
            dpi=300
        )
        plt.show()

else:
    print("five_seed_summary_df not available. Run Cell 44 first.")


# ------------------------------------------------------------
# Five-seed top-k recall plot
# ------------------------------------------------------------
if "topk_summary_df" in globals() and len(topk_summary_df) > 0:

    recall_plot_df = topk_summary_df.copy()

    plt.figure(figsize=(10, 6))

    for model_name in recall_plot_df["Model"].unique():
        sub = recall_plot_df[recall_plot_df["Model"] == model_name].sort_values("Top Fraction")

        plt.errorbar(
            sub["Top Fraction"] * 100,
            sub["Recall@k_mean"],
            yerr=sub["Recall@k_std"],
            marker="o",
            capsize=4,
            label=model_name
        )

    plt.xlabel("Alert budget (%)")
    plt.ylabel("Recall@k")
    plt.title("Five-Seed Top-k Alert Recall Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "five_seed_topk_recall_comparison.png"),
        dpi=300
    )
    plt.show()

else:
    print("topk_summary_df not available. Run Cell 44 first.")


# ------------------------------------------------------------
# NUTS diagnostic stability plot
# ------------------------------------------------------------
if "five_seed_nuts_diagnostics_df" in globals() and len(five_seed_nuts_diagnostics_df) > 0:

    diag_plot_df = five_seed_nuts_diagnostics_df.copy()

    plt.figure(figsize=(8, 5))
    plt.plot(diag_plot_df["Seed"], diag_plot_df["Max Rhat"], marker="o", label="Max R-hat")
    plt.axhline(1.01, linestyle="--", label="1.01 reference")
    plt.axhline(1.05, linestyle=":", label="1.05 warning")
    plt.xlabel("Seed")
    plt.ylabel("Max R-hat")
    plt.title("Five-Seed NUTS R-hat Stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "five_seed_nuts_rhat_stability.png"),
        dpi=300
    )
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(diag_plot_df["Seed"], diag_plot_df["Min Bulk ESS"], marker="o", label="Min Bulk ESS")
    plt.plot(diag_plot_df["Seed"], diag_plot_df["Min Tail ESS"], marker="o", label="Min Tail ESS")
    plt.axhline(400, linestyle="--", label="ESS=400 reference")
    plt.axhline(100, linestyle=":", label="ESS=100 warning")
    plt.xlabel("Seed")
    plt.ylabel("Effective sample size")
    plt.title("Five-Seed NUTS ESS Stability")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(FIG_DIR, "five_seed_nuts_ess_stability.png"),
        dpi=300
    )
    plt.show()

else:
    print("five_seed_nuts_diagnostics_df not available.")


# %% Cell 50
# ============================================================
# CELL 47: Inference Runtime and Memory Transparency
# ============================================================
# Purpose:
# Measure inference-time scoring cost and peak memory use for the main evidence,
# fusion, and posterior-scoring components on the held-out test partition.
#
# Notes:
# - These measurements are deployment-time scoring costs after models have been fitted.
# - NUTS posterior sampling is an offline fitting step and is not rerun inside these functions.
# - Runtime and memory values depend on hardware, Python process state, and batch size.

def _as_gb(num_bytes):
    return float(num_bytes) / (1024 ** 3)

def measure_inference_call(method_name, scoring_fn, repeats=3, warmup=True):
    """
    Measure wall-clock runtime and memory use for a scoring function.

    Returns the best runtime across repeats, the average runtime, Python peak allocation,
    and process RSS delta. The scoring function should return a NumPy array or dictionary
    of arrays and should not modify global state.
    """
    process = psutil.Process(os.getpid())

    if warmup:
        _ = scoring_fn()

    runtimes = []
    peak_python_allocations = []
    rss_deltas = []

    for _ in range(repeats):
        gc.collect()
        rss_before = process.memory_info().rss

        import tracemalloc
        tracemalloc.start()
        t0 = time.perf_counter()

        result = scoring_fn()

        elapsed = time.perf_counter() - t0
        current_alloc, peak_alloc = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rss_after = process.memory_info().rss

        # Keep result alive only long enough to avoid lazy evaluation issues.
        if isinstance(result, dict):
            result_size = sum(np.asarray(v).size for v in result.values())
        else:
            result_size = np.asarray(result).size

        runtimes.append(elapsed)
        peak_python_allocations.append(peak_alloc)
        rss_deltas.append(max(rss_after - rss_before, 0))

        del result
        _ = result_size
        gc.collect()

    return {
        "Method": method_name,
        "Best Runtime (s)": float(np.min(runtimes)),
        "Mean Runtime (s)": float(np.mean(runtimes)),
        "Std Runtime (s)": float(np.std(runtimes, ddof=1)) if len(runtimes) > 1 else 0.0,
        "Peak Python Allocation (GB)": _as_gb(max(peak_python_allocations)),
        "Peak Process RSS Delta (GB)": _as_gb(max(rss_deltas)),
        "Repeats": repeats
    }


inference_measurement_functions = {
    "Rule-Based AML Risk": lambda: rule_based_risk_score(test_df_f),
    "Tabular XGBoost": lambda: tabular_xgb_model.predict_proba(X_test_base)[:, 1],
    "BehaviorGraph-XGBoost": lambda: behaviorgraph_xgb_model.predict_proba(X_test_bg_selected)[:, 1],
    "RUS-XGBoost": lambda: rus_xgb_model.predict_proba(X_test_bg_selected)[:, 1],
    "Isolation Forest": lambda: np.clip(
        if_scaler.transform((-iforest.decision_function(X_test_bg_selected_scaled)).reshape(-1, 1)).ravel(),
        0,
        1
    ),
    "Deterministic Logistic Fusion": lambda: meta_lr.predict_proba(X_fusion_test)[:, 1],
    "BehaviorGraph-NUTS": lambda: posterior_predict_fusion(X_fusion_test, idata, batch_size=100000),
}

runtime_memory_rows = []
for method_name, scoring_fn in inference_measurement_functions.items():
    print(f"Measuring inference runtime and memory for: {method_name}")
    runtime_memory_rows.append(
        measure_inference_call(method_name, scoring_fn, repeats=3, warmup=True)
    )

runtime_memory_inference_df = pd.DataFrame(runtime_memory_rows)

# Relative runtime uses Tabular XGBoost as the reference when available.
if "Tabular XGBoost" in runtime_memory_inference_df["Method"].values:
    reference_time = float(
        runtime_memory_inference_df.loc[
            runtime_memory_inference_df["Method"] == "Tabular XGBoost",
            "Best Runtime (s)"
        ].iloc[0]
    )
    runtime_memory_inference_df["Relative to Tabular XGBoost (x)"] = (
        runtime_memory_inference_df["Best Runtime (s)"] / max(reference_time, EPS)
    )

display(runtime_memory_inference_df.round(6))

runtime_memory_inference_df.to_csv(
    os.path.join(TABLE_DIR, "behaviorgraph_nuts_inference_runtime_memory.csv"),
    index=False
)



# %% Cell 51
# ============================================================
# CELL 48: Save Core Artifacts
# ============================================================

# Purpose:
# Save final configuration, models, preprocessors, five-seed outputs,
# and artifact metadata. This should remain the final notebook cell.

artifacts = {
    "framework_name": FRAMEWORK_NAME,
    "paper_title": PAPER_TITLE,
    "seed": SEED,
    "random_seeds": RANDOM_SEEDS,
    "split_mode": SPLIT_MODE,
    "base_feature_cols": base_feature_cols,
    "behavior_graph_feature_cols": behavior_graph_feature_cols,
    "all_feature_cols": all_feature_cols,
    "selected_behaviorgraph_feature_set": selected_behaviorgraph_feature_set,
    "selected_behaviorgraph_feature_cols": selected_behaviorgraph_feature_cols,
    "fusion_feature_names": fusion_feature_names,
    "dropped_fusion_features": dropped_fusion_features,
    "nuts_max_fusion_rows": NUTS_MAX_FUSION_ROWS,
    "nuts_draws": NUTS_DRAWS,
    "nuts_tune": NUTS_TUNE,
    "nuts_chains": NUTS_CHAINS,
    "nuts_cores": NUTS_CORES,
    "nuts_target_accept": NUTS_TARGET_ACCEPT,
    "best_f2_constrained_threshold": float(best_f2_policy["Threshold"]),
    "best_mcc_constrained_threshold": float(best_mcc_policy["Threshold"]),
    "selected_bayes_variant": str(selected_bayes_variant),
    "score_selection_mode": SCORE_SELECTION_MODE,
    "max_nuts_fusion_features": MAX_NUTS_FUSION_FEATURES,
    "run_subset_size_convergence_diagnostic": RUN_SUBSET_SIZE_CONVERGENCE_DIAGNOSTIC,
    "max_manuscript_alert_rate": MAX_MANUSCRIPT_ALERT_RATE,
    "five_seed_evaluation_completed": bool("five_seed_metrics_df" in globals()),
    "five_seed_nuts_completed": bool(
        "five_seed_nuts_diagnostics_df" in globals()
        and len(five_seed_nuts_diagnostics_df) > 0
        and "BehaviorGraph-NUTS" in five_seed_metrics_df["Model"].unique()
        if "five_seed_metrics_df" in globals() else False
    ),
    "five_seed_list": FIVE_SEED_LIST if "FIVE_SEED_LIST" in globals() else RANDOM_SEEDS,
    "five_seed_nuts_draws": FIVE_SEED_NUTS_DRAWS if "FIVE_SEED_NUTS_DRAWS" in globals() else None,
    "five_seed_nuts_tune": FIVE_SEED_NUTS_TUNE if "FIVE_SEED_NUTS_TUNE" in globals() else None,
    "five_seed_nuts_chains": FIVE_SEED_NUTS_CHAINS if "FIVE_SEED_NUTS_CHAINS" in globals() else None,
    "five_seed_nuts_target_accept": FIVE_SEED_NUTS_TARGET_ACCEPT if "FIVE_SEED_NUTS_TARGET_ACCEPT" in globals() else None
}

with open(os.path.join(OUTPUT_DIR, "behaviorgraph_nuts_artifacts_config.json"), "w") as f:
    json.dump(artifacts, f, indent=2)

with open(os.path.join(MODEL_DIR, "behaviorgraph_nuts_preprocessors.pkl"), "wb") as f:
    pickle.dump({
        "base_imputer": base_imputer,
        "base_scaler": base_scaler,
        "imputer": imputer,
        "scaler": scaler,
        "bg_selected_imputer": bg_selected_imputer,
        "bg_selected_scaler": bg_selected_scaler,
        "fusion_imputer": fusion_imputer,
        "fusion_scaler": fusion_scaler,
        "category_maps": category_maps,
        "category_frequency_maps": category_frequency_maps,
        "calibrators": calibrators,
        "bayes_final_calibrator": bayes_final_calibrator
    }, f)

with open(os.path.join(MODEL_DIR, "behaviorgraph_nuts_models.pkl"), "wb") as f:
    pickle.dump({
        "tabular_xgb_model": tabular_xgb_model,
        "behaviorgraph_xgb_model": behaviorgraph_xgb_model,
        "rus_xgb_model": rus_xgb_model,
        "iforest": iforest,
        "meta_lr": meta_lr
    }, f)

# Save important in-memory five-seed outputs again, if available.
if "five_seed_metrics_df" in globals():
    five_seed_metrics_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_all_model_metrics_raw.csv"),
        index=False
    )

if "five_seed_summary_df" in globals():
    five_seed_summary_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_all_model_metric_summary.csv"),
        index=False
    )

if "five_seed_formatted_summary_df" in globals():
    five_seed_formatted_summary_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_all_model_metric_summary_formatted.csv"),
        index=False
    )

if "five_seed_topk_df" in globals():
    five_seed_topk_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_all_model_topk_raw.csv"),
        index=False
    )

if "topk_summary_df" in globals():
    topk_summary_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_topk_summary.csv"),
        index=False
    )

if "five_seed_topk_formatted_df" in globals():
    five_seed_topk_formatted_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_topk_summary_formatted.csv"),
        index=False
    )

if "five_seed_nuts_diagnostics_df" in globals():
    five_seed_nuts_diagnostics_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_nuts_diagnostics.csv"),
        index=False
    )

if "five_seed_nuts_diag_summary_df" in globals():
    five_seed_nuts_diag_summary_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_nuts_diagnostic_summary.csv"),
        index=False
    )

if "five_seed_stat_tests_df" in globals():
    five_seed_stat_tests_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_statistical_tests.csv"),
        index=False
    )

if "pvalue_table_df" in globals():
    pvalue_table_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_pvalue_table.csv"),
        index=False
    )

if "runtime_memory_inference_df" in globals():
    runtime_memory_inference_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_inference_runtime_memory.csv"),
        index=False
    )

if "five_seed_variant_selection_df" in globals() and not five_seed_variant_selection_df.empty:
    five_seed_variant_selection_df.to_csv(
        os.path.join(TABLE_DIR, "behaviorgraph_nuts_five_seed_variant_selection.csv"),
        index=False
    )

print("Artifacts saved to:", OUTPUT_DIR)
print("Tables saved to:", TABLE_DIR)
print("Figures saved to:", FIG_DIR)
print("Models saved to:", MODEL_DIR)


# ==============================================================================
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 
# 
# 
# ---
# 
# ==============================================================================


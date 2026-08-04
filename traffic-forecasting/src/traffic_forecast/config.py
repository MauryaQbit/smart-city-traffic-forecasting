"""Project-wide configuration: paths, seeds, hyperparameters, feature specs.

Single source of truth so the pipeline stages, models, and dashboard never
drift apart on magic numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "traffic_data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
TRAIN_PKL = PROCESSED_DIR / "train_df.pkl"
VAL_PKL = PROCESSED_DIR / "val_df.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42
VAL_FRAC = 0.2

START = "2022-01-01"
END = "2023-12-31 23:00:00"
MISSING_RATIO = 0.001
SPIKE_RATE = 0.003

JUNCTIONS = {
    1: {"base": 55, "amp": 35, "noise": 6},
    2: {"base": 30, "amp": 18, "noise": 4},
    3: {"base": 18, "amp": 10, "noise": 3},
    4: {"base": 10, "amp": 5, "noise": 2},
}

HOLIDAYS = pd.to_datetime(
    [
        "2022-01-26",
        "2022-03-18",
        "2022-08-15",
        "2022-10-02",
        "2022-10-24",
        "2022-11-08",
        "2022-12-25",
        "2023-01-26",
        "2023-03-08",
        "2023-08-15",
        "2023-10-02",
        "2023-11-12",
        "2023-12-25",
    ]
)

FEATURE_COLS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "hour_sin_2",
    "hour_cos_2",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "roll_mean_3",
    "roll_mean_24",
    "roll_mean_168",
    "is_outlier",
]
LSTM_FEATURE_COLS = [
    "hour",
    "dayofweek",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_24",
    "lag_168",
    "roll_mean_3",
    "roll_mean_24",
    "roll_mean_168",
]

SEQ_LEN = 24

RF_PARAM_GRID = {
    "n_estimators": [100, 200, 400],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", 0.7],
}
RF_GRID_SUBSAMPLE = 12000
RF_RANDOM_STATE = SEED

SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)
SARIMA_MAXITER = 50
SARIMA_TRAIN_WINDOW = 24 * 30 * 2
SARIMA_EVAL_HOURS = 24 * 14

LSTM_HYPERPARAMS = {
    "hidden": 32,
    "layers": 2,
    "lr": 2e-3,
    "batch_size": 512,
    "epochs": 15,
    "seq_len": SEQ_LEN,
    "grad_clip": 1.0,
    "patience": 4,
}

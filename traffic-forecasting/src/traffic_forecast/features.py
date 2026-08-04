"""Feature engineering + chronological train/val split.

Two parallel preprocessed views of the same feature set:
  - one-hot + scaled (for Linear Regression)
  - integer-coded, unscaled (for tree models and the dashboard's predictor form)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from traffic_forecast.config import FEATURE_COLS, HOLIDAYS, VAL_FRAC


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["DateTime"])


def compute_iqr_thresholds(df: pd.DataFrame, col: str = "Vehicles") -> tuple[float, float]:
    """Global (cross-junction) IQR thresholds. Use this on the TRAIN half only,
    then pass the returned tuple into engineer_features for both train and val
    so the val outlier flag is not informed by val quantiles (leakage fix)."""
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def flag_outliers_iqr(
    df: pd.DataFrame,
    col: str = "Vehicles",
    iqr_thresholds: tuple[float, float] | None = None,
) -> pd.Series:
    """Per-junction IQR outlier flag. If iqr_thresholds is None, compute per
    junction from the data (legacy behaviour). If supplied, apply the global
    thresholds uniformly (leakage-free behaviour)."""
    flags = pd.Series(False, index=df.index)
    if iqr_thresholds is None:
        for _, grp in df.groupby("Junction"):
            q1, q3 = grp[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flags.loc[grp.index] = (grp[col] < lower) | (grp[col] > upper)
    else:
        lower, upper = iqr_thresholds
        flags = (df[col] < lower) | (df[col] > upper)
    return flags


def engineer_features(
    df: pd.DataFrame,
    iqr_thresholds: tuple[float, float] | None = None,
) -> pd.DataFrame:
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    df["Vehicles"] = df.groupby("Junction")["Vehicles"].ffill().bfill()
    df["is_outlier"] = flag_outliers_iqr(df, iqr_thresholds=iqr_thresholds)

    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek
    df["month"] = df["DateTime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = df["DateTime"].dt.normalize().isin(HOLIDAYS).astype(int)

    # Cyclical encodings let linear models see hour 23 and hour 0 as adjacent
    # rather than maximally distant. Two harmonics for hour (24h and 12h) capture
    # the asymmetric morning/evening rush shape; one for day-of-week.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["hour_sin_2"] = np.sin(2 * np.pi * df["hour"] / 12)
    df["hour_cos_2"] = np.cos(2 * np.pi * df["hour"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    grp = df.groupby("Junction")["Vehicles"]
    df["lag_1"] = grp.shift(1)
    df["lag_24"] = grp.shift(24)
    df["lag_168"] = grp.shift(168)
    df["roll_mean_3"] = grp.transform(lambda s: s.shift(1).rolling(3).mean())
    df["roll_mean_24"] = grp.transform(lambda s: s.shift(1).rolling(24).mean())
    df["roll_mean_168"] = grp.transform(lambda s: s.shift(1).rolling(168).mean())

    df = df.dropna(
        subset=["lag_1", "lag_24", "lag_168", "roll_mean_3", "roll_mean_24", "roll_mean_168"]
    ).reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["DateTime"].quantile(1 - val_frac)
    train = df[df["DateTime"] < cutoff].copy()
    val = df[df["DateTime"] >= cutoff].copy()
    return train, val


def make_linear_view(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    cols = FEATURE_COLS + ["Junction"]
    train_x = pd.get_dummies(train[cols], columns=["Junction"], prefix="junc")
    val_x = pd.get_dummies(val[cols], columns=["Junction"], prefix="junc")
    val_x = val_x.reindex(columns=train_x.columns, fill_value=0)

    scaler = StandardScaler()
    train_x_scaled = pd.DataFrame(
        scaler.fit_transform(train_x), columns=train_x.columns, index=train.index
    )
    val_x_scaled = pd.DataFrame(scaler.transform(val_x), columns=val_x.columns, index=val.index)
    return train_x_scaled, val_x_scaled, scaler


def make_tree_view(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = FEATURE_COLS + ["Junction"]
    return train[cols].copy(), val[cols].copy()

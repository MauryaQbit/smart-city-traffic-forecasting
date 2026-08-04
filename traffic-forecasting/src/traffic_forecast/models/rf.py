"""Random Forest: default baseline + grid-search tuned variant."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from traffic_forecast.config import RF_GRID_SUBSAMPLE, RF_PARAM_GRID, RF_RANDOM_STATE
from traffic_forecast.eval.metrics import mae, rmse
from traffic_forecast.features import make_tree_view


def train_default(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    train_x, val_x = make_tree_view(train_df, val_df)
    y_train = train_df["Vehicles"].values
    y_val = val_df["Vehicles"].values

    t0 = time.time()
    model = RandomForestRegressor(random_state=RF_RANDOM_STATE, n_jobs=-1, n_estimators=100)
    model.fit(train_x, y_train)
    pred_val = model.predict(val_x)
    train_seconds = time.time() - t0

    return {
        "model": model,
        "pred_val": pred_val,
        "metrics": {
            "MAE": mae(y_val, pred_val),
            "RMSE": rmse(y_val, pred_val),
            "train_seconds": train_seconds,
        },
    }


def train_tuned(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    grid_subsample: int = RF_GRID_SUBSAMPLE,
) -> dict:
    train_x, val_x = make_tree_view(train_df, val_df)
    y_train = train_df["Vehicles"].values
    y_val = val_df["Vehicles"].values

    t0 = time.time()
    sample_idx = np.random.default_rng(0).choice(
        len(train_x),
        size=min(grid_subsample, len(train_x)),
        replace=False,
    )
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RF_RANDOM_STATE, n_jobs=-1),
        RF_PARAM_GRID,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(train_x.iloc[sample_idx], y_train[sample_idx])
    best_params = grid.best_params_

    model = RandomForestRegressor(random_state=RF_RANDOM_STATE, n_jobs=-1, **best_params)
    model.fit(train_x, y_train)
    pred_val = model.predict(val_x)
    train_seconds = time.time() - t0

    return {
        "model": model,
        "pred_val": pred_val,
        "best_params": best_params,
        "metrics": {
            "MAE": mae(y_val, pred_val),
            "RMSE": rmse(y_val, pred_val),
            "train_seconds": train_seconds,
        },
    }

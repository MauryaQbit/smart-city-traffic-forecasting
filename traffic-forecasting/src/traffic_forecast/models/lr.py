"""Linear Regression model."""

from __future__ import annotations

import time

import pandas as pd
from sklearn.linear_model import LinearRegression

from traffic_forecast.eval.metrics import mae, rmse
from traffic_forecast.features import make_linear_view


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    train_x, val_x, scaler = make_linear_view(train_df, val_df)
    y_train = train_df["Vehicles"].values
    y_val = val_df["Vehicles"].values

    t0 = time.time()
    model = LinearRegression().fit(train_x, y_train)
    pred_val = model.predict(val_x)
    train_seconds = time.time() - t0

    return {
        "model": model,
        "scaler": scaler,
        "pred_val": pred_val,
        "metrics": {
            "MAE": mae(y_val, pred_val),
            "RMSE": rmse(y_val, pred_val),
            "train_seconds": train_seconds,
        },
    }

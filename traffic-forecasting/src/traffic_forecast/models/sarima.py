"""SARIMA per-junction on a rolling 2-month train window, forecasting the
first 2 weeks of the validation window. Mirrors how SARIMA is used in practice
(periodic refit on recent data) rather than a one-shot fit over years."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from traffic_forecast.config import (
    SARIMA_EVAL_HOURS,
    SARIMA_MAXITER,
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
    SARIMA_TRAIN_WINDOW,
)
from traffic_forecast.eval.metrics import mae, rmse


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    predictions = {}
    all_actual, all_pred = [], []

    t0 = time.time()
    for jid in sorted(train_df["Junction"].unique()):
        tr = (
            train_df.loc[train_df["Junction"] == jid].set_index("DateTime")["Vehicles"].sort_index()
        )
        va = val_df.loc[val_df["Junction"] == jid].set_index("DateTime")["Vehicles"].sort_index()
        tr_recent = tr.iloc[-SARIMA_TRAIN_WINDOW:]
        va_eval = va.iloc[:SARIMA_EVAL_HOURS]

        model = SARIMAX(
            tr_recent,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=SARIMA_MAXITER)
        fcast = fit.forecast(steps=len(va_eval))

        predictions[int(jid)] = {
            "datetime": va_eval.index.astype(str).tolist(),
            "actual": va_eval.values.tolist(),
            "pred": fcast.values.tolist(),
        }
        all_actual.extend(va_eval.values.tolist())
        all_pred.extend(fcast.values.tolist())

    actual = np.array(all_actual)
    pred = np.array(all_pred)
    return {
        "predictions": predictions,
        "metrics": {
            "MAE": mae(actual, pred),
            "RMSE": rmse(actual, pred),
            "train_seconds": time.time() - t0,
        },
    }

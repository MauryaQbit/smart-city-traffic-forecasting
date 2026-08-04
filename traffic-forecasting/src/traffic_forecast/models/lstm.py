"""LSTM model with target scaling, gradient clipping, early stopping, and
junction identity. The previous implementation collapsed to a constant output
because the raw 0..135 target combined with MSE loss and no clipping pushed
the recurrent gradients into a saturation regime; the network settled at the
global mean and never escaped. These four fixes together recover useful
learning."""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from traffic_forecast.config import LSTM_FEATURE_COLS, LSTM_HYPERPARAMS, SEED
from traffic_forecast.eval.metrics import mae, rmse


class TrafficLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, grp in df.groupby("Junction"):
        grp = grp.sort_values("DateTime")
        vals = grp[feature_cols + [target_col]].values.astype(np.float32)
        for i in range(len(vals) - seq_len):
            xs.append(vals[i : i + seq_len, :-1])
            ys.append(vals[i + seq_len, -1])
    return np.array(xs), np.array(ys)


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int | None = None,
) -> dict:
    hp = LSTM_HYPERPARAMS
    epochs = epochs if epochs is not None else hp["epochs"]
    seq_len = hp["seq_len"]
    feature_cols = LSTM_FEATURE_COLS + ["Junction"]

    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    feature_scaler = MinMaxScaler().fit(train_df[feature_cols])
    target_scaler = MinMaxScaler().fit(train_df[["Vehicles"]])

    tr = train_df.copy()
    va = val_df.copy()
    tr[feature_cols] = feature_scaler.transform(train_df[feature_cols])
    va[feature_cols] = feature_scaler.transform(val_df[feature_cols])
    tr["Vehicles"] = target_scaler.transform(train_df[["Vehicles"]])
    va["Vehicles"] = target_scaler.transform(val_df[["Vehicles"]])

    X_train, y_train = make_sequences(tr, feature_cols, "Vehicles", seq_len)
    X_val, y_val = make_sequences(va, feature_cols, "Vehicles", seq_len)

    t0 = time.time()
    model = TrafficLSTM(n_features=len(feature_cols), hidden=hp["hidden"], layers=hp["layers"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    n = len(X_train_t)
    batch_size = hp["batch_size"]
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp["grad_clip"])
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= hp["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_val_t).numpy()

    pred_val = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual_val = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    return {
        "model": model,
        "pred_val": pred_val,
        "actual_val": actual_val,
        "metrics": {
            "MAE": mae(actual_val, pred_val),
            "RMSE": rmse(actual_val, pred_val),
            "train_seconds": time.time() - t0,
        },
    }

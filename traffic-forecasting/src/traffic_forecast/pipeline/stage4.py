"""Stage 4: combine reports, build dashboard data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from traffic_forecast import config
from traffic_forecast.pipeline._common import ensure_dirs, load_pickled_split


def main() -> None:
    ensure_dirs()
    REPORTS = config.REPORTS_DIR

    results = {}
    for fname in ("results_stage1.json", "results_stage2_sarima.json", "results_stage3_lstm.json"):
        with open(REPORTS / fname) as f:
            results.update(json.load(f))

    comp_df = pd.DataFrame(results).T
    comp_df.index.name = "Model"
    comp_df = comp_df.sort_values("MAE")
    comp_df.to_csv(REPORTS / "model_comparison.csv")
    print(comp_df)

    _, val_df = load_pickled_split()
    val_df = val_df.reset_index(drop=True)

    rf_pred = np.load(REPORTS / "pred_rf_tuned.npy")
    lr_pred = np.load(REPORTS / "pred_linear.npy")
    val_df["pred_rf_tuned"] = rf_pred
    val_df["pred_linear"] = lr_pred
    val_df["abs_error_rf"] = (val_df["Vehicles"] - val_df["pred_rf_tuned"]).abs()

    err = val_df.groupby(["Junction", "hour"])["abs_error_rf"].mean().reset_index()
    err.to_csv(REPORTS / "error_by_junction_hour.csv", index=False)

    with open(REPORTS / "sarima_predictions.json") as f:
        sarima = json.load(f)
    sarima_rows = []
    for jid, d in sarima.items():
        for dt, pred in zip(d["datetime"], d["pred"], strict=False):
            sarima_rows.append({"DateTime": dt, "Junction": int(jid), "pred_sarima": pred})
    sarima_df = pd.DataFrame(sarima_rows)
    sarima_df["DateTime"] = pd.to_datetime(sarima_df["DateTime"])
    val_df["DateTime"] = pd.to_datetime(val_df["DateTime"])
    merged = val_df.merge(sarima_df, on=["DateTime", "Junction"], how="left")

    lstm_pred = np.load(REPORTS / "pred_lstm.npy")
    seq_len = config.SEQ_LEN
    lstm_rows = []
    for jid, grp in val_df.sort_values("DateTime").groupby("Junction"):
        grp = grp.sort_values("DateTime").reset_index(drop=True)
        n_seq = len(grp) - seq_len
        dts = grp["DateTime"].iloc[seq_len : seq_len + n_seq].values
        lstm_rows.append(pd.DataFrame({"DateTime": dts, "Junction": jid}))
    lstm_meta = pd.concat(lstm_rows, ignore_index=True)
    lstm_meta["pred_lstm"] = lstm_pred[: len(lstm_meta)]
    lstm_meta["DateTime"] = pd.to_datetime(lstm_meta["DateTime"])
    merged = merged.merge(lstm_meta, on=["DateTime", "Junction"], how="left")

    cols = [
        "DateTime",
        "Junction",
        "Vehicles",
        "pred_linear",
        "pred_rf_tuned",
        "pred_sarima",
        "pred_lstm",
        "is_holiday",
        "is_outlier",
    ]
    merged[cols].to_csv(REPORTS / "predictions_full.csv", index=False)
    sample = merged[merged["DateTime"] < merged["DateTime"].min() + pd.Timedelta(days=14)]
    sample[cols].to_csv(REPORTS / "predictions_sample.csv", index=False)
    print("Stage 4 done.")


if __name__ == "__main__":
    main()

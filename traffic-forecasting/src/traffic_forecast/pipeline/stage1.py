"""Stage 1: Linear Regression + Random Forest (default + tuned)."""

from __future__ import annotations

import joblib
import numpy as np

from traffic_forecast import config
from traffic_forecast.models import lr as lr_model
from traffic_forecast.models import rf as rf_model
from traffic_forecast.pipeline._common import (
    ensure_dirs,
    load_and_split,
    load_pickled_split,
    write_json,
)


def main() -> dict:
    ensure_dirs()
    try:
        train_df, val_df = load_pickled_split()
    except FileNotFoundError:
        train_df, val_df = load_and_split()

    results = {}

    lr_out = lr_model.train(train_df, val_df)
    joblib.dump(lr_out["model"], config.MODELS_DIR / "linear_regression.joblib")
    joblib.dump(lr_out["scaler"], config.MODELS_DIR / "lr_scaler.joblib")
    np.save(config.REPORTS_DIR / "pred_linear.npy", lr_out["pred_val"])
    results["Linear Regression"] = lr_out["metrics"]

    rf_default_out = rf_model.train_default(train_df, val_df)
    results["Random Forest (default)"] = rf_default_out["metrics"]

    rf_tuned_out = rf_model.train_tuned(train_df, val_df)
    joblib.dump(rf_tuned_out["model"], config.MODELS_DIR / "random_forest_tuned.joblib")
    np.save(config.REPORTS_DIR / "pred_rf_tuned.npy", rf_tuned_out["pred_val"])
    results["Random Forest (tuned)"] = rf_tuned_out["metrics"]

    write_json(config.REPORTS_DIR / "best_rf_params.json", rf_tuned_out["best_params"])
    write_json(config.REPORTS_DIR / "results_stage1.json", results)
    print("Stage 1 done.")
    return results


if __name__ == "__main__":
    main()

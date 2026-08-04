"""Stage 3: LSTM (fixed - target scaling, grad clipping, early stop, junction)."""

from __future__ import annotations

import numpy as np
import torch

from traffic_forecast import config
from traffic_forecast.models import lstm as lstm_model
from traffic_forecast.pipeline._common import (
    ensure_dirs,
    load_pickled_split,
    write_json,
)


def main(epochs: int | None = None) -> dict:
    ensure_dirs()
    train_df, val_df = load_pickled_split()
    out = lstm_model.train(train_df, val_df, epochs=epochs)
    torch.save(out["model"].state_dict(), config.MODELS_DIR / "lstm.pt")
    np.save(config.REPORTS_DIR / "pred_lstm.npy", out["pred_val"])
    np.save(config.REPORTS_DIR / "actual_lstm.npy", out["actual_val"])
    result = {"LSTM": out["metrics"]}
    write_json(config.REPORTS_DIR / "results_stage3_lstm.json", result)
    print("Stage 3 done.")
    return result


if __name__ == "__main__":
    main()

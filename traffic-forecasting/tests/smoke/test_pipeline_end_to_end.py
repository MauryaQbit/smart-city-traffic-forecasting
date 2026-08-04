"""Cheapest possible 'the whole pipeline still runs' test. Uses a tiny
synthetic fixture generated on the fly so it runs in under 90 seconds."""

import numpy as np
import pandas as pd
import pytest

from traffic_forecast import config
from traffic_forecast.pipeline import run_all


@pytest.mark.timeout(180)
def test_pipeline_runs_end_to_end_on_tiny_fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(24 * 75):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    df = pd.DataFrame(rows)

    raw_path = tmp_path / "raw.csv"
    df.to_csv(raw_path, index=False)

    (tmp_path / "processed").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(config, "RAW_PATH", raw_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "TRAIN_PKL", tmp_path / "processed" / "train_df.pkl")
    monkeypatch.setattr(config, "VAL_PKL", tmp_path / "processed" / "val_df.pkl")
    monkeypatch.setattr(config, "FEATURES_PATH", tmp_path / "processed" / "features.csv")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    run_all.main(epochs=2)

    for fname in ("model_comparison.csv", "predictions_full.csv", "pred_rf_tuned.npy"):
        assert (tmp_path / "reports" / fname).exists()
    assert (tmp_path / "models" / "random_forest_tuned.joblib").exists()

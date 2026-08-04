import json

import numpy as np
import pandas as pd
import pytest

from traffic_forecast import config
from traffic_forecast.pipeline import run_all


@pytest.mark.timeout(300)
def test_run_all_writes_expected_artifacts(tmp_path, monkeypatch):
    """End-to-end smoke on a tiny fixture. Patches the data path so we don't
    depend on the committed 70k-row CSV."""
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

    assert (tmp_path / "reports" / "model_comparison.csv").exists()
    assert (tmp_path / "reports" / "predictions_full.csv").exists()
    assert (tmp_path / "reports" / "pred_rf_tuned.npy").exists()
    assert (tmp_path / "models" / "random_forest_tuned.joblib").exists()

    with open(tmp_path / "reports" / "results_stage1.json") as f:
        r1 = json.load(f)
    assert "Linear Regression" in r1
    assert "Random Forest (tuned)" in r1

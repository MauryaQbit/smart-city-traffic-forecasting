import numpy as np
import pandas as pd

from traffic_forecast.models import lr


def _make_fixture():
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(200):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_lr_train_returns_expected_artifacts():
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = lr.train(train, val)
    assert set(out.keys()) >= {"model", "scaler", "pred_val", "metrics"}
    assert {"MAE", "RMSE"} <= set(out["metrics"].keys())
    assert len(out["pred_val"]) == len(val)
    assert out["metrics"]["MAE"] > 0

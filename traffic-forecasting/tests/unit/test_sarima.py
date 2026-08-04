import numpy as np
import pandas as pd

from traffic_forecast.models import sarima


def _make_fixture():
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(24 * 90):  # 90 days per junction so SARIMA window has data
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_sarima_returns_predictions_for_each_junction():
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = sarima.train(train, val)
    assert set(out["predictions"].keys()) == {1, 2, 3, 4}
    for payload in out["predictions"].values():
        assert len(payload["datetime"]) == len(payload["actual"]) == len(payload["pred"])
    assert {"MAE", "RMSE"} <= set(out["metrics"].keys())

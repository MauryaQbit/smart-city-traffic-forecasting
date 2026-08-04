import numpy as np
import pandas as pd

from traffic_forecast.models import rf


def _make_fixture():
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(300):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_rf_default_returns_metrics_and_predictions():
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = rf.train_default(train, val)
    assert {"model", "pred_val", "metrics"} <= set(out.keys())
    assert len(out["pred_val"]) == len(val)


def test_rf_tuned_returns_best_params_in_grid():
    from traffic_forecast import config, features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = rf.train_tuned(train, val, grid_subsample=200)
    assert set(out["best_params"].keys()) == set(config.RF_PARAM_GRID.keys())

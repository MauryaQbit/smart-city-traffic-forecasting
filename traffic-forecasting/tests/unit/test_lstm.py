import numpy as np
import pandas as pd

from traffic_forecast.models import lstm


def _make_fixture():
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(24 * 80):  # 80 days/junction so val sequences exist
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_lstm_predictions_are_not_constant():
    """The single most important test in this sub-project. Pre-fix the LSTM
    emitted the same scalar (~global mean) for every input. Post-fix the
    prediction distribution must have non-trivial spread."""
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = lstm.train(train, val, epochs=3)
    assert out["pred_val"].std() > 3.0, (
        f"LSTM predictions collapsed to a near-constant (std={out['pred_val'].std():.4f}); "
        "the fix is not working."
    )


def test_lstm_metrics_are_finite_and_positive():
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = lstm.train(train, val, epochs=3)
    assert np.isfinite(out["metrics"]["MAE"])
    assert out["metrics"]["MAE"] > 0

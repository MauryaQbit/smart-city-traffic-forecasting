import numpy as np
import pandas as pd
import pytest

from traffic_forecast import features


@pytest.fixture
def tiny_df():
    """4 junctions x 200 hours. Large enough that the 168h weekly-lag dropna
    leaves ~32 rows per junction for downstream tests."""
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(200):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_engineer_features_drops_first_168h_per_junction(tiny_df):
    out = features.engineer_features(tiny_df)
    # 4 junctions x (200 - 168) = 128 rows survive the weekly-lag dropna
    assert len(out) == 128


def test_engineer_features_no_nans(tiny_df):
    out = features.engineer_features(tiny_df)
    assert out.isna().sum().sum() == 0


def test_lag_1_equals_previous_hour_vehicle_count(tiny_df):
    out = features.engineer_features(tiny_df)
    prev = out[["Junction", "DateTime", "Vehicles"]].assign(
        DateTime=lambda d: d["DateTime"] + pd.Timedelta(hours=1)
    )
    merged = out.merge(prev, on=["Junction", "DateTime"], suffixes=("", "_prev"), how="left")
    has_prev = merged["Vehicles_prev"].notna()
    mismatches = (merged.loc[has_prev, "lag_1"] != merged.loc[has_prev, "Vehicles_prev"]).sum()
    assert mismatches == 0


def test_lag_24_equals_same_hour_yesterday_vehicle_count(tiny_df):
    out = features.engineer_features(tiny_df)
    prev = out[["Junction", "DateTime", "Vehicles"]].assign(
        DateTime=lambda d: d["DateTime"] + pd.Timedelta(hours=24)
    )
    merged = out.merge(prev, on=["Junction", "DateTime"], suffixes=("", "_prev"), how="left")
    has_prev = merged["Vehicles_prev"].notna()
    mismatches = (merged.loc[has_prev, "lag_24"] != merged.loc[has_prev, "Vehicles_prev"]).sum()
    assert mismatches == 0


def test_is_weekend_iff_dow_ge_5(tiny_df):
    out = features.engineer_features(tiny_df)
    assert (out["is_weekend"] == (out["dayofweek"] >= 5).astype(int)).all()


def test_chronological_split_train_before_val(tiny_df):
    feat = features.engineer_features(tiny_df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    assert train["DateTime"].max() < val["DateTime"].min()


def test_compute_iqr_thresholds_returns_lower_lt_upper():
    df = pd.DataFrame(
        {
            "Junction": [1, 1, 1, 1, 2, 2, 2, 2],
            "Vehicles": [10, 10, 10, 100, 1, 1, 1, 50],
        }
    )
    q1, q3 = features.compute_iqr_thresholds(df)
    assert q1 < q3


def test_iqr_thresholds_can_be_passed_into_engineer_features(tiny_df):
    # Need enough rows per junction that engineer_features (which now drops
    # the first 168h per junction) leaves something behind. Use a per-junction
    # split instead of an iloc slice.
    train_df = tiny_df[tiny_df["Junction"] <= 2]
    val_df = tiny_df[tiny_df["Junction"] > 2]
    q1, q3 = features.compute_iqr_thresholds(train_df)
    feat_train = features.engineer_features(train_df, iqr_thresholds=(q1, q3))
    feat_val = features.engineer_features(val_df, iqr_thresholds=(q1, q3))
    assert "is_outlier" in feat_train.columns
    assert "is_outlier" in feat_val.columns


def test_make_linear_view_same_columns_train_val(tiny_df):
    feat = features.engineer_features(tiny_df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    train_x, val_x, _ = features.make_linear_view(train, val)
    assert list(train_x.columns) == list(val_x.columns)


def test_make_linear_view_scaler_fit_on_train_only(tiny_df):
    feat = features.engineer_features(tiny_df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    _, _, scaler = features.make_linear_view(train, val)
    assert scaler.mean_ is not None

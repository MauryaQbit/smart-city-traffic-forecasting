import pandas as pd
import pytest

from traffic_forecast import features


@pytest.mark.characterization
@pytest.mark.slow
def test_raw_data_shape(real_raw_df):
    df = real_raw_df
    assert len(df) == 70080
    assert set(df["Junction"].unique()) == {1, 2, 3, 4}
    assert df["Junction"].value_counts().nunique() == 1


@pytest.mark.characterization
@pytest.mark.slow
def test_raw_date_range(real_raw_df):
    assert real_raw_df["DateTime"].min() == pd.Timestamp("2022-01-01 00:00:00")
    assert real_raw_df["DateTime"].max() == pd.Timestamp("2023-12-31 23:00:00")


@pytest.mark.characterization
@pytest.mark.slow
def test_raw_has_approx_70_missing(real_raw_df):
    n_missing = real_raw_df["Vehicles"].isna().sum()
    assert 40 <= n_missing <= 100


@pytest.mark.characterization
@pytest.mark.slow
def test_engineered_features_shape(real_raw_df):
    feat = features.engineer_features(real_raw_df)
    # 70080 raw - 4 junctions x 168h dropped for weekly-lag context = 69408
    assert len(feat) == 69408
    assert feat.isna().sum().sum() == 0
    for col in (
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "is_holiday",
        "hour_sin",
        "hour_cos",
        "hour_sin_2",
        "hour_cos_2",
        "dow_sin",
        "dow_cos",
        "lag_1",
        "lag_24",
        "lag_168",
        "roll_mean_3",
        "roll_mean_24",
        "roll_mean_168",
        "is_outlier",
    ):
        assert col in feat.columns


@pytest.mark.characterization
@pytest.mark.slow
def test_chronological_split_no_leakage(real_raw_df):
    feat = features.engineer_features(real_raw_df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    assert train["DateTime"].max() < val["DateTime"].min()

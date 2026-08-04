import numpy as np
import pandas as pd

from traffic_forecast.data.generate import build_dataset, hourly_profile, weekday_factor


def test_hourly_profile_is_nonneg_with_two_peaks():
    h = np.arange(24)
    p = hourly_profile(h)
    assert p.shape == (24,)
    assert (p >= 0).all()
    assert np.argmax(p) > 15


def test_weekday_factor_weekday_is_one_weekend_is_065():
    assert weekday_factor(np.array([0, 1, 2, 3, 4])).tolist() == [1.0] * 5
    assert weekday_factor(np.array([5, 6])).tolist() == [0.65, 0.65]


def test_build_dataset_has_expected_schema():
    df = build_dataset()
    assert list(df.columns) == ["DateTime", "Junction", "Vehicles", "ID"]
    assert len(df) == 70080
    assert set(df["Junction"].unique()) == {1, 2, 3, 4}


def test_build_dataset_is_reproducible_across_calls():
    df1 = build_dataset()
    df2 = build_dataset()
    pd.testing.assert_frame_equal(df1, df2)


def test_build_dataset_has_some_missing_values():
    df = build_dataset()
    n_missing = df["Vehicles"].isna().sum()
    assert 40 <= n_missing <= 100

import math

import numpy as np

from traffic_forecast.eval.metrics import mae, rmse


def test_rmse_matches_manual_sqrt_of_mse():
    y = np.array([3.0, 4.0])
    p = np.array([3.0, 5.0])
    assert rmse(y, p) == math.sqrt(0.5)


def test_mae_matches_manual_mean_abs_error():
    y = np.array([3.0, 4.0, 10.0])
    p = np.array([3.0, 5.0, 7.0])
    assert mae(y, p) == (0 + 1 + 3) / 3


def test_rmse_returns_python_float():
    out = rmse(np.array([1.0]), np.array([2.0]))
    assert isinstance(out, float)

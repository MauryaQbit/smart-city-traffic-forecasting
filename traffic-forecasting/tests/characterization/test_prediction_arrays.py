from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


@pytest.mark.characterization
@pytest.mark.slow
def test_pred_linear_shape_and_finite():
    arr = np.load(REPORTS / "pred_linear.npy")
    assert arr.shape == (13884,)
    assert np.isfinite(arr).all()


@pytest.mark.characterization
@pytest.mark.slow
def test_pred_rf_shape_and_nonneg():
    arr = np.load(REPORTS / "pred_rf_tuned.npy")
    assert arr.shape == (13884,)
    assert (arr >= 0).all()


@pytest.mark.characterization
@pytest.mark.slow
def test_pred_lstm_shape_and_non_constant():
    """The single most important regression-catch. The pre-fix LSTM emitted
    a literal constant (std=0.00) for all val sequences. Post-fix std is
    ~14.6. A threshold of 3.0 catches both the full collapse and subtler
    partial-collapse failures (a healthy LSTM on this data spans the 0..70
    range; std < 3 means it has stopped discriminating between inputs)."""
    arr = np.load(REPORTS / "pred_lstm.npy")
    assert arr.shape == (13788,)
    assert arr.std() > 3.0, f"LSTM may have partially collapsed (std={arr.std():.4f})"


@pytest.mark.characterization
@pytest.mark.slow
def test_actual_lstm_shape_and_nonneg():
    arr = np.load(REPORTS / "actual_lstm.npy")
    assert arr.shape == (13788,)
    assert (arr >= 0).all()

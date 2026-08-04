import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def _load(name):
    with open(REPORTS / name) as f:
        return json.load(f)


@pytest.mark.characterization
@pytest.mark.slow
def test_linear_regression_mae_within_10pct():
    r = _load("results_stage1.json")["Linear Regression"]
    assert 3.30 <= r["MAE"] <= 4.03


@pytest.mark.characterization
@pytest.mark.slow
def test_linear_regression_rmse_within_10pct():
    r = _load("results_stage1.json")["Linear Regression"]
    assert 4.51 <= r["RMSE"] <= 5.51


@pytest.mark.characterization
@pytest.mark.slow
def test_rf_tuned_mae_within_10pct():
    r = _load("results_stage1.json")["Random Forest (tuned)"]
    assert 2.86 <= r["MAE"] <= 3.49


@pytest.mark.characterization
@pytest.mark.slow
def test_rf_tuned_rmse_within_10pct():
    r = _load("results_stage1.json")["Random Forest (tuned)"]
    assert 3.97 <= r["RMSE"] <= 4.86


@pytest.mark.characterization
@pytest.mark.slow
def test_rf_best_params_unchanged():
    with open(REPORTS / "best_rf_params.json") as f:
        params = json.load(f)
    assert params == {
        "max_depth": 20,
        "max_features": "sqrt",
        "min_samples_leaf": 1,
        "n_estimators": 400,
    }


@pytest.mark.characterization
@pytest.mark.slow
def test_sarima_mae_within_10pct():
    r = _load("results_stage2_sarima.json")["SARIMA"]
    assert 4.19 <= r["MAE"] <= 5.12


@pytest.mark.characterization
@pytest.mark.slow
def test_lstm_mae_within_post_fix_band():
    """Post-feature-engineering the LSTM MAE should land in [2.85, 3.55]. The
    actual value is ~3.19. If this fails ABOVE the band, the LSTM has
    regressed. If it fails BELOW, sub-project work has improved it (update
    the band deliberately, do not widen silently)."""
    r = _load("results_stage3_lstm.json")["LSTM"]
    assert 2.85 <= r["MAE"] <= 3.55


@pytest.mark.characterization
@pytest.mark.slow
def test_model_ranking_by_mae(model_comparison):
    """RF tuned should still be at or near the top. If any model regresses by
    more than 1.0 MAE this fires - the per-model 10% bands catch smaller
    drift, this catches catastrophic regressions that flip the story."""
    mae = model_comparison["MAE"].sort_values()
    assert mae.iloc[0] < mae.iloc[-1]
    assert model_comparison.loc["Random Forest (tuned)", "MAE"] < 4.0
    assert mae.iloc[-1] < 12
    assert (mae.iloc[-1] - mae.iloc[0]) > 1.0

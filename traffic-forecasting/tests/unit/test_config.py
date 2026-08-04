from traffic_forecast import config


def test_paths_resolve_to_project_root():
    assert config.PROJECT_ROOT.exists()
    assert config.PROJECT_ROOT.name == "traffic-forecasting"
    assert config.RAW_PATH.name == "traffic_data.csv"
    assert config.PROCESSED_DIR.name == "processed"
    assert config.REPORTS_DIR.name == "reports"
    assert config.MODELS_DIR.name == "models"


def test_feature_cols_match_lstm_feature_cols_are_subset():
    assert set(config.LSTM_FEATURE_COLS).issubset(set(config.FEATURE_COLS))


def test_rf_grid_contains_expected_knobs():
    for key in ("n_estimators", "max_depth", "min_samples_leaf"):
        assert key in config.RF_PARAM_GRID


def test_lstm_hyperparams_present():
    for key in ("hidden", "layers", "lr", "batch_size", "epochs", "seq_len"):
        assert key in config.LSTM_HYPERPARAMS


def test_seq_len_is_24():
    assert config.SEQ_LEN == 24
    assert config.LSTM_HYPERPARAMS["seq_len"] == 24


def test_holidays_are_datetimes():
    import pandas as pd

    assert len(config.HOLIDAYS) == 13
    assert pd.api.types.is_datetime64_any_dtype(config.HOLIDAYS)

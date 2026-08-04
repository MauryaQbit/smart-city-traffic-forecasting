# Data Dictionary

Columns used through the traffic forecasting pipeline. Raw columns are written
to `data/raw/traffic_data.csv` by `tf-generate-data`. Engineered columns are
produced by `traffic_forecast.features.engineer_features` and consumed by the
model stages.

## Raw columns (`data/raw/traffic_data.csv`)

| Column | Dtype | Units | Source | Notes |
|---|---|---|---|---|
| `DateTime` | datetime64[ns] | hourly timestamp | generator | spans `2022-01-01 00:00` to `2023-12-31 23:00`, 17,520 rows per junction |
| `Junction` | int64 | junction id (1-4) | generator | junction 1 is busiest, junction 4 is quietest |
| `Vehicles` | float64 | vehicle count per hour | generator | ~0.1% of values are missing on purpose to mirror real-world data quality; the pipeline forward-fills them per junction before feature engineering |
| `ID` | string | row id | generator | `DateTime.strftime('%Y%m%d%H') + str(Junction)`, useful for debugging |

70,080 rows total (4 junctions x 17,520 hours). When swapping in real data,
match these four column names; everything downstream is column-name driven.

## Engineered columns (`data/processed/features.csv`)

All engineered columns are computed inside `engineer_features`. The first 168
hours per junction are dropped so every lag/rolling feature has full context,
leaving 69,408 rows.

### Temporal (deterministic from `DateTime`)

| Column | Dtype | Range | Notes |
|---|---|---|---|
| `hour` | int64 | 0-23 | kept raw; useful for trees |
| `dayofweek` | int64 | 0-6 (Mon-Sun) | kept raw |
| `month` | int64 | 1-12 | kept raw |
| `is_weekend` | int64 | 0 or 1 | 1 if `dayofweek >= 5` |
| `is_holiday` | int64 | 0 or 1 | 1 if the calendar day is in `config.HOLIDAYS` |
| `hour_sin`, `hour_cos` | float | [-1, 1] | `sin(2*pi*hour/24)`, `cos(2*pi*hour/24)` - cyclical encoding so linear models see hour 23 and hour 0 as adjacent |
| `hour_sin_2`, `hour_cos_2` | float | [-1, 1] | second harmonic (`/12`), captures the morning/evening rush asymmetry |
| `dow_sin`, `dow_cos` | float | [-1, 1] | `sin(2*pi*dayofweek/7)`, `cos(2*pi*dayofweek/7)` |

### Lag features (per-junction `shift`)

| Column | Dtype | Notes |
|---|---|---|
| `lag_1` | float64 | vehicle count 1 hour before |
| `lag_24` | float64 | vehicle count 24 hours before (same hour yesterday) |
| `lag_168` | float64 | vehicle count 168 hours before (same hour last week) |

### Rolling features (per-junction, `shift(1).rolling(...).mean()`)

The `shift(1)` before `rolling()` ensures the rolling window never includes
the current target hour (no leakage).

| Column | Dtype | Window | Notes |
|---|---|---|---|
| `roll_mean_3` | float64 | 3 hours | short-term level |
| `roll_mean_24` | float64 | 24 hours | daily level |
| `roll_mean_168` | float64 | 168 hours (7 days) | weekly level |

### Other

| Column | Dtype | Notes |
|---|---|---|
| `is_outlier` | bool / int | per-junction IQR flag, kept as a feature rather than removed (matches the Week 2 EDA decision). The pipeline computes IQR on the full frame; the `engineer_features(iqr_thresholds=...)` parameter lets you fit on train only if leakage is a concern for your application. |
| `Junction` | int64 | carried through to feature set for tree models and as a one-hot for LR |

## Per-model views

Two parallel views of the engineered frame are produced:

- `make_linear_view(train, val)` - one-hot encodes `Junction`, `StandardScaler`s
  every numeric column, fit on train only. Used by Linear Regression.
- `make_tree_view(train, val)` - integer `Junction`, no scaling. Used by
  Random Forest and the dashboard's predictor form.
- The LSTM consumes the engineered frame directly with `MinMaxScaler` on
  features and target, fit on train only.

## Per-stage artifacts

| Path | Producer | Notes |
|---|---|---|
| `data/raw/traffic_data.csv` | `tf-generate-data` | raw input |
| `data/processed/features.csv` | `features.py __main__` | engineered features (written when run as a script) |
| `data/processed/train_df.pkl`, `val_df.pkl` | `tf-train-stage1` (via `_common.load_and_split`) | chronological 80/20 split |
| `models/linear_regression.joblib`, `lr_scaler.joblib` | `tf-train-stage1` | LR model + fitted StandardScaler |
| `models/random_forest_tuned.joblib` | `tf-train-stage1` | best-tuned RF (≈60 MB) |
| `models/lstm.pt` | `tf-train-stage3` | LSTM weights |
| `reports/pred_linear.npy`, `pred_rf_tuned.npy` | stage1 | per-row validation predictions |
| `reports/pred_lstm.npy`, `actual_lstm.npy` | stage3 | LSTM predictions and aligned actuals (first 24 val hours per junction dropped for sequence context) |
| `reports/sarima_predictions.json` | stage2 | per-junction forecast payload for the 2-week eval window |
| `reports/results_stage{1,2_sarima,3_lstm}.json` | stages 1-3 | per-model MAE, RMSE, train_seconds |
| `reports/best_rf_params.json` | stage1 | grid-search winner |
| `reports/model_comparison.csv` | stage4 | combined leaderboard, sorted by MAE |
| `reports/predictions_full.csv` | stage4 | dashboard input: per-row actuals + all 4 models' predictions |
| `reports/predictions_sample.csv` | stage4 | first 14 days of val (window where all 4 models have predictions) |
| `reports/error_by_junction_hour.csv` | stage4 | mean abs error of RF, grouped by junction x hour |

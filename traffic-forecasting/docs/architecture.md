# Architecture

End-to-end data flow for the traffic forecasting pipeline. Generates data,
engineers features, trains four models, combines their predictions, and
serves an interactive dashboard.

## High-level flow

```mermaid
flowchart TD
    subgraph Data
        RAW[data/raw/traffic_data.csv<br/>70,080 rows]
        FEAT[data/processed/features.csv<br/>69,408 rows]
        SPLIT[data/processed/<br/>train_df.pkl, val_df.pkl]
    end

    subgraph Models
        LR[models/lr.py<br/>LinearRegression]
        RF[models/rf.py<br/>RandomForest default + tuned]
        SARIMA[models/sarima.py<br/>per-junction SARIMAX]
        LSTM[models/lstm.py<br/>TrafficLSTM]
    end

    subgraph Reports
        S1[reports/results_stage1.json<br/>pred_linear.npy, pred_rf_tuned.npy]
        S2[reports/results_stage2_sarima.json<br/>sarima_predictions.json]
        S3[reports/results_stage3_lstm.json<br/>pred_lstm.npy, actual_lstm.npy]
        S4[reports/model_comparison.csv<br/>predictions_full.csv<br/>error_by_junction_hour.csv]
    end

    subgraph UI
        DASH[dashboard/app.py<br/>Streamlit 4 tabs]
    end

    GEN[tf-generate-data<br/>data/generate.py] --> RAW
    RAW --> FEATURES[features.engineer_features<br/>features.py]
    FEATURES --> FEAT
    FEAT --> SPLITCHRON[features.chronological_split<br/>80/20 chronological] --> SPLIT

    SPLIT --> STAGE1[pipeline/stage1.py]
    SPLIT --> STAGE2[pipeline/stage2.py]
    SPLIT --> STAGE3[pipeline/stage3.py]
    SPLIT --> STAGE4[pipeline/stage4.py]

    STAGE1 --> LR
    STAGE1 --> RF
    LR --> S1
    RF --> S1

    STAGE2 --> SARIMA
    SARIMA --> S2

    STAGE3 --> LSTM
    LSTM --> S3

    S1 --> STAGE4
    S2 --> STAGE4
    S3 --> STAGE4
    STAGE4 --> S4

    S4 --> DASH
```

## Package layout

```mermaid
flowchart LR
    subgraph src/traffic_forecast/
        CFG[config.py<br/>paths, seeds, hyperparams]
        DATA[data/<br/>generate.py]
        FEAT[features.py]
        MODELS[models/<br/>lr.py, rf.py, sarima.py, lstm.py]
        EVAL[eval/<br/>metrics.py]
        PIPE[pipeline/<br/>_common.py, stage1-4, run_all]
        DASH[dashboard/<br/>app.py, theme.py]
        MAIN[__main__.py]
    end
    CFG -.-> DATA
    CFG -.-> FEAT
    CFG -.-> MODELS
    CFG -.-> PIPE
    CFG -.-> DASH
    FEAT --> MODELS
    EVAL --> MODELS
    MODELS --> PIPE
    PIPE --> DASH
```

## Console scripts

| Command | What it does |
|---|---|
| `tf-generate-data` | writes `data/raw/traffic_data.csv` |
| `tf-train-stage1` | LR + RF (default + tuned) |
| `tf-train-stage2` | SARIMA per junction |
| `tf-train-stage3` | LSTM (target-scaled, grad-clipped, early-stopped) |
| `tf-combine` | merges reports + builds dashboard inputs |
| `tf-train-all` | stages 1 -> 4 in one process |
| `tf-dashboard` | launches Streamlit |

All commands also available via `python -m traffic_forecast {generate-data|stage1|...}`.

## Determinism

Random sources and how they're handled:

| Source | Seeded? | Where |
|---|---|---|
| Data generation noise, spikes, missing-value mask | yes | `data/generate.py::build_dataset` (reentrant - seed constructed inside the function) |
| RF default + tuned | yes | `random_state=42` on every `RandomForestRegressor` constructor |
| RF grid-search subsample | yes | `np.random.default_rng(0)` in `models/rf.py::train_tuned` |
| SARIMA L-BFGS-B optimization | deterministic | fixed init, no shuffling |
| LSTM | yes | `torch.manual_seed(42); torch.set_num_threads(1)` in `models/lstm.py::train` |

`tf-train-all` from a clean checkout reproduces the metrics within the bands
enforced by `tests/characterization/`.

## Test pyramid

```mermaid
flowchart BT
    UNIT[tests/unit/<br/>~30 tests, <60s<br/>pure functions]
    INT[tests/integration/<br/>stage runs on a tiny fixture]
    SMOKE[tests/smoke/<br/>full pipeline on tiny fixture]
    CHAR[tests/characterization/<br/>real CSV, locks current metrics]

    UNIT --> INT --> SMOKE --> CHAR
```

- **Fast loop** (`pytest -m "not slow"`): unit + integration + smoke. ~60s.
- **Slow loop** (`pytest`): adds characterization. ~3 min. Locks the current
  metrics within +-10% bands so a refactor or library bump cannot silently
  regress the results.
- **CI**: ruff + fast tests on Python 3.10 / 3.11 / 3.12 (matrix),
  characterization on push events only.

# Smart City Traffic Forecasting

End-to-end build of Project 9 (ML Internship), based on the Week 1-4 progress
reports: forecasting hourly vehicle counts at 4 city junctions using Linear
Regression, Random Forest, SARIMA, and an LSTM, plus a Streamlit dashboard to
explore the results.

## What's included

```
traffic-forecasting/
├── src/traffic_forecast/           # installable python package
│   ├── config.py                   # paths, seeds, hyperparams (single source)
│   ├── data/generate.py            # synthetic dataset generator
│   ├── features.py                 # feature engineering + chronological split
│   ├── models/                     # lr.py, rf.py, sarima.py, lstm.py
│   ├── eval/metrics.py             # mae, rmse
│   ├── pipeline/                   # stage1..4 + run_all orchestrators
│   └── dashboard/app.py            # Streamlit app
├── tests/                          # unit, integration, characterization, smoke
├── data/                           # raw + processed (gitignored)
├── models/                         # trained artifacts (gitignored)
├── reports/                        # metrics, predictions, error breakdowns
├── docs/                           # design specs + implementation plans
├── pyproject.toml                  # package metadata, deps, console scripts
└── .github/workflows/ci.yml        # ruff + pytest on py 3.10/3.11/3.12
```

## Why synthetic data

The original dataset referenced in the reports (DateTime / Junction / Vehicle
count, from the traffic-forecasting hackathon/Kaggle-style dataset) wasn't
reachable from the environment this was built in, so `data/generate.py`
builds a realistic stand-in with the same schema and the same behavior your
EDA found: daily + weekly seasonality, junction-specific volume levels,
holiday dips, and occasional genuine spikes. **Swap in the real CSV** (same
column names) and everything downstream — features, models, dashboard —
works unchanged. Just drop your real file at `data/raw/traffic_data.csv`
and skip running `tf-generate-data`.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -e ".[lstm,dashboard,dev]"
```

## Run the full pipeline

```bash
tf-generate-data     # (skip if using your own real dataset)
tf-train-stage1      # Linear Regression + Random Forest (~5 min, mostly grid search)
tf-train-stage2      # SARIMA, per junction (~1 min)
tf-train-stage3      # LSTM (~2 min on CPU)
tf-combine           # builds reports/model_comparison.csv + dashboard data
```

Or run everything in one go with `tf-train-all`. Equivalently,
`python -m traffic_forecast all`.

## Launch the dashboard

```bash
tf-dashboard
```

> The dashboard reads `reports/predictions_full.csv`,
> `reports/model_comparison.csv`, `reports/error_by_junction_hour.csv`, and
> `models/random_forest_tuned.joblib`. If you haven't run the training
> pipeline yet, run `tf-train-all` first or the dashboard will throw
> `FileNotFoundError` on launch.

Opens at `http://localhost:8501` with four tabs:
- **Model Comparison** — MAE/RMSE table + bar chart across all 4 models
- **Predicted vs Actual** — pick a junction and models, see forecasts over any date range, holidays marked
- **Error Breakdown** — heatmap of where Random Forest struggles most (junction × hour)
- **Try a Prediction** — live form that calls the trained Random Forest model for a one-off forecast

## Results on this run (validation set)

| Model | MAE | RMSE |
|---|---|---|
| Random Forest (tuned) | 3.17 | 4.41 |
| LSTM | 3.19 | 4.65 |
| Random Forest (default) | 3.27 | 4.53 |
| Linear Regression | 3.66 | 5.01 |
| SARIMA | 4.66 | 6.63 |

The Week 3/4 reports had the LSTM trailing badly. That turned out to be a
training bug (target unscaled, no gradient clipping, no early stopping), not
a modeling ceiling. Fixing those brought it from MAE 11.99 to 3.19, essentially
tied with the tree models at the top.

Feature engineering matters here too: cyclical/Fourier encoding of `hour`
and `dayofweek` plus a weekly `lag_168` cut Linear Regression's MAE from
4.14 to 3.66 (the linear model can't discover nonlinear interactions the
way trees can, so giving it explicit sin/cos features is where the win lives).
Broadening the RF grid (added `max_features` and a 400-tree option) shaved
its MAE from 3.31 to 3.17.

SARIMA captures the daily seasonal shape well but trails on raw error since
its forecast horizon is only 2 weeks (full-history SARIMA refits on hourly
data with a 24-hour seasonal period are prohibitively slow).

## Notes on choices made to keep this runnable

- **SARIMA** is fit on a rolling ~2-month window per junction (not the full
  2 years) and evaluated on a 2-week forecast horizon — full-history SARIMA
  refits on hourly data with a 24-hour seasonal period are extremely slow.
  This mirrors how SARIMA is used in practice (periodic refit on a recent
  window) rather than a single one-shot fit.
- **LSTM** uses a 24-hour input window, 2 layers, 32 hidden units. Target is
  MinMax-scaled before training so the loss landscape behaves; gradients are
  clipped at norm 1.0; best validation-loss state is restored at the end
  (early stopping with patience 4); Junction is one-hot included as an input
  feature so per-junction mean levels are learnable. Trains in ~2 min on CPU.
- **Feature set** includes cyclical encodings of `hour` (sin/cos at periods
  24 and 12) and `dayofweek` (period 7), plus daily and weekly lag/rolling
  features (`lag_1`, `lag_24`, `lag_168`, `roll_mean_3`/`24`/`168`). The
  first 168 hours per junction are dropped so every weekly-lag feature has
  full context.
- **Random Forest grid search** runs on a 12k-row subsample of the training
  set to keep tuning time reasonable, then refits the best params on the
  full training set. The grid covers `n_estimators`, `max_depth`,
  `min_samples_leaf`, and `max_features`.

## Tests

```bash
pytest -m "not slow"   # unit + integration + smoke, fast loop (~60s)
pytest                 # adds the characterization suite (~2.5 min total)
```

Characterization tests under `tests/characterization/` lock in the current
metrics within ±10% (±15% for the LSTM) so a refactor or library bump can't
silently regress the results. The single most important assertion is
`test_pred_lstm_shape_and_non_constant` — the LSTM previously collapsed to
emitting a literal scalar for every input; that test catches a regression.

## Documentation

- [`docs/data_dictionary.md`](docs/data_dictionary.md) - raw and engineered columns
- [`docs/model_card.md`](docs/model_card.md) - per-model details, intended use, caveats
- [`docs/architecture.md`](docs/architecture.md) - Mermaid diagrams of data flow, package layout, test pyramid
- [`CONTRIBUTING.md`](CONTRIBUTING.md) - dev setup, layout, workflow, adding a new model

## Deployment

The dashboard needs the pre-trained artifacts in `models/` and `reports/`.
Two paths:

**Local Docker:**

```bash
docker build -t traffic-forecast .
docker run -p 8501:8501 traffic-forecast
# open http://localhost:8501
```

The image bundles the data, models, and reports from the repo. The trained
Random Forest pickle is ~60 MB so the image is ~1.2 GB; for a smaller image,
drop the `lstm` extra if you don't need the LSTM tab.

**Streamlit Community Cloud:**

1. Push the repo to GitHub. Make sure `requirements.txt`, `models/`, and
   `reports/` are committed (the `.gitignore` keeps the heavy `.npy`/`.joblib`
   files out by default - force-add them or retrain in the cloud).
2. New app on Streamlit Cloud, point at the repo.
3. Main file path: `src/traffic_forecast/dashboard/app.py`.
4. The free tier has 1 GB RAM. The Random Forest alone is ~60 MB in memory;
   if you hit OOM, lower `n_estimators` in `config.py` and retrain.

The dashboard exposes a health check at `/_stcore/health` (returns `ok`).

## Next steps if you want to extend it

- Swap in the real dataset once you can access it.
- Try the per-junction Random Forest variant (Week 4 found it helps the
  busiest junction but hurts the quieter ones with less data). Not yet
  re-implemented in the packaged pipeline.
- Add weather data alongside the holiday flag.
- Deploy the Streamlit app (Streamlit Community Cloud, or a small VM) so
  it's reachable outside your machine for your final presentation/demo.

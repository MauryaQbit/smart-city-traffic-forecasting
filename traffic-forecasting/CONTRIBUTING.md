# Contributing

Contributions welcome. The project is small enough that the rules below are
mostly common sense; read them before opening a PR.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[lstm,dashboard,dev]"
```

Verify the install:

```bash
pytest -m "not slow"   # ~60s
ruff check .
```

## Layout

See `docs/architecture.md` for the package map. Short version:

- `src/traffic_forecast/` - importable package. Don't put scripts here.
- `src/traffic_forecast/models/` - one module per model family. Each module
  exposes a `train(train_df, val_df, ...) -> dict` that returns `model`,
  `pred_val`, and `metrics` (with at least `MAE`, `RMSE`, `train_seconds`).
- `src/traffic_forecast/pipeline/` - thin orchestrators that read pickled
  splits, call `models.*`, and write `reports/*` artifacts. Stages must stay
  scriptable via a `main()` function and `if __name__ == "__main__":` guard.
- `tests/` - unit (fast, pure), integration (stage runs on a tiny fixture),
  smoke (full pipeline on a tiny fixture), characterization (real CSV,
  metric bands). Mark slow tests with `@pytest.mark.slow`.

## Workflow

1. Run the relevant tests before and after your change. If you're touching
   `features.py`, that's `tests/unit/test_features.py` and the smoke test.
2. Add a test for new behavior. Don't widen an existing characterization
   band to make a test pass - update it deliberately and call out the change
   in your PR description.
3. Don't commit generated artifacts (`models/*.joblib`, `reports/*.npy`,
   `data/processed/*`). They're gitignored. If a model file must change,
   retrain via `tf-train-all` and commit the resulting `reports/*.csv` and
   `reports/*.json` (small, dashboard-required).
4. Keep the dashboard importable. Don't add module-level side effects in
   `dashboard/app.py` - everything render-related belongs inside a function
   so the rest of the package can `from traffic_forecast.dashboard import app`
   without booting Streamlit.

## Style

- `ruff check .` and `ruff format --check .` must pass. Run `ruff format .`
  to auto-fix.
- Type hints on public functions. Internal helpers can skip them when the
  type is obvious.
- Comments explain *why*, not *what*. Don't restate the code.
- No emoji in code or commit messages.

## Commit messages

One line, lowercase, no body, no trailers. Examples:

```
lstm: add gradient clipping
features: cyclical hour encoding
fix: train_seconds missing from comparison table
docs: add data dictionary
```

## Adding a new model

1. Create `src/traffic_forecast/models/<name>.py` exposing
   `train(train_df, val_df) -> dict`.
2. Add a new pipeline stage `src/traffic_forecast/pipeline/stageN_<name>.py`
   that loads the pickled split, calls `models.<name>.train`, and writes
   `reports/results_stageN_<name>.json` and `reports/pred_<name>.npy`.
3. Wire it into `pipeline/run_all.py` and `__main__.py`.
4. Update `pipeline/stage4.py` to merge your predictions into
   `predictions_full.csv` as `pred_<name>`.
5. Add `pred_<name>` to `dashboard/theme.py::MODELS` so it appears in the
   predicted-vs-actual selector.
6. Add characterization bands in `tests/characterization/test_model_metrics.py`
   and a shape check in `tests/characterization/test_prediction_arrays.py`.

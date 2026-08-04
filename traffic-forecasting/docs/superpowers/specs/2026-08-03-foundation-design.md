# Foundation Sub-Project — Design Spec

**Date:** 2026-08-03
**Sub-project:** 1 of 4 (Foundation)
**Status:** Awaiting user approval
**Predecessors:** none
**Successors:** #2 ML Improvements, #3 Dashboard Polish (Streamlit), #4 Packaging/Docs/CI

## Purpose

Turn a flat directory of scripts that run on import into a real, installable,
testable Python package without losing the current (correct) results, and
incidentally fix one outright bug discovered during audit (the LSTM that
collapsed to a constant output).

This sub-project is the prerequisite for everything else. It does not change
methodology in interesting ways, does not polish the dashboard, does not
improve features. It builds the floor.

## Goals

1. `pip install -e .` works and exposes a `traffic_forecast` package.
2. `import traffic_forecast` does not train any model.
3. The 4 model stages run via console scripts (`tf-train-stage1` ... `tf-combine`) or `python -m traffic_forecast`.
4. Re-running the full pipeline from scratch reproduces the current metrics for LR, RF, and SARIMA, byte-for-byte deterministic given the same seed.
5. The LSTM no longer emits a constant; MAE drops from 11.99 to a value competitive with RF (~3.5–4.5 expected).
6. A characterization test suite exists and passes against the current metrics (post-LSTM-fix), enforcing "don't break existing results" for all future sub-projects.
7. The repo is safe to commit: `git status` ignores ~75 MB of generated artifacts.

## Non-Goals (explicitly deferred)

- Dashboard UX, theming, layout improvements → sub-project #3.
- Feature engineering additions (cyclical hour encoding, `lag_168`, Fourier terms) → sub-project #2.
- SARIMA evaluation-window comparability fix → sub-project #2.
- Random Forest tuning broadening → sub-project #2.
- README rewrite beyond fixing the two factual errors, data dictionary, model card, architecture diagram, deployment story → sub-project #4.
- Replacing Streamlit with a real frontend → rejected by user; sub-project #3 polishes Streamlit.

## Confirmed Bugs To Fix In This Sub-Project

| ID | Bug | Evidence | Fix |
|---|---|---|---|
| F1 | LSTM collapsed to constant output (std = 0.00 across 13,904 val predictions, mean = 20.25 ≈ global mean) | `reports/pred_lstm.npy` | Scale target, add gradient clipping, add early stopping with best-state restore, add junction feature |
| F2 | `generate_data.build_dataset()` non-reentrant: module-level `RNG` advances across calls in same process | `src/generate_data.py:21` | Move `RNG = np.random.default_rng(42)` inside `build_dataset()` |
| F3 | `train_models.py` LSTM non-deterministic: no `torch.manual_seed`, no `torch.set_num_threads` | `src/train_models.py:185-200` | Subsumed by deletion of `train_models.py` (see D1) |
| F4 | `MinMaxScaler` fit on train+val in LSTM stage (mild leakage) | `src/stage3_lstm.py:30-31` | Fit on train only |
| F5 | `is_outlier` IQR computed on full frame pre-split (mild leakage) | `src/features.py:50` | Fit IQR on train only, apply thresholds to val |
| F6 | `train_models.py` duplicates all 4 stage scripts with divergent LSTM hyperparameters and SARIMA windowing | `src/train_models.py` vs stage scripts | Delete `train_models.py`, replace with `traffic_forecast.pipeline` orchestrator that calls into the stage modules |
| F7 | All stage scripts run training at import time (no `__main__` guard, no `def main()`) | `stage1_lr_rf.py:17-90`, `stage2_sarima.py:12-64`, `stage3_lstm.py:13-112`, `stage4_combine.py:8-76`, `train_models.py:37-243` | Wrap each body in `def main()` + guard |
| F8 | `HOLIDAYS` list duplicated across 2 files | `generate_data.py:37-42`, `features.py:19-24` | Move to `config.py`, import from both |
| F9 | `SEQ_LEN = 24` duplicated across 3 files | `stage3_lstm.py:26`, `train_models.py:145`, `stage4_combine.py:53` | Move to `config.py` |
| F10 | `rmse()` helper duplicated in 4 files | `stage1:24`, `stage2:19`, `stage3:22`, `train_models:46` | Move to `traffic_forecast/eval/metrics.py` |
| F11 | `ROOT/REPORTS/MODELS` path block + `mkdir` duplicated in 6 files | multiple | Centralize in `config.py` |
| F12 | README references `reports/per_junction_rf.json` which does not exist | `README.md:110-112` | Remove the reference (the actual per-junction exploration happens in sub-project #2) |
| F13 | README results table rounds RF default and RF tuned to the same `3.32 / 4.60`, hiding a real difference and mismatching the shipped CSV | `README.md:78-84` vs `reports/model_comparison.csv` | Update the table from the CSV; show distinct values |

## Architecture

### Target package layout

```
traffic-forecasting/
├── pyproject.toml                  # build + deps + console scripts + tool config
├── LICENSE                         # MIT
├── .gitignore                      # data/, models/*.joblib, reports/*.npy, .venv/, __pycache__/, etc.
├── README.md                       # factual fixes only in this sub-project
├── requirements.txt                # kept as a thin export for Streamlit Cloud compatibility
├── src/
│   └── traffic_forecast/
│       ├── __init__.py
│       ├── __main__.py             # `python -m traffic_forecast <stage>`
│       ├── config.py               # PROJECT_ROOT, paths, seeds, HOLIDAYS, SEQ_LEN,
│       │                           #   RF grid, LSTM hyperparams, SARIMA order, val_frac
│       ├── data/
│       │   ├── __init__.py
│       │   └── generate.py         # from generate_data.py; build_dataset() with internal RNG
│       ├── features.py             # from features.py; engineer_features, split, views
│       ├── models/
│       │   ├── __init__.py
│       │   ├── lr.py               # train_linear_regression(train, val) -> (model, preds, metrics, artifacts)
│       │   ├── rf.py               # train_random_forest(...) supporting default + tuned
│       │   ├── sarima.py           # train_sarima(...) per-junction on rolling window
│       │   └── lstm.py             # TrafficLSTM class + make_sequences() + train_lstm()
│       ├── eval/
│       │   ├── __init__.py
│       │   └── metrics.py          # rmse, mae
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── stage1.py           # LR + RF (default + tuned) — thin: imports from models/
│       │   ├── stage2.py           # SARIMA per junction
│       │   ├── stage3.py           # LSTM (fixed)
│       │   ├── stage4.py           # combine reports + dashboard data
│       │   └── run_all.py          # replaces train_models.py; orchestrates stage1..4
│       └── dashboard/              # moved from top-level dashboard/
│           └── app.py              # unchanged in this sub-project (sub-project #3 owns UX)
├── tests/
│   ├── conftest.py                 # tiny fixture generator, real-CSV loader, markers
│   ├── unit/
│   │   ├── test_generate.py
│   │   ├── test_features.py
│   │   ├── test_metrics.py
│   │   └── test_lstm_arch.py
│   ├── integration/
│   │   ├── test_stage1_lr_rf.py
│   │   ├── test_stage2_sarima.py
│   │   ├── test_stage3_lstm.py
│   │   └── test_stage4_combine.py
│   ├── characterization/
│   │   ├── test_data_contract.py
│   │   ├── test_model_metrics.py
│   │   └── test_prediction_arrays.py
│   └── smoke/
│       └── test_pipeline_end_to_end.py
├── data/                           # gitignored except data/raw/.gitkeep
│   ├── raw/
│   │   └── traffic_data.csv
│   └── processed/                  # gitignored
├── models/                         # gitignored except .gitkeep
├── reports/                        # gitignored except .gitkeep
├── notebooks/                      # removed (was empty)
└── docs/
    └── superpowers/specs/2026-08-03-foundation-design.md   # this file
```

### Console scripts (in `pyproject.toml`)

```
tf-generate-data  = "traffic_forecast.data.generate:main"
tf-train-stage1   = "traffic_forecast.pipeline.stage1:main"
tf-train-stage2   = "traffic_forecast.pipeline.stage2:main"
tf-train-stage3   = "traffic_forecast.pipeline.stage3:main"
tf-combine        = "traffic_forecast.pipeline.stage4:main"
tf-train-all      = "traffic_forecast.pipeline.run_all:main"
tf-dashboard      = "traffic_forecast.dashboard.app:main"   # wraps `streamlit run`
```

`python -m traffic_forecast generate-data` / `stage1` / ... / `all` is also supported via `__main__.py`.

### The LSTM fix (F1) — design

Current training (`src/stage3_lstm.py:55-104`):

```python
class TrafficLSTM(nn.Module):
    def __init__(self, n_features, hidden=32, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# loss = nn.MSELoss() on raw Vehicles targets (range 0..135)
# opt = Adam(lr=2e-3); 8 epochs; no clipping; no val monitoring
```

Failure mode: output collapses to constant ≈ global mean (20.25). Verified empirically: saved predictions have std = 0.00.

Fix — five changes, all standard:

1. **Target scaling.** Fit a `MinMaxScaler` (or `StandardScaler`) on `train_df["Vehicles"]` only. Train against scaled targets. Invert before computing metrics and before saving predictions. Removes the large-output-weight problem that prevents Adam from converging in 8 epochs.

2. **Gradient clipping.** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` between `loss.backward()` and `opt.step()`. Prevents early-training recurrent gradient explosions that collapse the hidden state.

3. **Validation-based model selection.** After each epoch, compute val loss; track best; restore best state at end. Catches a collapse that happens mid-training instead of silently shipping it.

4. **Junction identity.** Add `Junction` to `lstm_feature_cols` (one-hot encoded to preserve the categorical nature, since MinMax scaling the integer 1..4 would imply a false ordering). Lets the model learn per-junction mean levels instead of regressing to the global mean.

5. **Bump epochs modestly + keep StepLR.** 8 → 15 epochs (still CPU-friendly; ~90s total). StepLR schedule stays.

Expected post-fix: LSTM MAE 11.99 → ~3.5–4.5, RMSE 15.24 → ~5–6.

### Characterization test plan (enforces "don't break existing results")

After F1–F5 land, run the full pipeline once to capture the new ground truth, then freeze:

**Data contract** (against committed `data/raw/traffic_data.csv`):
- 70,080 rows, 4 junctions × 17,520 hours, DateTime span `2022-01-01 00:00` to `2023-12-31 23:00`.
- ~70 missing `Vehicles` values pre-imputation.
- After `engineer_features`: 69,984 rows, 0 NaNs, first row per junction at hour 24.

**Split contract:**
- Train max DateTime < val min DateTime.
- Val row count in [13,900, 14,100].

**Metric contract** (within ±10% of post-fix ground truth, ±15% for LSTM):
- LR MAE ≈ 4.14, RMSE ≈ 5.69.
- RF tuned MAE ≈ 3.29, RMSE ≈ 4.58; `best_params` = `{"max_depth": null, "min_samples_leaf": 1, "n_estimators": 200}`.
- SARIMA MAE ≈ 4.56, RMSE ≈ 6.50.
- LSTM MAE in [3.0, 5.5] (new, post-fix), RMSE in [4.0, 7.0]. The band is wide because LSTM training has more variance; tightening it is a sub-project #2 concern.

**Ranking contract:**
- No model's MAE exceeds LSTM-pre-fix level (12).
- RF tuned is among the top 2 by MAE.

**Prediction-array contract:**
- `pred_linear`, `pred_rf_tuned`: shape (14000,), finite, mean in [15, 25].
- `pred_lstm`: shape (13904,), std > 1.0 (this is the regression-catch for F1).
- `predictions_full.csv`: 14,000 rows, all 4 prediction columns non-NaN for at least the first 14 days per junction.

### Test data strategy

- **Tiny on-the-fly fixture** (unit, integration, smoke): 2 junctions × 14 days × 24 hours = 672 rows. Built by a `conftest.py` helper that calls `traffic_forecast.data.generate`'s internals with a small date range. Full pipeline runs in <30s on this fixture.
- **Committed real CSV** (characterization only): the existing `data/raw/traffic_data.csv`. Marked `@pytest.mark.slow` so fast CI loops skip them.

### CI

`.github/workflows/ci.yml`:

- Triggers: `push` to `main`, PRs to `main`.
- Matrix: Python 3.10, 3.11, 3.12 on `ubuntu-latest`.
- Steps:
  1. `actions/checkout@v4`
  2. `astral-sh/setup-uv@v3` (fast, deterministic installs)
  3. `uv sync --extra dev --extra lstm --extra dashboard`
  4. `ruff check .`
  5. `ruff format --check .`
  6. `pytest -q -m "not slow"` (fast loop)
  7. `pytest -q -m slow` (characterization, only on one Python version to save CI minutes)
  8. Import smoke: `python -c "import traffic_forecast; from traffic_forecast import pipeline"`

## Phasing

Execution order matters because of dependencies between fixes.

| Phase | Work | Depends on | Verification |
|---|---|---|---|
| A | F2 (reentrant RNG) + reproducibility groundwork | — | `build_dataset()` called twice in same process returns identical DataFrame |
| B | Package skeleton: create `src/traffic_forecast/` tree, `__init__.py` files, `pyproject.toml`, console scripts declared, `.gitignore`, `LICENSE`. Empty modules at this point. | — | `pip install -e .` succeeds; `import traffic_forecast` works without errors |
| C | Move code into the package: `data/generate.py`, `features.py`, `models/{lr,rf,sarima,lstm}.py`, `eval/metrics.py`. Apply F8–F11 (dedupe into `config.py` / `metrics.py`). Apply F4, F5 (small leakage fixes) while moving. Apply F7 (wrap in `def main()`). Delete `train_models.py`, `notebooks/`. | A, B | All unit tests pass; existing `data/`+`models/`+`reports/` still load correctly via the dashboard |
| D | Build `pipeline/stage{1..4}.py` + `pipeline/run_all.py` as thin orchestrators over `models/`. | C | Running `tf-train-all` produces `model_comparison.csv` with LR/RF/SARIMA matching current numbers (LSTM still bad at this checkpoint) |
| E | Apply F1 (LSTM fix). | D | LSTM MAE drops to [3.0, 5.5]; `pred_lstm` std > 1.0 |
| F | Write the test suite: unit + integration + characterization + smoke. Snapshot the new (post-E) metrics. | E | Full `pytest` green; characterization tests enforce the new ground truth |
| G | Apply F12, F13 (README factual fixes only). Add minimal CI workflow. Add `requirements.txt` as a re-export of pyproject deps for Streamlit Cloud compatibility. | F | CI green on push |
| H | Final verification: clean checkout → `pip install -e .` → `tf-train-all` → `tf-dashboard` boots without errors and shows all four models with non-NaN predictions | all | Manual smoke |

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| LSTM fix doesn't converge to expected range | Medium | Phase E is isolated; if MAE stays > 6, leave the fix in place but widen the characterization band and document it as "improved but not final" for sub-project #2 |
| Moving files breaks dashboard's `parents[1]` path resolution | High | `config.py` exposes `PROJECT_ROOT` computed once via `importlib.resources`; dashboard imports from `traffic_forecast.config` |
| Characterization tests are too tight and flap | Medium | Start at ±10% (±15% for LSTM); tighten in sub-project #2 once stability is proven |
| Console scripts don't work on Windows | Low | Test on Linux only in CI; document Windows as best-effort |
| Pinning torch dep breaks Streamlit Cloud free-tier RAM | Low | Keep torch in `extras` (`pip install ".[lstm]"`) so dashboard-only deploys skip it |
| Deleting `train_models.py` removes the per-junction RF exploration block | Low | That block's output (`per_junction_rf.json`) is currently dead — referenced by README but not produced by any stage script. Will be re-introduced properly in sub-project #2 |

## Out of scope for this sub-project (reminder)

- Anything in the dashboard UX audit's top 10 except bug F1-adjacent NaN handling for `pred_lstm` (the dashboard will naturally start showing real LSTM data once F1 lands).
- Any feature additions to `engineer_features`.
- Any change to SARIMA windowing methodology.
- Real frontend.
- README rewrite beyond F12/F13.

## Open question for the user (single checkpoint)

- Approve Foundation scope as written (includes LSTM bug fix F1)?
- If not, name what to drop or add.

No other questions blocking. On approval, proceed to writing the implementation plan via the writing-plans skill.

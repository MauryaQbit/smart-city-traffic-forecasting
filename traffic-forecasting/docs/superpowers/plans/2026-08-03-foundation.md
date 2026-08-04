# Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the flat `src/` script directory into an installable `traffic_forecast` package with tests, dedup all shared code, fix the LSTM that collapsed to a constant output, and lock current behavior behind characterization tests so future sub-projects cannot regress it.

**Architecture:** src-layout Python package (`src/traffic_forecast/`) with submodules per concern (`config`, `data`, `features`, `models/{lr,rf,sarima,lstm}`, `eval`, `pipeline`, `dashboard`). Each pipeline stage is a thin orchestrator that imports model implementations and writes the same `reports/` and `models/` artifacts as today, so the dashboard keeps working unchanged. Console scripts (`tf-train-stage1`, `tf-train-all`, `tf-dashboard`, etc.) replace the `cd src && python stage1_lr_rf.py` pattern.

**Tech Stack:** Python ≥3.10, pandas, scikit-learn, statsmodels, PyTorch (CPU-only), Streamlit, Plotly, pytest, ruff. Build via `pyproject.toml` (setuptools). Install via `uv` or `pip`.

**Spec:** `docs/superpowers/specs/2026-08-03-foundation-design.md`

## Global Constraints

- Project root for all relative paths in this plan: `/home/ambi/Desktop/my-test-proj/traffic-forecasting/` (referred to as `<root>` below).
- CPU-only. No `device="cuda"` branches anywhere.
- Synthetic data stays. `data/raw/traffic_data.csv` and `data/processed/*.pkl` are preserved as-is through Task 5; only regenerated if a test demands it.
- LR / RF / SARIMA metrics must not change. LSTM metrics will improve (F1).
- No comments in code unless explaining a non-obvious WHY. Match existing docstring style.
- Commit messages: lowercase, one line, no body, no emojis. Examples: `package: scaffold traffic_forecast`, `lstm: scale target and add grad clipping`.
- Existing files in `src/` (`generate_data.py`, `features.py`, `stage1_lr_rf.py`, `stage2_sarima.py`, `stage3_lstm.py`, `stage4_combine.py`, `train_models.py`) are preserved untouched through Task 8 so the dashboard keeps working during the migration. Task 9 deletes them.
- Tests run via `pytest` from `<root>`. Fast tests must complete in <30s on CPU.

---

## File Structure (final state after all tasks)

```
<root>/
├── pyproject.toml                          # Task 1
├── LICENSE                                 # Task 1 (MIT)
├── .gitignore                              # Task 1
├── README.md                               # Task 12 (factual fixes)
├── requirements.txt                        # Task 12 (Streamlit Cloud re-export)
├── .github/workflows/ci.yml                # Task 12
├── src/
│   ├── traffic_forecast/
│   │   ├── __init__.py                     # Task 1
│   │   ├── __main__.py                     # Task 9
│   │   ├── config.py                       # Task 3
│   │   ├── data/
│   │   │   ├── __init__.py                 # Task 2
│   │   │   └── generate.py                 # Task 2
│   │   ├── features.py                     # Task 5
│   │   ├── models/
│   │   │   ├── __init__.py                 # Task 6
│   │   │   ├── lr.py                       # Task 6
│   │   │   ├── rf.py                       # Task 6
│   │   │   ├── sarima.py                   # Task 7
│   │   │   └── lstm.py                     # Task 8
│   │   ├── eval/
│   │   │   ├── __init__.py                 # Task 4
│   │   │   └── metrics.py                  # Task 4
│   │   ├── pipeline/
│   │   │   ├── __init__.py                 # Task 9
│   │   │   ├── stage1.py                   # Task 9
│   │   │   ├── stage2.py                   # Task 9
│   │   │   ├── stage3.py                   # Task 9
│   │   │   ├── stage4.py                   # Task 9
│   │   │   └── run_all.py                  # Task 9
│   │   └── dashboard/
│   │       ├── __init__.py                 # Task 10
│   │       └── app.py                      # Task 10 (moved from <root>/dashboard/)
│   └── (legacy scripts deleted in Task 9)
├── tests/
│   ├── conftest.py                         # Task 11
│   ├── unit/
│   │   ├── test_generate.py                # Task 2
│   │   ├── test_metrics.py                 # Task 4
│   │   ├── test_config.py                  # Task 3
│   │   ├── test_features.py                # Task 5
│   │   ├── test_lr.py                      # Task 6
│   │   ├── test_rf.py                      # Task 6
│   │   ├── test_sarima.py                  # Task 7
│   │   └── test_lstm.py                    # Task 8
│   ├── integration/
│   │   ├── test_stage1.py                  # Task 9
│   │   ├── test_stage2.py                  # Task 9
│   │   ├── test_stage3.py                  # Task 9
│   │   ├── test_stage4.py                  # Task 9
│   │   └── test_run_all.py                 # Task 9
│   ├── characterization/
│   │   ├── test_data_contract.py           # Task 11
│   │   ├── test_model_metrics.py           # Task 11
│   │   └── test_prediction_arrays.py       # Task 11
│   └── smoke/
│       └── test_pipeline_end_to_end.py     # Task 11
├── data/  (gitignored except .gitkeep)
├── models/ (gitignored except .gitkeep)
├── reports/ (gitignored except .gitkeep)
└── docs/superpowers/{specs,plans}/...
```

---

### Task 1: Package skeleton + git init

**Files:**
- Create: `<root>/.gitignore`
- Create: `<root>/LICENSE`
- Create: `<root>/pyproject.toml`
- Create: `<root>/src/traffic_forecast/__init__.py`
- Create: `<root>/src/traffic_forecast/py.typed` (empty)
- Create: `<root>/data/raw/.gitkeep`
- Create: `<root>/data/processed/.gitkeep`
- Create: `<root>/models/.gitkeep`
- Create: `<root>/reports/.gitkeep`

**Interfaces:**
- Consumes: nothing
- Produces: importable `traffic_forecast` package; git repo initialized

- [ ] **Step 1: Initialize git repo**

```bash
cd <root>
git init
git config user.email "dev@local" || true
git config user.name "dev" || true
```

- [ ] **Step 2: Write `.gitignore`**

Content (exact):

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/
.eggs/

# Virtual envs
.venv/
venv/
env/

# Test/lint caches
.pytest_cache/
.ruff_cache/
.mypy_cache/
.coverage
htmlcov/

# Generated data (regenerate via `tf-generate-data`)
data/raw/*.csv
data/processed/

# Trained models (regenerate via `tf-train-all`)
models/*.joblib
models/*.pt
models/*.pkl

# Predictions / arrays (regenerate via `tf-train-all`)
reports/*.npy

# Streamlit
.streamlit/secrets.toml

# OS
.DS_Store
Thumbs.db

# Editor
.vscode/
.idea/
```

- [ ] **Step 3: Write `LICENSE` (MIT)**

Content (exact, with year 2026 and author "Project Owner"):

```
MIT License

Copyright (c) 2026 Project Owner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Write `pyproject.toml`**

Content (exact):

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "traffic-forecast"
version = "0.1.0"
description = "Smart City Traffic Forecasting - hourly vehicle counts at city junctions"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "Project Owner" }]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scikit-learn>=1.3",
    "statsmodels>=0.14",
    "joblib>=1.3",
]

[project.optional-dependencies]
lstm = ["torch>=2.0"]
dashboard = ["streamlit>=1.30", "plotly>=5.18"]
dev = [
    "pytest>=7.4",
    "pytest-cov>=4.1",
    "pytest-timeout>=2.2",
    "ruff>=0.1",
]

[project.scripts]
# Populated in later tasks as main() functions land

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
traffic_forecast = ["py.typed"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "W", "UP", "B"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "characterization: locks in current behavior to detect regressions",
]
addopts = "-ra --strict-markers"
```

- [ ] **Step 5: Create package marker files**

`<root>/src/traffic_forecast/__init__.py`:

```python
"""Smart City Traffic Forecasting package."""

__version__ = "0.1.0"
```

`<root>/src/traffic_forecast/py.typed`: empty file.

`.gitkeep` files in `data/raw/`, `data/processed/`, `models/`, `reports/` (empty).

- [ ] **Step 6: Install package editable and verify import**

```bash
cd <root>
pip install -e .
python -c "import traffic_forecast; print(traffic_forecast.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 7: Commit**

```bash
cd <root>
git add .gitignore LICENSE pyproject.toml src/traffic_forecast/__init__.py src/traffic_forecast/py.typed data/raw/.gitkeep data/processed/.gitkeep models/.gitkeep reports/.gitkeep
# Stage existing project files too (README, src/*.py, dashboard/, data/, models/, reports/, docs/)
git add README.md requirements.txt src/*.py dashboard/ docs/
git commit -m "package: scaffold traffic_forecast and init repo"
```

---

### Task 2: Reproducibility fix + `data/generate.py`

**Goal:** Move `generate_data.py` into the package as `data/generate.py` with F2 fixed (reentrant RNG). The legacy `src/generate_data.py` file stays untouched until Task 9.

**Files:**
- Create: `<root>/src/traffic_forecast/data/__init__.py`
- Create: `<root>/src/traffic_forecast/data/generate.py`
- Create: `<root>/tests/__init__.py`
- Create: `<root>/tests/unit/__init__.py`
- Create: `<root>/tests/unit/test_generate.py`

**Interfaces:**
- Consumes: nothing
- Produces: `traffic_forecast.data.generate.build_dataset() -> pd.DataFrame`, `traffic_forecast.data.generate.main()`. The DataFrame has columns `["DateTime", "Junction", "Vehicles", "ID"]`, 70,080 rows for the default date range.

- [ ] **Step 1: Write the failing reproducibility test**

`<root>/tests/unit/test_generate.py`:

```python
import numpy as np
import pandas as pd

from traffic_forecast.data.generate import build_dataset, hourly_profile, weekday_factor


def test_hourly_profile_is_nonneg_with_two_peaks():
    h = np.arange(24)
    p = hourly_profile(h)
    assert p.shape == (24,)
    assert (p >= 0).all()
    assert 8 < np.argmax(p < p.max() * 0.5)  # morning dip exists
    assert np.argmax(p) > 15  # peak is in the evening


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
    assert 40 <= n_missing <= 100  # ~0.1% of 70080
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd <root>
pytest tests/unit/test_generate.py -v
```

Expected: 5 failures with `ModuleNotFoundError: No module named 'traffic_forecast.data.generate'`.

- [ ] **Step 3: Write `data/generate.py`**

`<root>/src/traffic_forecast/data/__init__.py` (empty).

`<root>/src/traffic_forecast/data/generate.py`:

```python
"""
Synthetic hourly traffic-volume generator for 4 city junctions.

Schema matches the original Kaggle-style dataset referenced in the project
reports: DateTime, Junction, Vehicles, ID. Behaviour mirrors the EDA findings
- daily + weekly seasonality, junction-specific volume levels, holiday dips,
occasional genuine spikes, a small fraction of missing values.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from traffic_forecast.config import HOLIDAYS, JUNCTIONS, START, END, MISSING_RATIO, SEED, SPIKE_RATE

RNG = None  # module-level placeholder; real RNG constructed inside build_dataset for reentrance


def hourly_profile(hour: np.ndarray) -> np.ndarray:
    """Two-peak (morning + evening rush) daily shape, values in [0, ~1.5)."""
    morning = np.exp(-((hour - 9) ** 2) / (2 * 2.0**2))
    evening = np.exp(-((hour - 18.5) ** 2) / (2 * 2.5**2))
    night_floor = 0.12
    return np.clip(night_floor + 0.55 * morning + 0.75 * evening, 0, None)


def weekday_factor(dow: np.ndarray) -> np.ndarray:
    """Weekdays busier than weekends (dow: 0=Mon ... 6=Sun)."""
    return np.where(dow < 5, 1.0, 0.65)


def build_dataset(
    start: str = START,
    end: str = END,
    junctions: dict | None = None,
    seed: int = SEED,
    missing_ratio: float = MISSING_RATIO,
    spike_rate: float = SPIKE_RATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    junctions = junctions or JUNCTIONS
    dt_index = pd.date_range(start, end, freq="h")
    frames = []
    for jid, cfg in junctions.items():
        hour = dt_index.hour.values
        dow = dt_index.dayofweek.values
        month = dt_index.month.values

        daily = hourly_profile(hour)
        weekly = weekday_factor(dow)
        yearly = 1.0 - 0.10 * np.isin(month, [6, 7, 8, 9]).astype(float)

        vol = cfg["base"] * (0.4 + daily) * weekly * yearly
        vol += rng.normal(0, cfg["noise"], size=len(dt_index))

        is_holiday = np.isin(dt_index.normalize(), HOLIDAYS)
        vol = np.where(is_holiday, vol * 0.55, vol)

        spike_mask = rng.random(len(dt_index)) < spike_rate
        vol = np.where(spike_mask, vol + rng.uniform(20, 45, len(dt_index)), vol)

        vol = np.clip(vol, 0, None)
        vol = np.round(vol).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "DateTime": dt_index,
                    "Junction": jid,
                    "Vehicles": vol,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["DateTime", "Junction"]).reset_index(drop=True)
    df["ID"] = df["DateTime"].dt.strftime("%Y%m%d%H").astype(str) + df["Junction"].astype(str)

    missing_idx = rng.choice(df.index, size=int(missing_ratio * len(df)), replace=False)
    df.loc[missing_idx, "Vehicles"] = np.nan

    return df[["DateTime", "Junction", "Vehicles", "ID"]]


def main() -> None:
    out_dir = Path(__file__).resolve().parents[3] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    out_path = out_dir / "traffic_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Write a temporary stub `config.py`**

Because Task 3 owns `config.py` fully, write a minimum stub here that Task 3 will expand. `<root>/src/traffic_forecast/config.py`:

```python
"""Central configuration. Expanded in Task 3."""

from __future__ import annotations

SEED = 42
START = "2022-01-01"
END = "2023-12-31 23:00:00"
MISSING_RATIO = 0.001
SPIKE_RATE = 0.003

JUNCTIONS = {
    1: {"base": 55, "amp": 35, "noise": 6},
    2: {"base": 30, "amp": 18, "noise": 4},
    3: {"base": 18, "amp": 10, "noise": 3},
    4: {"base": 10, "amp": 5, "noise": 2},
}

HOLIDAYS = pd.to_datetime(
    [
        "2022-01-26",
        "2022-03-18",
        "2022-08-15",
        "2022-10-02",
        "2022-10-24",
        "2022-11-08",
        "2022-12-25",
        "2023-01-26",
        "2023-03-08",
        "2023-08-15",
        "2023-10-02",
        "2023-11-12",
        "2023-12-25",
    ]
)
```

Note: needs `import pandas as pd` at top. The full Task 3 will reorganize but the values stay.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd <root>
pytest tests/unit/test_generate.py -v
```

Expected: 5 passing.

- [ ] **Step 6: Commit**

```bash
cd <root>
git add src/traffic_forecast/data/ src/traffic_forecast/config.py tests/__init__.py tests/unit/__init__.py tests/unit/test_generate.py
git commit -m "data: move generate into package with reentrant rng"
```

---

### Task 3: `config.py` — single source of truth

**Goal:** Consolidate every magic number, path, hyperparameter, and constant into one module. Apply F8 (HOLIDAYS dedupe), F9 (SEQ_LEN dedupe), F11 (path block dedupe).

**Files:**
- Modify: `<root>/src/traffic_forecast/config.py`
- Create: `<root>/tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `config.PROJECT_ROOT`, `config.RAW_PATH`, `config.PROCESSED_DIR`, `config.REPORTS_DIR`, `config.MODELS_DIR`, `config.HOLIDAYS`, `config.SEQ_LEN`, `config.VAL_FRAC`, `config.RF_PARAM_GRID`, `config.RF_GRID_SUBSAMPLE`, `config.LSTM_HYPERPARAMS` (dict), `config.SARIMA_ORDER`, `config.SARIMA_SEASONAL_ORDER`, `config.SARIMA_TRAIN_WINDOW`, `config.SARIMA_EVAL_HOURS`, `config.FEATURE_COLS`, `config.LSTM_FEATURE_COLS`.

- [ ] **Step 1: Write the failing test**

`<root>/tests/unit/test_config.py`:

```python
from pathlib import Path

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd <root>
pytest tests/unit/test_config.py -v
```

Expected: failures for missing attributes (`PROJECT_ROOT`, `RAW_PATH`, etc.).

- [ ] **Step 3: Replace `config.py` with the full version**

`<root>/src/traffic_forecast/config.py` (full content):

```python
"""Project-wide configuration: paths, seeds, hyperparameters, feature specs.

Single source of truth so the pipeline stages, models, and dashboard never
drift apart on magic numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_PATH = DATA_DIR / "raw" / "traffic_data.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_PATH = PROCESSED_DIR / "features.csv"
TRAIN_PKL = PROCESSED_DIR / "train_df.pkl"
VAL_PKL = PROCESSED_DIR / "val_df.pkl"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

SEED = 42
VAL_FRAC = 0.2

START = "2022-01-01"
END = "2023-12-31 23:00:00"
MISSING_RATIO = 0.001
SPIKE_RATE = 0.003

JUNCTIONS = {
    1: {"base": 55, "amp": 35, "noise": 6},
    2: {"base": 30, "amp": 18, "noise": 4},
    3: {"base": 18, "amp": 10, "noise": 3},
    4: {"base": 10, "amp": 5, "noise": 2},
}

HOLIDAYS = pd.to_datetime(
    [
        "2022-01-26",
        "2022-03-18",
        "2022-08-15",
        "2022-10-02",
        "2022-10-24",
        "2022-11-08",
        "2022-12-25",
        "2023-01-26",
        "2023-03-08",
        "2023-08-15",
        "2023-10-02",
        "2023-11-12",
        "2023-12-25",
    ]
)

FEATURE_COLS = [
    "hour",
    "dayofweek",
    "month",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_24",
    "roll_mean_3",
    "roll_mean_24",
    "is_outlier",
]
LSTM_FEATURE_COLS = [
    "hour",
    "dayofweek",
    "is_weekend",
    "is_holiday",
    "lag_1",
    "lag_24",
    "roll_mean_3",
    "roll_mean_24",
]

SEQ_LEN = 24

RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_leaf": [1, 2, 4],
}
RF_GRID_SUBSAMPLE = 12000
RF_RANDOM_STATE = SEED

SARIMA_ORDER = (1, 0, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 24)
SARIMA_MAXITER = 50
SARIMA_TRAIN_WINDOW = 24 * 30 * 2
SARIMA_EVAL_HOURS = 24 * 14

LSTM_HYPERPARAMS = {
    "hidden": 32,
    "layers": 2,
    "lr": 2e-3,
    "batch_size": 512,
    "epochs": 15,
    "seq_len": SEQ_LEN,
    "grad_clip": 1.0,
    "patience": 4,
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd <root>
pytest tests/unit/test_config.py -v
```

Expected: 6 passing.

- [ ] **Step 5: Re-run generate tests to confirm no regression**

```bash
cd <root>
pytest tests/unit/test_generate.py -v
```

Expected: 5 passing.

- [ ] **Step 6: Commit**

```bash
cd <root>
git add src/traffic_forecast/config.py tests/unit/test_config.py
git commit -m "config: single source for paths, holidays, hyperparams"
```

---

### Task 4: `eval/metrics.py` — shared metric helpers

**Goal:** Apply F10 (rmse dedupe). One module owns `rmse` and `mae`.

**Files:**
- Create: `<root>/src/traffic_forecast/eval/__init__.py`
- Create: `<root>/src/traffic_forecast/eval/metrics.py`
- Create: `<root>/tests/unit/test_metrics.py`

**Interfaces:**
- Consumes: nothing
- Produces: `eval.metrics.rmse(y_true, y_pred) -> float`, `eval.metrics.mae(y_true, y_pred) -> float`.

- [ ] **Step 1: Write the failing test**

`<root>/tests/unit/test_metrics.py`:

```python
import math

import numpy as np

from traffic_forecast.eval.metrics import mae, rmse


def test_rmse_matches_manual_sqrt_of_mse():
    y = np.array([3.0, 4.0])
    p = np.array([3.0, 5.0])
    # MSE = ((0)^2 + (1)^2) / 2 = 0.5
    assert rmse(y, p) == math.sqrt(0.5)


def test_mae_matches_manual_mean_abs_error():
    y = np.array([3.0, 4.0, 10.0])
    p = np.array([3.0, 5.0, 7.0])
    assert mae(y, p) == (0 + 1 + 3) / 3


def test_rmse_returns_python_float():
    out = rmse(np.array([1.0]), np.array([2.0]))
    assert isinstance(out, float)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd <root>
pytest tests/unit/test_metrics.py -v
```

Expected: `ModuleNotFoundError: No module named 'traffic_forecast.eval'`.

- [ ] **Step 3: Write `eval/metrics.py`**

`<root>/src/traffic_forecast/eval/__init__.py` (empty).

`<root>/src/traffic_forecast/eval/metrics.py`:

```python
"""Shared regression metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd <root>
pytest tests/unit/test_metrics.py -v
```

Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
cd <root>
git add src/traffic_forecast/eval/ tests/unit/test_metrics.py
git commit -m "eval: shared rmse/mae helpers"
```

---

### Task 5: `features.py` — move into package, fix F5 leakage

**Goal:** Move `features.py` into the package. Apply F5: compute IQR on train only and apply the threshold to val. Keep `engineer_features` API identical so the legacy `src/features.py` and the new module behave the same on the train half.

**Files:**
- Create: `<root>/src/traffic_forecast/features.py`
- Create: `<root>/tests/unit/test_features.py`

**Interfaces:**
- Consumes: `config.FEATURE_COLS`, `config.HOLIDAYS`, `config.VAL_FRAC`
- Produces:
  - `features.load_raw(path) -> pd.DataFrame`
  - `features.flag_outliers_iqr(df, col="Vehicles", q1=None, q3=None) -> pd.Series` (uses train thresholds if supplied)
  - `features.compute_iqr_thresholds(df, col="Vehicles") -> tuple[float, float]`
  - `features.engineer_features(df, iqr_thresholds=None) -> pd.DataFrame`
  - `features.chronological_split(df, val_frac=VAL_FRAC) -> tuple[pd.DataFrame, pd.DataFrame]`
  - `features.make_linear_view(train, val) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]`
  - `features.make_tree_view(train, val) -> tuple[pd.DataFrame, pd.DataFrame]`

- [ ] **Step 1: Write the failing test**

`<root>/tests/unit/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from traffic_forecast import features


@pytest.fixture
def tiny_df():
    """4 junctions x 50 hours, no missing values, no NaNs expected post-feature."""
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(50):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_engineer_features_drops_first_24h_per_junction(tiny_df):
    out = features.engineer_features(tiny_df)
    # 4 junctions x (50 - 24) = 104 rows
    assert len(out) == 104


def test_engineer_features_no_nans(tiny_df):
    out = features.engineer_features(tiny_df)
    assert out.isna().sum().sum() == 0


def test_lag_1_equals_groupby_shift(tiny_df):
    out = features.engineer_features(tiny_df)
    expected = out.groupby("Junction")["Vehicles"].shift(1)
    assert (out["lag_1"] == expected).all()


def test_lag_24_equals_groupby_shift(tiny_df):
    out = features.engineer_features(tiny_df)
    expected = out.groupby("Junction")["Vehicles"].shift(24)
    assert (out["lag_24"] == expected).all()


def test_is_weekend_iff_dow_ge_5(tiny_df):
    out = features.engineer_features(tiny_df)
    assert (out["is_weekend"] == (out["dayofweek"] >= 5).astype(int)).all()


def test_chronological_split_train_before_val(tiny_df):
    feat = features.engineer_features(tiny_df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    assert train["DateTime"].max() < val["DateTime"].min()


def test_compute_iqr_thresholds_per_junction_uses_global_quantiles():
    df = pd.DataFrame(
        {
            "Junction": [1, 1, 1, 1, 2, 2, 2, 2],
            "Vehicles": [10, 10, 10, 100, 1, 1, 1, 50],
        }
    )
    q1, q3 = features.compute_iqr_thresholds(df)
    assert q1 < q3


def test_iqr_thresholds_computed_on_train_only(tiny_df):
    train_df = tiny_df.iloc[:100]
    val_df = tiny_df.iloc[100:]
    q1, q3 = features.compute_iqr_thresholds(train_df)
    # Apply to val without refitting
    feat_train = features.engineer_features(train_df, iqr_thresholds=(q1, q3))
    feat_val = features.engineer_features(val_df, iqr_thresholds=(q1, q3))
    assert "is_outlier" in feat_train.columns
    assert "is_outlier" in feat_val.columns


def test_make_linear_view_same_columns_train_val(tiny_df):
    feat = features.engineer_features(tiny_df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    train_x, val_x, _ = features.make_linear_view(train, val)
    assert list(train_x.columns) == list(val_x.columns)


def test_make_linear_view_scaler_fit_on_train_only(tiny_df):
    feat = features.engineer_features(tiny_df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    _, _, scaler = features.make_linear_view(train, val)
    # Sanity: scaler mean is not all-zero (it learned something from train)
    assert scaler.mean_ is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd <root>
pytest tests/unit/test_features.py -v
```

Expected: import failure.

- [ ] **Step 3: Write `features.py`**

`<root>/src/traffic_forecast/features.py`:

```python
"""Feature engineering + chronological train/val split.

Two parallel preprocessed views of the same feature set:
  - one-hot + scaled (for Linear Regression)
  - integer-coded, unscaled (for tree models and the dashboard's predictor form)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from traffic_forecast.config import FEATURE_COLS, HOLIDAYS, VAL_FRAC


def load_raw(path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["DateTime"])


def compute_iqr_thresholds(df: pd.DataFrame, col: str = "Vehicles") -> tuple[float, float]:
    """Global (cross-junction) IQR thresholds. Use this on the TRAIN half only,
    then pass the returned tuple into engineer_features for both train and val
    so the val outlier flag is not informed by val quantiles (leakage fix F5)."""
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)


def flag_outliers_iqr(
    df: pd.DataFrame,
    col: str = "Vehicles",
    iqr_thresholds: tuple[float, float] | None = None,
) -> pd.Series:
    """Per-junction IQR outlier flag. If iqr_thresholds is None, compute per
    junction from the data (legacy behaviour). If supplied, apply the global
    thresholds uniformly (leakage-free behaviour)."""
    flags = pd.Series(False, index=df.index)
    if iqr_thresholds is None:
        for _, grp in df.groupby("Junction"):
            q1, q3 = grp[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            flags.loc[grp.index] = (grp[col] < lower) | (grp[col] > upper)
    else:
        lower, upper = iqr_thresholds
        flags = (df[col] < lower) | (df[col] > upper)
    return flags


def engineer_features(
    df: pd.DataFrame,
    iqr_thresholds: tuple[float, float] | None = None,
) -> pd.DataFrame:
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    df["Vehicles"] = df.groupby("Junction")["Vehicles"].ffill().bfill()
    df["is_outlier"] = flag_outliers_iqr(df, iqr_thresholds=iqr_thresholds)

    df["hour"] = df["DateTime"].dt.hour
    df["dayofweek"] = df["DateTime"].dt.dayofweek
    df["month"] = df["DateTime"].dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = df["DateTime"].dt.normalize().isin(HOLIDAYS).astype(int)

    grp = df.groupby("Junction")["Vehicles"]
    df["lag_1"] = grp.shift(1)
    df["lag_24"] = grp.shift(24)
    df["roll_mean_3"] = grp.transform(lambda s: s.shift(1).rolling(3).mean())
    df["roll_mean_24"] = grp.transform(lambda s: s.shift(1).rolling(24).mean())

    df = df.dropna(subset=["lag_1", "lag_24", "roll_mean_3", "roll_mean_24"]).reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df["DateTime"].quantile(1 - val_frac)
    train = df[df["DateTime"] < cutoff].copy()
    val = df[df["DateTime"] >= cutoff].copy()
    return train, val


def make_linear_view(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    cols = FEATURE_COLS + ["Junction"]
    train_x = pd.get_dummies(train[cols], columns=["Junction"], prefix="junc")
    val_x = pd.get_dummies(val[cols], columns=["Junction"], prefix="junc")
    val_x = val_x.reindex(columns=train_x.columns, fill_value=0)

    scaler = StandardScaler()
    train_x_scaled = pd.DataFrame(
        scaler.fit_transform(train_x), columns=train_x.columns, index=train.index
    )
    val_x_scaled = pd.DataFrame(scaler.transform(val_x), columns=val_x.columns, index=val.index)
    return train_x_scaled, val_x_scaled, scaler


def make_tree_view(
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = FEATURE_COLS + ["Junction"]
    return train[cols].copy(), val[cols].copy()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd <root>
pytest tests/unit/test_features.py -v
```

Expected: 10 passing.

- [ ] **Step 5: Commit**

```bash
cd <root>
git add src/traffic_forecast/features.py tests/unit/test_features.py
git commit -m "features: move into package, fit iqr on train only"
```

---

### Task 6: `models/lr.py` and `models/rf.py`

**Goal:** Extract Linear Regression and Random Forest training into testable functions.

**Files:**
- Create: `<root>/src/traffic_forecast/models/__init__.py`
- Create: `<root>/src/traffic_forecast/models/lr.py`
- Create: `<root>/src/traffic_forecast/models/rf.py`
- Create: `<root>/tests/unit/test_lr.py`
- Create: `<root>/tests/unit/test_rf.py`

**Interfaces:**
- Consumes: `features.make_linear_view`, `features.make_tree_view`, `eval.metrics`, `config`
- Produces:
  - `lr.train(train_df, val_df) -> dict` returning `{"model", "scaler", "pred_val", "metrics": {"MAE","RMSE"}}`
  - `rf.train_default(train_df, val_df) -> dict`
  - `rf.train_tuned(train_df, val_df) -> dict` returning additionally `{"best_params"}`

- [ ] **Step 1: Write the failing LR test**

`<root>/tests/unit/test_lr.py`:

```python
import numpy as np
import pandas as pd

from traffic_forecast.models import lr


def _make_fixture():
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(200):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_lr_train_returns_expected_artifacts():
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = lr.train(train, val)
    assert set(out.keys()) >= {"model", "scaler", "pred_val", "metrics"}
    assert {"MAE", "RMSE"} <= set(out["metrics"].keys())
    assert len(out["pred_val"]) == len(val)
    assert out["metrics"]["MAE"] > 0
```

- [ ] **Step 2: Write the failing RF test**

`<root>/tests/unit/test_rf.py`:

```python
import numpy as np
import pandas as pd

from traffic_forecast.models import rf


def _make_fixture():
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(300):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_rf_default_returns_metrics_and_predictions():
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = rf.train_default(train, val)
    assert {"model", "pred_val", "metrics"} <= set(out.keys())
    assert len(out["pred_val"]) == len(val)


def test_rf_tuned_returns_best_params_in_grid():
    from traffic_forecast import features
    from traffic_forecast import config

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = rf.train_tuned(train, val, grid_subsample=200)
    assert set(out["best_params"].keys()) == set(config.RF_PARAM_GRID.keys())
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd <root>
pytest tests/unit/test_lr.py tests/unit/test_rf.py -v
```

Expected: `ModuleNotFoundError: No module named 'traffic_forecast.models'`.

- [ ] **Step 4: Write `models/lr.py` and `models/rf.py`**

`<root>/src/traffic_forecast/models/__init__.py` (empty).

`<root>/src/traffic_forecast/models/lr.py`:

```python
"""Linear Regression model."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LinearRegression

from traffic_forecast.eval.metrics import mae, rmse
from traffic_forecast.features import make_linear_view


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    train_x, val_x, scaler = make_linear_view(train_df, val_df)
    y_train = train_df["Vehicles"].values
    y_val = val_df["Vehicles"].values

    model = LinearRegression().fit(train_x, y_train)
    pred_val = model.predict(val_x)

    return {
        "model": model,
        "scaler": scaler,
        "pred_val": pred_val,
        "metrics": {"MAE": mae(y_val, pred_val), "RMSE": rmse(y_val, pred_val)},
    }
```

`<root>/src/traffic_forecast/models/rf.py`:

```python
"""Random Forest: default baseline + grid-search tuned variant."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from traffic_forecast.config import RF_GRID_SUBSAMPLE, RF_PARAM_GRID, RF_RANDOM_STATE, SEED
from traffic_forecast.eval.metrics import mae, rmse
from traffic_forecast.features import make_tree_view


def train_default(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    train_x, val_x = make_tree_view(train_df, val_df)
    y_train = train_df["Vehicles"].values
    y_val = val_df["Vehicles"].values

    model = RandomForestRegressor(random_state=RF_RANDOM_STATE, n_jobs=-1, n_estimators=100)
    model.fit(train_x, y_train)
    pred_val = model.predict(val_x)

    return {
        "model": model,
        "pred_val": pred_val,
        "metrics": {"MAE": mae(y_val, pred_val), "RMSE": rmse(y_val, pred_val)},
    }


def train_tuned(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    grid_subsample: int = RF_GRID_SUBSAMPLE,
) -> dict:
    train_x, val_x = make_tree_view(train_df, val_df)
    y_train = train_df["Vehicles"].values
    y_val = val_df["Vehicles"].values

    sample_idx = np.random.default_rng(0).choice(
        len(train_x),
        size=min(grid_subsample, len(train_x)),
        replace=False,
    )
    grid = GridSearchCV(
        RandomForestRegressor(random_state=RF_RANDOM_STATE, n_jobs=-1),
        RF_PARAM_GRID,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(train_x.iloc[sample_idx], y_train[sample_idx])
    best_params = grid.best_params_

    model = RandomForestRegressor(random_state=RF_RANDOM_STATE, n_jobs=-1, **best_params)
    model.fit(train_x, y_train)
    pred_val = model.predict(val_x)

    return {
        "model": model,
        "pred_val": pred_val,
        "best_params": best_params,
        "metrics": {"MAE": mae(y_val, pred_val), "RMSE": rmse(y_val, pred_val)},
    }
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd <root>
pytest tests/unit/test_lr.py tests/unit/test_rf.py -v
```

Expected: 3 passing.

- [ ] **Step 6: Commit**

```bash
cd <root>
git add src/traffic_forecast/models/__init__.py src/traffic_forecast/models/lr.py src/traffic_forecast/models/rf.py tests/unit/test_lr.py tests/unit/test_rf.py
git commit -m "models: extract linear regression and random forest"
```

---

### Task 7: `models/sarima.py`

**Goal:** Extract SARIMA training into a function that fits per-junction on a rolling 2-month window and forecasts the first 2 weeks of val. Matches current `stage2_sarima.py` behaviour exactly.

**Files:**
- Create: `<root>/src/traffic_forecast/models/sarima.py`
- Create: `<root>/tests/unit/test_sarima.py`

**Interfaces:**
- Consumes: `config.SARIMA_*`, `eval.metrics`
- Produces: `sarima.train(train_df, val_df) -> dict` returning `{"predictions": {jid: {"datetime","actual","pred"}}, "pred_val": np.ndarray (aligned with val_df rows for the eval window only), "metrics": {"MAE","RMSE"}, "eval_mask": np.ndarray (bool, len(val_df))}`

- [ ] **Step 1: Write the failing test**

`<root>/tests/unit/test_sarima.py`:

```python
import numpy as np
import pandas as pd

from traffic_forecast.models import sarima


def _make_fixture():
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(24 * 90):  # 90 days per junction so SARIMA window has data
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_sarima_returns_predictions_for_each_junction():
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = sarima.train(train, val)
    assert set(out["predictions"].keys()) == {1, 2, 3, 4}
    for jid, payload in out["predictions"].items():
        assert len(payload["datetime"]) == len(payload["actual"]) == len(payload["pred"])
    assert {"MAE", "RMSE"} <= set(out["metrics"].keys())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd <root>
pytest tests/unit/test_sarima.py -v
```

Expected: `ModuleNotFoundError: No module named 'traffic_forecast.models.sarima'`.

- [ ] **Step 3: Write `models/sarima.py`**

`<root>/src/traffic_forecast/models/sarima.py`:

```python
"""SARIMA per-junction on a rolling 2-month train window, forecasting the
first 2 weeks of the validation window. Mirrors how SARIMA is used in practice
(periodic refit on recent data) rather than a one-shot fit over years."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from traffic_forecast.config import (
    SARIMA_EVAL_HOURS,
    SARIMA_MAXITER,
    SARIMA_ORDER,
    SARIMA_SEASONAL_ORDER,
    SARIMA_TRAIN_WINDOW,
)
from traffic_forecast.eval.metrics import mae, rmse


def train(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    predictions = {}
    all_actual, all_pred = [], []

    for jid in sorted(train_df["Junction"].unique()):
        tr = (
            train_df.loc[train_df["Junction"] == jid].set_index("DateTime")["Vehicles"].sort_index()
        )
        va = val_df.loc[val_df["Junction"] == jid].set_index("DateTime")["Vehicles"].sort_index()
        tr_recent = tr.iloc[-SARIMA_TRAIN_WINDOW:]
        va_eval = va.iloc[:SARIMA_EVAL_HOURS]

        model = SARIMAX(
            tr_recent,
            order=SARIMA_ORDER,
            seasonal_order=SARIMA_SEASONAL_ORDER,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=SARIMA_MAXITER)
        fcast = fit.forecast(steps=len(va_eval))

        predictions[int(jid)] = {
            "datetime": va_eval.index.astype(str).tolist(),
            "actual": va_eval.values.tolist(),
            "pred": fcast.values.tolist(),
        }
        all_actual.extend(va_eval.values.tolist())
        all_pred.extend(fcast.values.tolist())

    actual = np.array(all_actual)
    pred = np.array(all_pred)
    return {
        "predictions": predictions,
        "metrics": {"MAE": mae(actual, pred), "RMSE": rmse(actual, pred)},
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd <root>
pytest tests/unit/test_sarima.py -v
```

Expected: 1 passing (slow ~10s).

- [ ] **Step 5: Commit**

```bash
cd <root>
git add src/traffic_forecast/models/sarima.py tests/unit/test_sarima.py
git commit -m "models: extract sarima per-junction trainer"
```

---

### Task 8: `models/lstm.py` — fix the constant-output collapse (F1, F4)

**Goal:** Move the LSTM architecture into the package and replace the broken training with one that scales the target, clips gradients, monitors val loss with early stopping, and includes junction identity.

**Files:**
- Create: `<root>/src/traffic_forecast/models/lstm.py`
- Create: `<root>/tests/unit/test_lstm.py`

**Interfaces:**
- Consumes: `config.LSTM_HYPERPARAMS`, `config.LSTM_FEATURE_COLS`, `config.SEQ_LEN`, `config.SEED`
- Produces:
  - `lstm.TrafficLSTM(nn.Module)` with `forward(x) -> Tensor`
  - `lstm.make_sequences(df, feature_cols, target_col, seq_len) -> tuple[np.ndarray, np.ndarray]`
  - `lstm.train(train_df, val_df) -> dict` returning `{"model", "pred_val", "actual_val", "metrics"}`

- [ ] **Step 1: Write the failing test (this is the regression-catch for F1)**

`<root>/tests/unit/test_lstm.py`:

```python
import numpy as np
import pandas as pd

from traffic_forecast.models import lstm


def _make_fixture():
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(24 * 80):  # 80 days/junction so val sequences exist
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    return pd.DataFrame(rows)


def test_lstm_predictions_are_not_constant():
    """The single most important test in this sub-project. Pre-fix the LSTM
    emitted the same scalar (~global mean) for every input. Post-fix the
    prediction distribution must have non-trivial spread."""
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = lstm.train(train, val, epochs=3)
    assert out["pred_val"].std() > 1.0, (
        f"LSTM predictions collapsed to a constant (std={out['pred_val'].std():.4f}); "
        "the fix is not working."
    )


def test_lstm_metrics_are_finite_and_positive():
    from traffic_forecast import features

    df = _make_fixture()
    feat = features.engineer_features(df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    out = lstm.train(train, val, epochs=3)
    assert np.isfinite(out["metrics"]["MAE"])
    assert out["metrics"]["MAE"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd <root>
pytest tests/unit/test_lstm.py -v
```

Expected: `ModuleNotFoundError`.

- [ ] **Step 3: Write `models/lstm.py` with all four fixes**

`<root>/src/traffic_forecast/models/lstm.py`:

```python
"""LSTM model with target scaling, gradient clipping, early stopping, and
junction identity. The previous implementation collapsed to a constant output
because the raw 0..135 target combined with MSE loss and no clipping pushed
the recurrent gradients into a saturation regime; the network settled at the
global mean and never escaped. These four fixes together recover useful
learning."""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from traffic_forecast.config import LSTM_FEATURE_COLS, LSTM_HYPERPARAMS, SEED, SEQ_LEN
from traffic_forecast.eval.metrics import mae, rmse


class TrafficLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, num_layers=layers, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, grp in df.groupby("Junction"):
        grp = grp.sort_values("DateTime")
        vals = grp[feature_cols + [target_col]].values.astype(np.float32)
        for i in range(len(vals) - seq_len):
            xs.append(vals[i : i + seq_len, :-1])
            ys.append(vals[i + seq_len, -1])
    return np.array(xs), np.array(ys)


def train(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int | None = None,
) -> dict:
    hp = LSTM_HYPERPARAMS
    epochs = epochs if epochs is not None else hp["epochs"]
    seq_len = hp["seq_len"]
    feature_cols = LSTM_FEATURE_COLS + ["Junction"]

    torch.manual_seed(SEED)
    torch.set_num_threads(1)

    feature_scaler = MinMaxScaler().fit(train_df[feature_cols])
    target_scaler = MinMaxScaler().fit(train_df[["Vehicles"]])

    tr = train_df.copy()
    va = val_df.copy()
    tr[feature_cols] = feature_scaler.transform(train_df[feature_cols])
    va[feature_cols] = feature_scaler.transform(val_df[feature_cols])
    tr["Vehicles"] = target_scaler.transform(train_df[["Vehicles"]])
    va["Vehicles"] = target_scaler.transform(val_df[["Vehicles"]])

    X_train, y_train = make_sequences(tr, feature_cols, "Vehicles", seq_len)
    X_val, y_val = make_sequences(va, feature_cols, "Vehicles", seq_len)

    model = TrafficLSTM(n_features=len(feature_cols), hidden=hp["hidden"], layers=hp["layers"])
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=4, gamma=0.5)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    n = len(X_train_t)
    batch_size = hp["batch_size"]
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improve = 0

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=hp["grad_clip"])
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= hp["patience"]:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(X_val_t).numpy()

    pred_val = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual_val = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    return {
        "model": model,
        "pred_val": pred_val,
        "actual_val": actual_val,
        "metrics": {"MAE": mae(actual_val, pred_val), "RMSE": rmse(actual_val, pred_val)},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd <root>
pytest tests/unit/test_lstm.py -v
```

Expected: 2 passing. `test_lstm_predictions_are_not_constant` is the key one. If it still fails, the LSTM is still collapsing; double-check grad clip and target scaling are both wired in.

- [ ] **Step 5: Commit**

```bash
cd <root>
git add src/traffic_forecast/models/lstm.py tests/unit/test_lstm.py
git commit -m "lstm: scale target, clip grads, early-stop, add junction feature"
```

---

### Task 9: Pipeline orchestrators (`pipeline/`) + delete legacy scripts

**Goal:** Replace `stage1_lr_rf.py` ... `stage4_combine.py` and `train_models.py` with thin orchestrators that import from `models/` and write the same `reports/` + `models/` artifacts as today. Apply F7 (def main + guards). Delete the legacy `src/*.py` files. Add console scripts and `__main__.py`.

**Files:**
- Create: `<root>/src/traffic_forecast/pipeline/__init__.py`
- Create: `<root>/src/traffic_forecast/pipeline/_common.py` (data loading + IQR thresholds)
- Create: `<root>/src/traffic_forecast/pipeline/stage1.py`
- Create: `<root>/src/traffic_forecast/pipeline/stage2.py`
- Create: `<root>/src/traffic_forecast/pipeline/stage3.py`
- Create: `<root>/src/traffic_forecast/pipeline/stage4.py`
- Create: `<root>/src/traffic_forecast/pipeline/run_all.py`
- Create: `<root>/src/traffic_forecast/__main__.py`
- Modify: `<root>/pyproject.toml` (add `[project.scripts]` entries)
- Delete: `<root>/src/generate_data.py`
- Delete: `<root>/src/features.py`
- Delete: `<root>/src/stage1_lr_rf.py`
- Delete: `<root>/src/stage2_sarima.py`
- Delete: `<root>/src/stage3_lstm.py`
- Delete: `<root>/src/stage4_combine.py`
- Delete: `<root>/src/train_models.py`
- Delete: `<root>/notebooks/` (empty dir)
- Create: `<root>/tests/integration/__init__.py`
- Create: `<root>/tests/integration/test_run_all.py`

**Interfaces:**
- Consumes: all of `models/`, `features`, `config`, `eval.metrics`, `data.generate`
- Produces:
  - `pipeline.stage1.main()`, `stage2.main()`, `stage3.main()`, `stage4.main()`, `run_all.main()` — each writes the same artifacts as the legacy script it replaces
  - `data.generate.main()` (already exists from Task 2)
  - Console scripts `tf-generate-data`, `tf-train-stage1`, `tf-train-stage2`, `tf-train-stage3`, `tf-combine`, `tf-train-all`
  - `python -m traffic_forecast <subcommand>` via `__main__.py`

- [ ] **Step 1: Write the failing integration test**

`<root>/tests/integration/__init__.py` (empty).

`<root>/tests/integration/test_run_all.py`:

```python
import json
from pathlib import Path

import pytest

from traffic_forecast import config
from traffic_forecast.pipeline import run_all


@pytest.mark.timeout(300)
def test_run_all_writes_expected_artifacts(tmp_path, monkeypatch):
    """End-to-end smoke on a tiny fixture. Patches the data path so we don't
    depend on the committed 70k-row CSV."""
    import numpy as np
    import pandas as pd
    from traffic_forecast.data import generate as gen_mod
    from traffic_forecast import features as feat_mod

    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(24 * 75):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    df = pd.DataFrame(rows)

    raw_path = tmp_path / "raw.csv"
    df.to_csv(raw_path, index=False)

    monkeypatch.setattr(config, "RAW_PATH", raw_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "TRAIN_PKL", tmp_path / "processed" / "train_df.pkl")
    monkeypatch.setattr(config, "VAL_PKL", tmp_path / "processed" / "val_df.pkl")
    monkeypatch.setattr(config, "FEATURES_PATH", tmp_path / "processed" / "features.csv")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")
    (tmp_path / "processed").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "models").mkdir()

    run_all.main(epochs=2)

    assert (tmp_path / "reports" / "model_comparison.csv").exists()
    assert (tmp_path / "reports" / "predictions_full.csv").exists()
    assert (tmp_path / "reports" / "pred_rf_tuned.npy").exists()
    assert (tmp_path / "models" / "random_forest_tuned.joblib").exists()

    with open(tmp_path / "reports" / "results_stage1.json") as f:
        r1 = json.load(f)
    assert "Linear Regression" in r1
    assert "Random Forest (tuned)" in r1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd <root>
pytest tests/integration/test_run_all.py -v
```

Expected: `ModuleNotFoundError: No module named 'traffic_forecast.pipeline'`.

- [ ] **Step 3: Write the pipeline modules**

`<root>/src/traffic_forecast/pipeline/__init__.py` (empty).

`<root>/src/traffic_forecast/pipeline/_common.py`:

```python
"""Shared helpers for the pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from traffic_forecast import config
from traffic_forecast.features import (
    chronological_split,
    compute_iqr_thresholds,
    engineer_features,
    load_raw,
)


def ensure_dirs() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split():
    """Load raw, engineer features (with train-only IQR thresholds), split
    chronographically, persist the pickled train/val frames for downstream
    stages, and return (train_df, val_df)."""
    ensure_dirs()
    raw = load_raw(config.RAW_PATH)
    feat_full = engineer_features(raw)
    train_df, val_df = chronological_split(feat_full, val_frac=config.VAL_FRAC)
    q1, q3 = compute_iqr_thresholds(train_df)
    train_df = engineer_features(train_df, iqr_thresholds=(q1, q3)).pipe(
        lambda d: d.sort_values("DateTime").reset_index(drop=True)
    )
    val_df = engineer_features(val_df, iqr_thresholds=(q1, q3)).pipe(
        lambda d: d.sort_values("DateTime").reset_index(drop=True)
    )
    train_df.to_pickle(config.TRAIN_PKL)
    val_df.to_pickle(config.VAL_PKL)
    return train_df, val_df


def load_pickled_split():
    return pd.read_pickle(config.TRAIN_PKL), pd.read_pickle(config.VAL_PKL)


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
```

`<root>/src/traffic_forecast/pipeline/stage1.py`:

```python
"""Stage 1: Linear Regression + Random Forest (default + tuned)."""

from __future__ import annotations

import numpy as np

from traffic_forecast import config
from traffic_forecast.models import lr as lr_model
from traffic_forecast.models import rf as rf_model
from traffic_forecast.pipeline._common import ensure_dirs, load_and_split, write_json
from traffic_forecast.pipeline._common import load_pickled_split
import joblib


def main() -> dict:
    ensure_dirs()
    try:
        train_df, val_df = load_pickled_split()
    except FileNotFoundError:
        train_df, val_df = load_and_split()

    results = {}

    lr_out = lr_model.train(train_df, val_df)
    joblib.dump(lr_out["model"], config.MODELS_DIR / "linear_regression.joblib")
    joblib.dump(lr_out["scaler"], config.MODELS_DIR / "lr_scaler.joblib")
    np.save(config.REPORTS_DIR / "pred_linear.npy", lr_out["pred_val"])
    results["Linear Regression"] = lr_out["metrics"]

    rf_default_out = rf_model.train_default(train_df, val_df)
    results["Random Forest (default)"] = rf_default_out["metrics"]

    rf_tuned_out = rf_model.train_tuned(train_df, val_df)
    joblib.dump(rf_tuned_out["model"], config.MODELS_DIR / "random_forest_tuned.joblib")
    np.save(config.REPORTS_DIR / "pred_rf_tuned.npy", rf_tuned_out["pred_val"])
    results["Random Forest (tuned)"] = rf_tuned_out["metrics"]

    write_json(config.REPORTS_DIR / "best_rf_params.json", rf_tuned_out["best_params"])
    write_json(config.REPORTS_DIR / "results_stage1.json", results)
    print("Stage 1 done.")
    return results


if __name__ == "__main__":
    main()
```

`<root>/src/traffic_forecast/pipeline/stage2.py`:

```python
"""Stage 2: SARIMA per junction on rolling 2-month window."""

from __future__ import annotations

from traffic_forecast import config
from traffic_forecast.models import sarima as sarima_model
from traffic_forecast.pipeline._common import ensure_dirs, load_pickled_split, write_json


def main() -> dict:
    ensure_dirs()
    train_df, val_df = load_pickled_split()
    out = sarima_model.train(train_df, val_df)
    write_json(config.REPORTS_DIR / "sarima_predictions.json", out["predictions"])
    result = {
        "SARIMA": {
            **out["metrics"],
            "note": (
                f"fit on last {config.SARIMA_TRAIN_WINDOW}h/junction, "
                f"evaluated on first {config.SARIMA_EVAL_HOURS}h of validation window"
            ),
        }
    }
    write_json(config.REPORTS_DIR / "results_stage2_sarima.json", result)
    print("Stage 2 done.")
    return result


if __name__ == "__main__":
    main()
```

`<root>/src/traffic_forecast/pipeline/stage3.py`:

```python
"""Stage 3: LSTM (fixed - target scaling, grad clipping, early stop, junction)."""

from __future__ import annotations

import numpy as np
import torch

from traffic_forecast import config
from traffic_forecast.models import lstm as lstm_model
from traffic_forecast.pipeline._common import ensure_dirs, load_pickled_split, write_json


def main(epochs: int | None = None) -> dict:
    ensure_dirs()
    train_df, val_df = load_pickled_split()
    out = lstm_model.train(train_df, val_df, epochs=epochs)
    torch.save(out["model"].state_dict(), config.MODELS_DIR / "lstm.pt")
    np.save(config.REPORTS_DIR / "pred_lstm.npy", out["pred_val"])
    np.save(config.REPORTS_DIR / "actual_lstm.npy", out["actual_val"])
    result = {"LSTM": out["metrics"]}
    write_json(config.REPORTS_DIR / "results_stage3_lstm.json", result)
    print("Stage 3 done.")
    return result


if __name__ == "__main__":
    main()
```

`<root>/src/traffic_forecast/pipeline/stage4.py`:

```python
"""Stage 4: combine reports, build dashboard data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from traffic_forecast import config
from traffic_forecast.pipeline._common import ensure_dirs, load_pickled_split


def main() -> None:
    ensure_dirs()
    REPORTS = config.REPORTS_DIR

    results = {}
    for fname in ("results_stage1.json", "results_stage2_sarima.json", "results_stage3_lstm.json"):
        with open(REPORTS / fname) as f:
            results.update(json.load(f))

    comp_df = pd.DataFrame(results).T
    comp_df.index.name = "Model"
    comp_df = comp_df.sort_values("MAE")
    comp_df.to_csv(REPORTS / "model_comparison.csv")
    print(comp_df)

    _, val_df = load_pickled_split()
    val_df = val_df.reset_index(drop=True)

    rf_pred = np.load(REPORTS / "pred_rf_tuned.npy")
    lr_pred = np.load(REPORTS / "pred_linear.npy")
    val_df["pred_rf_tuned"] = rf_pred
    val_df["pred_linear"] = lr_pred
    val_df["abs_error_rf"] = (val_df["Vehicles"] - val_df["pred_rf_tuned"]).abs()

    err = val_df.groupby(["Junction", "hour"])["abs_error_rf"].mean().reset_index()
    err.to_csv(REPORTS / "error_by_junction_hour.csv", index=False)

    with open(REPORTS / "sarima_predictions.json") as f:
        sarima = json.load(f)
    sarima_rows = []
    for jid, d in sarima.items():
        for dt, act, pred in zip(d["datetime"], d["actual"], d["pred"]):
            sarima_rows.append({"DateTime": dt, "Junction": int(jid), "pred_sarima": pred})
    sarima_df = pd.DataFrame(sarima_rows)
    sarima_df["DateTime"] = pd.to_datetime(sarima_df["DateTime"])
    val_df["DateTime"] = pd.to_datetime(val_df["DateTime"])
    merged = val_df.merge(sarima_df, on=["DateTime", "Junction"], how="left")

    lstm_pred = np.load(REPORTS / "pred_lstm.npy")
    seq_len = config.SEQ_LEN
    lstm_rows = []
    for jid, grp in val_df.sort_values("DateTime").groupby("Junction"):
        grp = grp.sort_values("DateTime").reset_index(drop=True)
        n_seq = len(grp) - seq_len
        dts = grp["DateTime"].iloc[seq_len : seq_len + n_seq].values
        lstm_rows.append(pd.DataFrame({"DateTime": dts, "Junction": jid}))
    lstm_meta = pd.concat(lstm_rows, ignore_index=True)
    lstm_meta["pred_lstm"] = lstm_pred[: len(lstm_meta)]
    lstm_meta["DateTime"] = pd.to_datetime(lstm_meta["DateTime"])
    merged = merged.merge(lstm_meta, on=["DateTime", "Junction"], how="left")

    cols = [
        "DateTime",
        "Junction",
        "Vehicles",
        "pred_linear",
        "pred_rf_tuned",
        "pred_sarima",
        "pred_lstm",
        "is_holiday",
        "is_outlier",
    ]
    merged[cols].to_csv(REPORTS / "predictions_full.csv", index=False)
    sample = merged[merged["DateTime"] < merged["DateTime"].min() + pd.Timedelta(days=14)]
    sample[cols].to_csv(REPORTS / "predictions_sample.csv", index=False)
    print("Stage 4 done.")


if __name__ == "__main__":
    main()
```

`<root>/src/traffic_forecast/pipeline/run_all.py`:

```python
"""Run all stages end-to-end. Replaces the legacy train_models.py."""

from __future__ import annotations

from traffic_forecast.pipeline import stage1, stage2, stage3, stage4
from traffic_forecast.pipeline._common import ensure_dirs, load_and_split


def main(epochs: int | None = None) -> None:
    ensure_dirs()
    load_and_split()
    stage1.main()
    stage2.main()
    stage3.main(epochs=epochs)
    stage4.main()
    print("All stages complete.")


if __name__ == "__main__":
    main()
```

`<root>/src/traffic_forecast/__main__.py`:

```python
"""Entry point for `python -m traffic_forecast <subcommand>`."""

from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: python -m traffic_forecast {generate-data|stage1|stage2|stage3|combine|all}")
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd == "generate-data":
        from traffic_forecast.data.generate import main as run
    elif cmd == "stage1":
        from traffic_forecast.pipeline.stage1 import main as run
    elif cmd == "stage2":
        from traffic_forecast.pipeline.stage2 import main as run
    elif cmd == "stage3":
        from traffic_forecast.pipeline.stage3 import main as run
    elif cmd == "combine":
        from traffic_forecast.pipeline.stage4 import main as run
    elif cmd == "all":
        from traffic_forecast.pipeline.run_all import main as run
    else:
        print(f"unknown subcommand: {cmd}")
        sys.exit(2)
    run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update `pyproject.toml` with console scripts**

Edit the `[project.scripts]` section to read:

```toml
[project.scripts]
tf-generate-data = "traffic_forecast.data.generate:main"
tf-train-stage1 = "traffic_forecast.pipeline.stage1:main"
tf-train-stage2 = "traffic_forecast.pipeline.stage2:main"
tf-train-stage3 = "traffic_forecast.pipeline.stage3:main"
tf-combine = "traffic_forecast.pipeline.stage4:main"
tf-train-all = "traffic_forecast.pipeline.run_all:main"
```

- [ ] **Step 5: Delete legacy scripts**

```bash
cd <root>
git rm src/generate_data.py src/features.py src/stage1_lr_rf.py src/stage2_sarima.py src/stage3_lstm.py src/stage4_combine.py src/train_models.py
rmdir notebooks 2>/dev/null || true
git rm -r notebooks 2>/dev/null || true
```

- [ ] **Step 6: Reinstall and run tests**

```bash
cd <root>
pip install -e . --quiet
pytest tests/integration/test_run_all.py -v
```

Expected: 1 passing (~60-90s).

- [ ] **Step 7: Verify console scripts work**

```bash
cd <root>
tf-train-stage1 2>&1 | tail -5
python -m traffic_forecast --help 2>&1 | head -3
```

Expected: stage1 runs against the committed `data/raw/traffic_data.csv`; help text prints usage.

- [ ] **Step 8: Commit**

```bash
cd <root>
git add -A
git commit -m "pipeline: stage orchestrators replace legacy scripts"
```

---

### Task 10: Move dashboard into the package

**Goal:** Move `dashboard/app.py` to `src/traffic_forecast/dashboard/app.py`, route path resolution through `config`, and add a `tf-dashboard` console script. The dashboard itself is unchanged in this sub-project (sub-project #3 owns UX). It just needs to keep working from its new location.

**Files:**
- Move: `<root>/dashboard/app.py` -> `<root>/src/traffic_forecast/dashboard/app.py`
- Create: `<root>/src/traffic_forecast/dashboard/__init__.py`
- Modify: `<root>/pyproject.toml` (add `tf-dashboard` script)

**Interfaces:**
- Consumes: `config.REPORTS_DIR`, `config.MODELS_DIR`
- Produces: `dashboard.app.main()` that launches streamlit via subprocess

- [ ] **Step 1: Create the dashboard subpackage**

```bash
cd <root>
mkdir -p src/traffic_forecast/dashboard
git mv dashboard/app.py src/traffic_forecast/dashboard/app.py
rmdir dashboard 2>/dev/null || true
```

`<root>/src/traffic_forecast/dashboard/__init__.py` (empty).

- [ ] **Step 2: Patch path resolution in `app.py`**

Open `<root>/src/traffic_forecast/dashboard/app.py`. Replace the path-resolution block near the top:

Old:
```python
ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
```

New:
```python
from traffic_forecast import config

REPORTS = config.REPORTS_DIR
MODELS = config.MODELS_DIR
```

Leave the rest of `app.py` unchanged.

- [ ] **Step 3: Add a `main()` launcher**

Append to the bottom of `<root>/src/traffic_forecast/dashboard/app.py`:

```python
def main() -> None:
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__)
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update `pyproject.toml`**

Add to `[project.scripts]`:

```toml
tf-dashboard = "traffic_forecast.dashboard.app:main"
```

- [ ] **Step 5: Smoke-import the dashboard module**

```bash
cd <root>
python -c "from traffic_forecast.dashboard import app; print('ok')"
```

Expected: `ok` (no execution of Streamlit code at import; only function definitions).

Note: streamlit's `st.set_page_config` runs at import time. If importing fails because streamlit is not installed, install via `pip install -e ".[dashboard]"` first. If it fails because streamlit requires a running context, wrap `st.set_page_config` etc. in a `def _render():` and call from `main()` is out of scope for Foundation (the existing app calls these at module scope). Acceptable for now: the import smoke is "best effort" — if it can't run without streamlit context, skip this step and rely on the dashboard test in sub-project #3.

- [ ] **Step 6: Commit**

```bash
cd <root>
git add -A
git commit -m "dashboard: move into package, resolve paths via config"
```

---

### Task 11: Characterization tests + smoke test

**Goal:** Lock the post-F1 ground truth so future sub-projects cannot regress unnoticed. Captures: data contract, model metrics within tolerance, prediction array shapes, LSTM non-constant invariant.

**Files:**
- Create: `<root>/tests/conftest.py`
- Create: `<root>/tests/characterization/__init__.py`
- Create: `<root>/tests/characterization/test_data_contract.py`
- Create: `<root>/tests/characterization/test_model_metrics.py`
- Create: `<root>/tests/characterization/test_prediction_arrays.py`
- Create: `<root>/tests/smoke/__init__.py`
- Create: `<root>/tests/smoke/test_pipeline_end_to_end.py`

**Prerequisite:** Before writing the metric characterization tests, run `tf-train-all` once on the committed `data/raw/traffic_data.csv` so the new (post-F1) metrics and prediction arrays are present. Read the resulting `reports/model_comparison.csv` to fill in the actual LSTM numbers below.

- [ ] **Step 1: Regenerate ground-truth artifacts**

```bash
cd <root>
tf-train-all
cat reports/model_comparison.csv
```

Record the LSTM row's MAE and RMSE for use in the test below.

- [ ] **Step 2: Write `conftest.py`**

`<root>/tests/conftest.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def real_raw_df():
    return pd.read_csv(ROOT / "data" / "raw" / "traffic_data.csv", parse_dates=["DateTime"])


@pytest.fixture(scope="session")
def model_comparison():
    return pd.read_csv(ROOT / "reports" / "model_comparison.csv").set_index("Model")
```

- [ ] **Step 3: Write data-contract tests**

`<root>/tests/characterization/__init__.py` (empty).

`<root>/tests/characterization/test_data_contract.py`:

```python
import pandas as pd
import pytest

from traffic_forecast import features


@pytest.mark.characterization
@pytest.mark.slow
def test_raw_data_shape(real_raw_df):
    df = real_raw_df
    assert len(df) == 70080
    assert set(df["Junction"].unique()) == {1, 2, 3, 4}
    assert df["Junction"].value_counts().nunique() == 1


@pytest.mark.characterization
@pytest.mark.slow
def test_raw_date_range(real_raw_df):
    assert real_raw_df["DateTime"].min() == pd.Timestamp("2022-01-01 00:00:00")
    assert real_raw_df["DateTime"].max() == pd.Timestamp("2023-12-31 23:00:00")


@pytest.mark.characterization
@pytest.mark.slow
def test_raw_has_approx_70_missing(real_raw_df):
    n_missing = real_raw_df["Vehicles"].isna().sum()
    assert 40 <= n_missing <= 100


@pytest.mark.characterization
@pytest.mark.slow
def test_engineered_features_shape(real_raw_df):
    feat = features.engineer_features(real_raw_df)
    assert len(feat) == 69984
    assert feat.isna().sum().sum() == 0
    for col in features.__dict__.get("FEATURE_COLS", []) or []:
        # FEATURE_COLS lives in config now; just sanity-check the canonical set
        pass
    for col in (
        "hour",
        "dayofweek",
        "is_weekend",
        "is_holiday",
        "lag_1",
        "lag_24",
        "roll_mean_3",
        "roll_mean_24",
        "is_outlier",
    ):
        assert col in feat.columns


@pytest.mark.characterization
@pytest.mark.slow
def test_chronological_split_no_leakage(real_raw_df):
    feat = features.engineer_features(real_raw_df)
    train, val = features.chronological_split(feat, val_frac=0.2)
    assert train["DateTime"].max() < val["DateTime"].min()
```

- [ ] **Step 4: Write metric characterization tests**

`<root>/tests/characterization/test_model_metrics.py`:

```python
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"


def _load(name):
    with open(REPORTS / name) as f:
        return json.load(f)


@pytest.mark.characterization
@pytest.mark.slow
def test_linear_regression_mae_within_10pct():
    r = _load("results_stage1.json")["Linear Regression"]
    assert 3.73 <= r["MAE"] <= 4.55


@pytest.mark.characterization
@pytest.mark.slow
def test_linear_regression_rmse_within_10pct():
    r = _load("results_stage1.json")["Linear Regression"]
    assert 5.12 <= r["RMSE"] <= 6.26


@pytest.mark.characterization
@pytest.mark.slow
def test_rf_tuned_mae_within_10pct():
    r = _load("results_stage1.json")["Random Forest (tuned)"]
    assert 2.96 <= r["MAE"] <= 3.62


@pytest.mark.characterization
@pytest.mark.slow
def test_rf_tuned_rmse_within_10pct():
    r = _load("results_stage1.json")["Random Forest (tuned)"]
    assert 4.12 <= r["RMSE"] <= 5.04


@pytest.mark.characterization
@pytest.mark.slow
def test_rf_best_params_unchanged():
    with open(REPORTS / "best_rf_params.json") as f:
        params = json.load(f)
    assert params == {"max_depth": None, "min_samples_leaf": 1, "n_estimators": 200}


@pytest.mark.characterization
@pytest.mark.slow
def test_sarima_mae_within_10pct():
    r = _load("results_stage2_sarima.json")["SARIMA"]
    assert 4.10 <= r["MAE"] <= 5.01


@pytest.mark.characterization
@pytest.mark.slow
def test_lstm_mae_within_post_fix_band():
    """Post-F1 the LSTM MAE should land in [3.0, 5.5]. If this fails ABOVE the
    band, the LSTM has regressed back toward the constant-output collapse. If
    it fails BELOW, sub-project #2 has improved it (update the band
    deliberately, do not widen silently)."""
    r = _load("results_stage3_lstm.json")["LSTM"]
    assert 3.0 <= r["MAE"] <= 5.5


@pytest.mark.characterization
@pytest.mark.slow
def test_model_ranking_by_mae(model_comparison):
    mae = model_comparison["MAE"].sort_values()
    assert mae.iloc[0] < mae.iloc[-1]
    assert mae.iloc[-1] < 12  # no model is as bad as the pre-fix LSTM
```

- [ ] **Step 5: Write prediction-array tests**

`<root>/tests/characterization/test_prediction_arrays.py`:

```python
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports"


@pytest.mark.characterization
@pytest.mark.slow
def test_pred_linear_shape_and_finite():
    arr = np.load(REPORTS / "pred_linear.npy")
    assert arr.shape == (14000,)
    assert np.isfinite(arr).all()


@pytest.mark.characterization
@pytest.mark.slow
def test_pred_rf_shape_and_nonneg():
    arr = np.load(REPORTS / "pred_rf_tuned.npy")
    assert arr.shape == (14000,)
    assert (arr >= 0).all()


@pytest.mark.characterization
@pytest.mark.slow
def test_pred_lstm_shape_and_non_constant():
    """The single most important regression-catch. The pre-fix LSTM emitted
    a literal constant (std=0.00) for all 13,904 val sequences. Post-fix the
    prediction distribution must have non-trivial spread."""
    arr = np.load(REPORTS / "pred_lstm.npy")
    assert arr.shape == (13904,)
    assert arr.std() > 1.0, f"LSTM collapsed to constant (std={arr.std():.4f})"


@pytest.mark.characterization
@pytest.mark.slow
def test_actual_lstm_shape_and_nonneg():
    arr = np.load(REPORTS / "actual_lstm.npy")
    assert arr.shape == (13904,)
    assert (arr >= 0).all()
```

- [ ] **Step 6: Write the end-to-end smoke test**

`<root>/tests/smoke/__init__.py` (empty).

`<root>/tests/smoke/test_pipeline_end_to_end.py`:

```python
"""Cheapest possible 'the whole pipeline still runs' test. Uses a tiny
synthetic fixture generated on the fly so it runs in under 90 seconds."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from traffic_forecast import config
from traffic_forecast.pipeline import run_all


@pytest.mark.timeout(180)
def test_pipeline_runs_end_to_end_on_tiny_fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    rows = []
    for jid in range(1, 5):
        base = 30 + jid * 5
        for h in range(24 * 75):
            rows.append(
                {
                    "DateTime": pd.Timestamp("2023-01-01") + pd.Timedelta(hours=h),
                    "Junction": jid,
                    "Vehicles": int(base + 10 * np.sin(h / 24 * 2 * np.pi) + rng.normal(0, 3)),
                }
            )
    df = pd.DataFrame(rows)

    raw_path = tmp_path / "raw.csv"
    df.to_csv(raw_path, index=False)

    (tmp_path / "processed").mkdir()
    (tmp_path / "reports").mkdir()
    (tmp_path / "models").mkdir()

    monkeypatch.setattr(config, "RAW_PATH", raw_path)
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(config, "TRAIN_PKL", tmp_path / "processed" / "train_df.pkl")
    monkeypatch.setattr(config, "VAL_PKL", tmp_path / "processed" / "val_df.pkl")
    monkeypatch.setattr(config, "FEATURES_PATH", tmp_path / "processed" / "features.csv")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path / "models")

    run_all.main(epochs=2)

    for fname in ("model_comparison.csv", "predictions_full.csv", "pred_rf_tuned.npy"):
        assert (tmp_path / "reports" / fname).exists()
    assert (tmp_path / "models" / "random_forest_tuned.joblib").exists()
```

- [ ] **Step 7: Run the full test suite**

```bash
cd <root>
pytest -v
```

Expected: all tests pass (characterization tests are slow; smoke test ~60-90s).

- [ ] **Step 8: Commit**

```bash
cd <root>
git add -A
git commit -m "tests: characterization + smoke suite locks current behavior"
```

---

### Task 12: README factual fixes + CI workflow + Streamlit Cloud requirements export

**Goal:** Apply F12 (drop dead `per_junction_rf.json` reference) and F13 (fix the misleading results table). Add the GitHub Actions CI workflow. Re-export a `requirements.txt` for Streamlit Cloud compatibility.

**Files:**
- Modify: `<root>/README.md`
- Create: `<root>/.github/workflows/ci.yml`
- Modify: `<root>/requirements.txt` (re-export from pyproject)

- [ ] **Step 1: Fix README results table**

In `<root>/README.md`, replace the results table block with values read live from `reports/model_comparison.csv`. Open the file and find the section under "## Results on this run (validation set):" and replace it with:

```markdown
## Results on this run (validation set)

| Model | MAE | RMSE |
|---|---|---|
| Random Forest (tuned) | 3.29 | 4.58 |
| Random Forest (default) | 3.32 | 4.60 |
| Linear Regression | 4.14 | 5.69 |
| SARIMA | 4.56 | 6.50 |
| LSTM | (post-fix value from CSV) | (post-fix value from CSV) |
```

Substitute the LSTM row with the actual values from `reports/model_comparison.csv` after Task 11 step 1. If the LSTM MAE landed at e.g. 4.1 / RMSE 5.6, use those numbers.

Also update the closing paragraph below the table to remove "LSTM needs more data/tuning to close the gap" and replace with a note that the LSTM collapse was diagnosed and fixed (target scaling + grad clipping + early stopping + junction feature).

- [ ] **Step 2: Drop the dead per-junction reference**

In `<root>/README.md`, find the bullet:

```
- Try the per-junction Random Forest variant (already computed in
  `reports/per_junction_rf.json` — Week 4 found it helps the busiest
  junction but hurts the quieter ones with less data).
```

Replace with:

```
- Try the per-junction Random Forest variant (Week 4 found it helps the
  busiest junction but hurts the quieter ones with less data). Not yet
  re-implemented in the packaged pipeline.
```

- [ ] **Step 3: Add a "Run the pipeline first" note to the dashboard section**

Find the "## Launch the dashboard" section. Insert this paragraph directly under the `streamlit run dashboard/app.py` command:

```markdown
> The dashboard reads `reports/predictions_full.csv`, `reports/model_comparison.csv`,
> `reports/error_by_junction_hour.csv`, and `models/random_forest_tuned.joblib`.
> If you haven't run the training pipeline yet, run `tf-train-all` first or the
> dashboard will throw `FileNotFoundError` on launch.
```

- [ ] **Step 4: Update the setup + run sections to reflect the new console scripts**

Replace the "## Setup" and "## Run the full pipeline" sections with:

```markdown
## Setup

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -e ".[lstm,dashboard,dev]"
```

## Run the full pipeline

```bash
tf-generate-data     # (skip if using your own real dataset at data/raw/traffic_data.csv)
tf-train-stage1      # Linear Regression + Random Forest (~5 min, mostly grid search)
tf-train-stage2      # SARIMA, per junction (~1 min)
tf-train-stage3      # LSTM (~2 min on CPU)
tf-combine           # builds reports/model_comparison.csv + dashboard data
```

Or run everything in one go with `tf-train-all` (does all 4 stages in a single
process — slightly slower since it doesn't checkpoint between stages).
```

- [ ] **Step 5: Write the CI workflow**

`<root>/.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install
        run: uv pip install --system -e ".[lstm,dashboard,dev]"

      - name: Lint (ruff)
        run: ruff check .

      - name: Format check (ruff)
        run: ruff format --check .

      - name: Fast tests
        run: pytest -q -m "not slow"

      - name: Import smoke
        run: |
          python -c "import traffic_forecast; from traffic_forecast import pipeline; from traffic_forecast.dashboard import app"

  characterization:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push'
    steps:
      - uses: actions/checkout@v4

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          python-version: "3.11"

      - name: Install
        run: uv pip install --system -e ".[lstm,dashboard,dev]"

      - name: Generate data + train
        run: |
          tf-generate-data
          tf-train-all

      - name: Characterization tests
        run: pytest -q -m "characterization"
```

- [ ] **Step 6: Re-export `requirements.txt`**

`<root>/requirements.txt` (hand-maintained, for Streamlit Community Cloud which doesn't read optional-dependencies well). Content:

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
statsmodels>=0.14
torch>=2.0
joblib>=1.3
streamlit>=1.30
plotly>=5.18
```

Add a comment at the top:

```
# Runtime dependencies for Streamlit Community Cloud.
# For local development use `pip install -e ".[lstm,dashboard,dev]"` instead.
```

- [ ] **Step 7: Verify lint and tests**

```bash
cd <root>
ruff check .
ruff format --check .
pytest -q -m "not slow"
```

Expected: no lint errors, all fast tests pass.

- [ ] **Step 8: Commit**

```bash
cd <root>
git add -A
git commit -m "docs: fix stale results + dead refs; add CI; export requirements"
```

---

## Final Verification (after Task 12)

- [ ] **Clean-room install** (in a fresh venv):

```bash
cd <root>
python -m venv .venv-clean
source .venv-clean/bin/activate
pip install -e ".[lstm,dashboard,dev]"
```

- [ ] **Full pipeline from scratch**:

```bash
tf-generate-data
tf-train-all
cat reports/model_comparison.csv
```

Expected: all four models present; LSTM MAE in [3.0, 5.5]; LR/RF/SARIMA within 10% of historical values.

- [ ] **Dashboard boots**:

```bash
tf-dashboard &
sleep 5
curl -sf http://localhost:8501/_stcore/health
kill %1
```

Expected: streamlit health endpoint returns "ok".

- [ ] **Full test suite**:

```bash
pytest -v
```

Expected: all tests pass, including slow characterization tests.

- [ ] **Final commit (if any verification step required a fix)**:

```bash
git add -A
git commit -m "verify: clean-room install + dashboard boot + full suite green"
```

---

## Self-Review

**Spec coverage:**
- F1 (LSTM constant): Task 8 ✓
- F2 (reentrant RNG): Task 2 ✓
- F3 (train_models.py torch seed): subsumed by deletion in Task 9 ✓
- F4 (MinMaxScaler fit on train+val): Task 8 ✓
- F5 (is_outlier IQR on full frame): Task 5 + Task 9 `_common.load_and_split` ✓
- F6 (train_models.py duplicates): Task 9 (delete) + Task 9 run_all.py (orchestrator) ✓
- F7 (no __main__ guards): Tasks 2, 9, 10 ✓
- F8 (HOLIDAYS dup): Task 3 ✓
- F9 (SEQ_LEN dup): Task 3 ✓
- F10 (rmse dup): Task 4 ✓
- F11 (path block dup): Task 3 + use across Tasks 5-10 ✓
- F12 (per_junction_rf dead ref): Task 12 ✓
- F13 (results table): Task 12 ✓
- Phasing A-H: Task 2 (A), Task 1 (B), Tasks 3-8 (C), Task 9 (D+E via stages), Task 11 (F), Task 12 (G), Final verification (H) ✓

**Placeholder scan:** One explicit "fill in actual LSTM numbers from CSV" in Task 12 step 1. This is intentional — the value only exists after Task 11 runs. Marked as such, not a placeholder failure.

**Type consistency:** `lr.train`, `rf.train_default`, `rf.train_tuned`, `sarima.train`, `lstm.train` all return dicts with `"metrics"` containing `{"MAE", "RMSE"}` floats. Pipeline stages consume these consistently. `lstm.train` returns `"actual_val"` and `"pred_val"`; stage3 saves both. ✓

**Scope check:** Plan covers exactly sub-project #1 as scoped in the spec. No spillover into dashboard UX, feature engineering, or real frontend. ✓

No issues to fix inline. Plan is ready for execution.

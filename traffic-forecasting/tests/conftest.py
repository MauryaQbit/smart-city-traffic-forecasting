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

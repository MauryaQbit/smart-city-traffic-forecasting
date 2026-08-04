"""
Smart City Traffic Forecasting -- Dashboard
Run with:  tf-dashboard   or   streamlit run src/traffic_forecast/dashboard/app.py

Tabs:
  1. Model Comparison  - KPIs + MAE/RMSE table + bar chart + cost-vs-accuracy
  2. Predicted vs Actual - junction + model selectors, date range, holidays
  3. Error Breakdown   - junction x hour heatmap of RF mean abs error
  4. Try a Prediction  - live form calling the trained RF model
"""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from traffic_forecast import config
from traffic_forecast.dashboard.components import (
    comparison,
    error_heatmap,
    kpis,
    pred_vs_actual,
    predictor,
)
from traffic_forecast.dashboard.theme import UNIT_VEHICLES

REPORTS = config.REPORTS_DIR
MODELS_DIR = config.MODELS_DIR

REQUIRED_ARTIFACTS = [
    ("reports/predictions_full.csv", REPORTS / "predictions_full.csv"),
    ("reports/model_comparison.csv", REPORTS / "model_comparison.csv"),
    ("reports/error_by_junction_hour.csv", REPORTS / "error_by_junction_hour.csv"),
    ("models/random_forest_tuned.joblib", MODELS_DIR / "random_forest_tuned.joblib"),
]


@st.cache_data
def load_data():
    comp = pd.read_csv(REPORTS / "model_comparison.csv").set_index("Model")
    preds = pd.read_csv(REPORTS / "predictions_full.csv", parse_dates=["DateTime"])
    err = pd.read_csv(REPORTS / "error_by_junction_hour.csv")
    return comp, preds, err


@st.cache_resource
def load_rf_model():
    return joblib.load(MODELS_DIR / "random_forest_tuned.joblib")


def require_files():
    """Friendly error if the user lands here without training first, or if
    a stage wrote an empty/header-only artifact."""
    missing = [
        label for label, path in REQUIRED_ARTIFACTS if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        st.error(
            "Required artifacts are missing or empty. Run the training pipeline first:\n\n"
            "```\ntf-train-all\n```\n\n"
            f"Missing: {', '.join(missing)}"
        )
        st.stop()


st.set_page_config(page_title="Smart City Traffic Forecasting", layout="wide")
require_files()

comp, preds, err = load_data()

st.title("Smart City Traffic Forecasting")
st.caption(
    "Project 9 - ML Internship. Hourly vehicle counts at 4 city junctions, "
    "forecast by Random Forest, SARIMA, LSTM and Linear Regression."
)
kpis.render_executive_summary(comp, UNIT_VEHICLES)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Model Comparison", "Predicted vs Actual", "Error Breakdown", "Try a Prediction"]
)

with tab1:
    kpis.render_kpi_row(comp, val_rows=len(preds))
    comparison.render(comp)

with tab2:
    pred_vs_actual.render(preds)

with tab3:
    error_heatmap.render(err)

with tab4:
    predictor.render(preds, load_rf_model())

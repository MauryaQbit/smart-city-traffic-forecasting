"""Dashboard theme: shared palette, plotly template, model metadata.

One source of truth so charts, legends, and the predicted-vs-actual lines
never drift out of sync.
"""

from __future__ import annotations

PALETTE = {
    "actual": "#1f2328",
    "pred_linear": "#0072B2",
    "pred_rf_tuned": "#009E73",
    "pred_sarima": "#D55E00",
    "pred_lstm": "#CC79A7",
    "holiday": "#56B4E9",
}

MODELS = {
    "pred_linear": {"label": "Linear Regression", "color": PALETTE["pred_linear"]},
    "pred_rf_tuned": {"label": "Random Forest (tuned)", "color": PALETTE["pred_rf_tuned"]},
    "pred_sarima": {"label": "SARIMA", "color": PALETTE["pred_sarima"]},
    "pred_lstm": {"label": "LSTM", "color": PALETTE["pred_lstm"]},
}

PLOTLY_TEMPLATE = "plotly_white"

UNIT_VEHICLES = "vehicles/hr"

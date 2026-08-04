"""Try a Prediction tab: grouped form calling the trained RF model.

Predictor inputs are grouped semantically (Time / Lag features / Rolling
features / Flags) instead of scattered across a 3-column soup. The last
prediction persists in session_state across reruns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st


def _build_row(inputs: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "hour": inputs["hour"],
                "dayofweek": inputs["dow"],
                "month": inputs["month"],
                "is_weekend": inputs["is_weekend"],
                "is_holiday": inputs["is_holiday"],
                "hour_sin": float(np.sin(2 * np.pi * inputs["hour"] / 24)),
                "hour_cos": float(np.cos(2 * np.pi * inputs["hour"] / 24)),
                "hour_sin_2": float(np.sin(2 * np.pi * inputs["hour"] / 12)),
                "hour_cos_2": float(np.cos(2 * np.pi * inputs["hour"] / 12)),
                "dow_sin": float(np.sin(2 * np.pi * inputs["dow"] / 7)),
                "dow_cos": float(np.cos(2 * np.pi * inputs["dow"] / 7)),
                "lag_1": inputs["lag_1"],
                "lag_24": inputs["lag_24"],
                "lag_168": inputs["lag_168"],
                "roll_mean_3": inputs["roll_3"],
                "roll_mean_24": inputs["roll_24"],
                "roll_mean_168": inputs["roll_168"],
                "is_outlier": inputs["is_outlier"],
                "Junction": inputs["junction"],
            }
        ]
    )


def render(preds: pd.DataFrame, rf_model) -> None:
    st.subheader("Try a prediction")
    st.caption(
        "Uses the tuned Random Forest model. Fill in the fields to estimate traffic volume for a given hour."
    )

    junctions = sorted(preds["Junction"].unique())

    with st.expander("Time", expanded=True):
        tc1, tc2, tc3 = st.columns(3)
        junction_in = tc1.selectbox("Junction", junctions, key="pred_junction")
        hour_in = tc2.slider("Hour of day", 0, 23, 9, key="pred_hour")
        dow_in = tc3.selectbox(
            "Day of week",
            options=list(range(7)),
            format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x],
            key="pred_dow",
        )
        month_in = tc1.slider("Month", 1, 12, 6, key="pred_month")

    with st.expander("Lag features (recent traffic at this junction)", expanded=True):
        lc1, lc2, lc3 = st.columns(3)
        lag_1_in = lc1.number_input("1 hour ago", min_value=0, value=30, key="pred_lag_1")
        lag_24_in = lc2.number_input(
            "24h ago (same hour yesterday)", min_value=0, value=30, key="pred_lag_24"
        )
        lag_168_in = lc3.number_input(
            "1 week ago (same hour last week)", min_value=0, value=30, key="pred_lag_168"
        )

    with st.expander("Rolling features (recent averages)", expanded=True):
        rc1, rc2, rc3 = st.columns(3)
        roll3_in = rc1.number_input(
            "Avg, last 3 hours", min_value=0.0, value=30.0, key="pred_roll_3"
        )
        roll24_in = rc2.number_input(
            "Avg, last 24 hours", min_value=0.0, value=30.0, key="pred_roll_24"
        )
        roll168_in = rc3.number_input(
            "Avg, last 7 days", min_value=0.0, value=30.0, key="pred_roll_168"
        )

    with st.expander("Flags", expanded=False):
        fc1, fc2 = st.columns(2)
        is_holiday_in = fc1.checkbox("Holiday?", key="pred_holiday")
        outlier_in = fc2.checkbox("Flag as statistical outlier context", key="pred_outlier")

    inputs = {
        "junction": junction_in,
        "hour": hour_in,
        "dow": dow_in,
        "month": month_in,
        "is_weekend": 1 if dow_in >= 5 else 0,
        "is_holiday": int(is_holiday_in),
        "lag_1": lag_1_in,
        "lag_24": lag_24_in,
        "lag_168": lag_168_in,
        "roll_3": roll3_in,
        "roll_24": roll24_in,
        "roll_168": roll168_in,
        "is_outlier": outlier_in,
    }

    if "last_prediction" not in st.session_state:
        st.session_state["last_prediction"] = None

    btn_col, result_col = st.columns([1, 3])
    if btn_col.button("Predict traffic volume", type="primary", key="pred_button"):
        row = _build_row(inputs)
        pred = float(rf_model.predict(row)[0])
        st.session_state["last_prediction"] = pred

    if st.session_state["last_prediction"] is not None:
        result_col.metric("Predicted vehicle count", f"{st.session_state['last_prediction']:.0f}")

"""Error Breakdown tab: junction x hour heatmap."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from traffic_forecast.dashboard.theme import PLOTLY_TEMPLATE, UNIT_VEHICLES


def render(err: pd.DataFrame) -> None:
    st.subheader("Where the model struggles: error by junction & hour")
    if err.empty:
        st.info("No error breakdown available.")
        return

    pivot = err.pivot(index="Junction", columns="hour", values="abs_error_rf")
    fig = px.imshow(
        pivot,
        color_continuous_scale="Reds",
        title=f"Mean absolute error, Random Forest ({UNIT_VEHICLES})",
        template=PLOTLY_TEMPLATE,
        aspect="auto",
        labels=dict(x="Hour of day", y="Junction", color="MAE"),
    )
    fig.update_layout(height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Errors are consistently higher during rush hours and at the busiest junction - "
        "exactly what the Week 3 error analysis found."
    )

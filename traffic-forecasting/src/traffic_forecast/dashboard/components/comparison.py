"""Model Comparison tab: table + bar chart + cost-vs-accuracy + methodology."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from traffic_forecast.dashboard.components.kpis import fmt_seconds
from traffic_forecast.dashboard.theme import PLOTLY_TEMPLATE, UNIT_VEHICLES


def build_comparison_bar(comp_df: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        comp_df.reset_index(),
        x="Model",
        y="MAE",
        color="Model",
        color_discrete_sequence=px.colors.qualitative.Safe,
        title=f"MAE by model ({UNIT_VEHICLES}, lower is better)",
        template=PLOTLY_TEMPLATE,
    )
    fig.update_layout(showlegend=False, height=380)
    return fig


def build_cost_chart(comp_df: pd.DataFrame) -> go.Figure:
    df = comp_df.reset_index().copy()
    df["train_seconds"] = df["train_seconds"].fillna(0)
    df["MAE"] = df["MAE"].astype(float)
    fig = px.scatter(
        df,
        x="train_seconds",
        y="MAE",
        text="Model",
        title="Cost vs accuracy: training time vs MAE",
        template=PLOTLY_TEMPLATE,
        labels={"train_seconds": "training time (s)", "MAE": f"MAE ({UNIT_VEHICLES})"},
    )
    fig.update_traces(textposition="top center", marker_size=12)
    fig.update_layout(height=380)
    return fig


def render(comp: pd.DataFrame) -> None:
    if comp.empty:
        st.info("No comparison data yet.")
        return

    st.subheader("Model comparison (validation set)")
    comp_display = comp.copy()
    comp_display["train_seconds"] = comp_display["train_seconds"].apply(fmt_seconds)
    st.dataframe(
        comp_display[["MAE", "RMSE", "train_seconds"]].style.highlight_min(
            subset=["MAE", "RMSE"], color="lightgreen"
        ),
        use_container_width=True,
    )

    left, right = st.columns(2)
    left.plotly_chart(build_comparison_bar(comp), use_container_width=True)
    if "train_seconds" in comp.columns:
        right.plotly_chart(build_cost_chart(comp), use_container_width=True)

    with st.expander("How these numbers were computed", expanded=False):
        st.markdown(
            "- **Train/val split:** chronological (no shuffle), last ~20.6 weeks of hours held out.\n"
            "- **LR / RF / LSTM:** evaluated on the full validation window (~13,884 rows).\n"
            "- **SARIMA:** fit on the trailing 2 months of train per junction, evaluated on the "
            "first 2 weeks of validation. This makes its number not directly comparable to the "
            "other three (it sees a smaller, holiday-lighter window) but reflects how SARIMA is "
            "typically deployed in practice.\n"
            "- **LSTM:** trained with target scaling, gradient clipping, early stopping on val "
            "loss, and Junction as an input feature. Earlier reports had the LSTM trailing badly "
            "due to a training bug (no scaling, no clipping); fixing that brought it to the top."
        )
    st.download_button(
        "Download comparison CSV",
        comp.to_csv().encode(),
        file_name="model_comparison.csv",
        mime="text/csv",
    )

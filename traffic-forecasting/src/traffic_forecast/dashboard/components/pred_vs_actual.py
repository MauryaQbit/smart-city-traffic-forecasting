"""Predicted vs Actual tab with deep-linkable state via st.query_params."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from traffic_forecast.dashboard.theme import MODELS, PALETTE, PLOTLY_TEMPLATE, UNIT_VEHICLES


def build_figure(sub: pd.DataFrame, selected: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sub["DateTime"],
            y=sub["Vehicles"],
            name="Actual",
            line=dict(color=PALETTE["actual"], width=2),
        )
    )
    for key in selected:
        col = sub[key]
        if col.isna().all():
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["DateTime"],
                y=col,
                name=MODELS[key]["label"],
                line=dict(color=MODELS[key]["color"], dash="dot", width=1.6),
            )
        )

    holiday_days = sub.loc[sub["is_holiday"] == 1, "DateTime"].dt.normalize().unique()
    for day in holiday_days:
        fig.add_vrect(
            x0=pd.Timestamp(day),
            x1=pd.Timestamp(day) + pd.Timedelta(days=1),
            fillcolor=PALETTE["holiday"],
            opacity=0.18,
            layer="below",
            line_width=0,
        )
    if len(holiday_days) > 0:
        # Single legend entry for all vrects via an invisible marker trace.
        fig.add_trace(
            go.Scatter(
                x=[pd.Timestamp(holiday_days[0])],
                y=[sub["Vehicles"].max()],
                mode="markers",
                marker=dict(color=PALETTE["holiday"], size=10, symbol="square"),
                name="Holiday",
                showlegend=True,
            )
        )

    fig.update_layout(
        height=480,
        xaxis_title="Time",
        yaxis_title=f"Vehicle count ({UNIT_VEHICLES})",
        template=PLOTLY_TEMPLATE,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _read_junction_from_url(default: int) -> int:
    raw = st.query_params.get("junction")
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _read_models_from_url(available: list[str]) -> list[str] | None:
    raw = st.query_params.get("models")
    if not raw:
        return None
    keys = [m.strip() for m in raw.split(",") if m.strip() in available]
    return keys or None


def render(preds: pd.DataFrame) -> None:
    st.subheader("Predicted vs Actual traffic volume")
    if preds.empty:
        st.info("No predictions available.")
        return

    junctions = sorted(preds["Junction"].unique())
    default_junction = junctions[0]
    initial_junction = _read_junction_from_url(default_junction)
    if initial_junction not in junctions:
        initial_junction = default_junction

    available_models = [m for m in MODELS if m in preds.columns and not preds[m].isna().all()]
    initial_models = _read_models_from_url(available_models) or (
        ["pred_rf_tuned"] if "pred_rf_tuned" in available_models else available_models[:1]
    )

    col1, col2 = st.columns(2)
    junction = col1.selectbox(
        "Junction", junctions, index=junctions.index(initial_junction), key="pva_junction"
    )
    selected = col2.multiselect(
        "Models to show",
        available_models,
        default=initial_models,
        format_func=lambda x: MODELS[x]["label"],
        key="pva_models",
    )

    # Sync to URL for shareable deep links.
    st.query_params["junction"] = str(junction)
    if selected:
        st.query_params["models"] = ",".join(selected)
    elif "models" in st.query_params:
        del st.query_params["models"]

    sub_all = preds[preds["Junction"] == junction].sort_values("DateTime")
    if sub_all.empty:
        st.info(f"No data for junction {junction}.")
        return

    date_range = st.slider(
        "Date range",
        min_value=sub_all["DateTime"].min().to_pydatetime(),
        max_value=sub_all["DateTime"].max().to_pydatetime(),
        value=(
            sub_all["DateTime"].min().to_pydatetime(),
            sub_all["DateTime"].min().to_pydatetime() + pd.Timedelta(days=7),
        ),
        key="pva_date_range",
    )
    sub = sub_all[(sub_all["DateTime"] >= date_range[0]) & (sub_all["DateTime"] <= date_range[1])]
    if sub.empty:
        st.info("No rows in the selected range.")
        return

    st.plotly_chart(build_figure(sub, selected), use_container_width=True)
    st.download_button(
        "Download filtered rows",
        sub.to_csv(index=False).encode(),
        file_name=f"predictions_junction_{junction}.csv",
        mime="text/csv",
    )

"""KPI row + executive summary at the top of the Model Comparison tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def fmt_seconds(s: float) -> str:
    if pd.isna(s):
        return "-"
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(s), 60)
    return f"{m}m {sec}s"


def render_executive_summary(comp: pd.DataFrame, unit: str) -> None:
    if comp.empty or "MAE" not in comp.columns:
        return
    best_idx = comp["MAE"].idxmin()
    best_label = comp.loc[best_idx].name
    best_mae = float(comp.loc[best_idx, "MAE"])
    worst_mae = float(comp["MAE"].max())
    improvement = (worst_mae - best_mae) / worst_mae * 100 if worst_mae else 0.0
    st.caption(
        f"**Headline:** {best_label} wins at MAE {best_mae:.2f} {unit} - "
        f"{improvement:.0f}% better than the worst model on this run."
    )


def render_kpi_row(comp: pd.DataFrame, val_rows: int) -> None:
    if comp.empty:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best MAE", f"{comp['MAE'].min():.2f}", help=f"by {comp['MAE'].idxmin()}")
    c2.metric("Best RMSE", f"{comp['RMSE'].min():.2f}", help=f"by {comp['RMSE'].idxmin()}")
    fastest_idx = comp["train_seconds"].idxmin()
    c3.metric(
        "Fastest to train",
        fmt_seconds(comp["train_seconds"].min()),
        help=f"by {comp.loc[fastest_idx].name}",
    )
    c4.metric("Validation rows", f"{val_rows:,}")

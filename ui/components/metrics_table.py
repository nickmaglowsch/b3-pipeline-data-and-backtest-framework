"""
Metrics Table and Summary Components
=====================================
Reusable Streamlit components for displaying performance metrics, parameter
summaries, and database statistics. Used across Pipeline, Backtest, and Dashboard pages.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


def render_metrics_table(metrics_list: list[dict], highlight_best: bool = True) -> None:
    """
    Display a performance metrics table with number formatting.

    Args:
        metrics_list:   List of dicts from build_metrics().
        highlight_best: Currently unused; reserved for future column highlighting.
    """
    if not metrics_list:
        st.info("No metrics to display.")
        return

    df = pd.DataFrame(metrics_list)
    if "Strategy" in df.columns:
        df = df.set_index("Strategy")

    st.dataframe(
        df,
        column_config={
            "Ann. Return (%)": st.column_config.NumberColumn(
                "Ann. Return", format="%.2f%%"
            ),
            "Ann. Volatility (%)": st.column_config.NumberColumn(
                "Ann. Vol", format="%.2f%%"
            ),
            "Sharpe": st.column_config.NumberColumn(format="%.2f"),
            "Max Drawdown (%)": st.column_config.NumberColumn(
                "Max DD", format="%.2f%%"
            ),
            "Calmar": st.column_config.NumberColumn(format="%.2f"),
        },
        use_container_width=True,
    )


def render_metrics_cards(metrics: dict) -> None:
    """
    Display a single strategy's metrics as Streamlit metric cards (one row of 5).

    Args:
        metrics: Single dict from build_metrics().
    """
    if not metrics:
        return
    cols = st.columns(5)
    cols[0].metric("Ann. Return", f"{metrics.get('Ann. Return (%)', 0):.2f}%")
    cols[1].metric("Ann. Volatility", f"{metrics.get('Ann. Volatility (%)', 0):.2f}%")
    cols[2].metric("Sharpe", f"{metrics.get('Sharpe', 0):.2f}")
    cols[3].metric("Max Drawdown", f"{metrics.get('Max Drawdown (%)', 0):.2f}%")
    cols[4].metric("Calmar", f"{metrics.get('Calmar', 0):.2f}")


def render_parameters_table(params: dict, specs: Optional[list] = None) -> None:
    """
    Display strategy parameters as a formatted table.

    Args:
        params: Dict of {param_name: value}.
        specs:  Optional list of ParameterSpec for human-readable labels and descriptions.
    """
    if not params:
        st.info("No parameters to display.")
        return

    if specs:
        spec_map = {s.name: s for s in specs}
        rows = []
        for name, value in params.items():
            spec = spec_map.get(name)
            rows.append({
                "Parameter": spec.label if spec else name,
                "Value": str(value),
                "Description": spec.description if spec else "",
            })
        df = pd.DataFrame(rows)
    else:
        df = pd.DataFrame([
            {"Parameter": k, "Value": str(v)}
            for k, v in params.items()
        ])

    st.dataframe(df, use_container_width=True, hide_index=True)


def render_db_stats(stats: dict) -> None:
    """
    Display database summary statistics as metric cards (two rows of 3).

    Args:
        stats: Dict from storage.get_summary_stats().
    """
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Price Records", f"{stats.get('total_prices', 0):,}")
    col2.metric("Unique Tickers", f"{stats.get('total_tickers', 0):,}")
    col3.metric("Unique ISINs", f"{stats.get('total_isins', 0):,}")

    col4, col5, col6 = st.columns(3)
    date_range = stats.get("date_range", ("N/A", "N/A"))
    date_min, date_max = date_range
    col4.metric("Date Range", f"{date_min} to {date_max}")
    col5.metric("Corporate Actions", f"{stats.get('total_corporate_actions', 0):,}")
    col6.metric("Stock Actions", f"{stats.get('total_stock_actions', 0):,}")

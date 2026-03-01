"""
Results Dashboard Page -- implemented fully.
"""
from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Dashboard", page_icon="bar_chart", layout="wide")

st.title("Results Dashboard")
st.caption("Browse, compare, and analyze all backtest results.")

try:
    from ui.services.result_store import ResultStore
    from ui.components.charts import (
        plot_equity_curves,
        plot_drawdown,
        plot_tax_detail,
        plot_strategy_comparison,
        plot_correlation_heatmap,
    )
    from ui.components.metrics_table import render_metrics_table
    _LOADED = True
except ImportError as e:
    _LOADED = False
    st.warning(f"Dashboard services not yet available: {e}")

if not _LOADED:
    st.info("This page will be fully functional after all task implementations are complete.")
    st.stop()

store = ResultStore()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.subheader("Filters")
show_legacy = st.sidebar.checkbox("Show legacy results", value=True)

try:
    all_results = store.list_results()
except Exception as e:
    st.error(f"Error loading results: {e}")
    all_results = []

if not show_legacy:
    all_results = [r for r in all_results if not r.is_legacy]

strategy_options = sorted(set(r.strategy_name for r in all_results))
selected_strategies = st.sidebar.multiselect(
    "Filter by strategy",
    options=strategy_options,
    default=[],
)

if selected_strategies:
    all_results = [r for r in all_results if r.strategy_name in selected_strategies]

if not all_results:
    st.info("No results found. Run some backtests from the Backtest Runner page.")
    st.stop()

# ── Results Table ─────────────────────────────────────────────────────────────
st.subheader(f"Results ({len(all_results)} total)")

results_rows = []
for r in all_results:
    sharpe = None
    ann_ret = None
    max_dd = None
    if r.metrics:
        # Find after-tax row
        for m in r.metrics:
            if "After-Tax" in m.get("Strategy", "") or (not r.is_legacy and not sharpe):
                sharpe = m.get("Sharpe")
                ann_ret = m.get("Ann. Return (%)")
                max_dd = m.get("Max Drawdown (%)")
                break

    results_rows.append({
        "result_id": r.result_id,
        "Strategy": r.strategy_name,
        "Date": r.timestamp,
        "Sharpe": sharpe,
        "Ann. Return (%)": ann_ret,
        "Max Drawdown (%)": max_dd,
        "Type": "Legacy (CLI)" if r.is_legacy else "New",
    })

results_df = pd.DataFrame(results_rows)

# Multi-select via checkboxes using st.data_editor
select_col = [False] * len(results_df)
results_df.insert(0, "Select", select_col)

edited_df = st.data_editor(
    results_df.drop(columns=["result_id"]),
    column_config={
        "Select": st.column_config.CheckboxColumn("Select", default=False),
        "Sharpe": st.column_config.NumberColumn(format="%.2f"),
        "Ann. Return (%)": st.column_config.NumberColumn(format="%.2f%%"),
        "Max Drawdown (%)": st.column_config.NumberColumn(format="%.2f%%"),
    },
    use_container_width=True,
    hide_index=True,
    key="results_table",
)

selected_indices = edited_df[edited_df["Select"]].index.tolist()
selected_results = [all_results[i] for i in selected_indices]

st.divider()

# ── Individual Result View ────────────────────────────────────────────────────
if len(selected_results) == 1:
    r = selected_results[0]
    st.subheader(f"Result: {r.strategy_name} ({r.timestamp})")

    if r.is_legacy:
        if r.legacy_png_path and r.legacy_png_path.exists():
            st.image(str(r.legacy_png_path))
            st.caption(f"File: {r.legacy_png_path}")
        st.info("Legacy result -- run this strategy from the Backtest Runner for interactive charts.")
    else:
        data = store.load_data(r)
        if data:
            tab_eq, tab_dd, tab_tax, tab_met, tab_params = st.tabs([
                "Equity Curves", "Drawdown", "Tax Detail", "Metrics", "Parameters",
            ])
            with tab_eq:
                try:
                    fig = plot_equity_curves(
                        data["pretax_values"],
                        data["aftertax_values"],
                        data["ibov_ret"],
                        data.get("cdi_ret"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
            with tab_dd:
                try:
                    fig = plot_drawdown(
                        data["pretax_values"],
                        data["aftertax_values"],
                        data["ibov_ret"],
                        data.get("cdi_ret"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
            with tab_tax:
                try:
                    fig = plot_tax_detail(
                        data["tax_paid"],
                        data["loss_carryforward"],
                        data["turnover"],
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {e}")
            with tab_met:
                if r.metrics:
                    render_metrics_table(r.metrics)
            with tab_params:
                if r.params:
                    params_df = pd.DataFrame([
                        {"Parameter": k, "Value": str(v)}
                        for k, v in r.params.items()
                    ])
                    st.dataframe(params_df, use_container_width=True, hide_index=True)

        # Delete button
        if st.button("Delete this result", key=f"del_{r.result_id}"):
            store.delete(r.result_id)
            st.success("Result deleted.")
            st.rerun()

# ── Comparison Mode ───────────────────────────────────────────────────────────
elif len(selected_results) >= 2:
    st.subheader(f"Comparing {len(selected_results)} strategies")

    non_legacy = [r for r in selected_results if not r.is_legacy]

    if len(non_legacy) >= 2:
        tab_overlay, tab_metrics, tab_corr = st.tabs([
            "Equity Overlay", "Metrics Comparison", "Correlation Matrix",
        ])

        equity_curves = {}
        for r in non_legacy:
            data = store.load_data(r)
            if data and "aftertax_values" in data:
                curve = data["aftertax_values"]
                curve = curve / curve.iloc[0]
                # Use a unique key that includes the timestamp to avoid
                # silently dropping duplicate strategy names (bug #21).
                label = f"{r.strategy_name} ({r.timestamp})"
                equity_curves[label] = curve

        with tab_overlay:
            if equity_curves:
                try:
                    fig = plot_strategy_comparison(equity_curves)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Comparison chart error: {e}")
            else:
                st.info("No data available for selected results.")

        with tab_metrics:
            all_metrics = []
            for r in non_legacy:
                if r.metrics:
                    all_metrics.extend(r.metrics)
            if all_metrics:
                render_metrics_table(all_metrics)

        with tab_corr:
            if len(equity_curves) >= 3:
                returns_dict = {}
                for name, curve in equity_curves.items():
                    returns_dict[name] = curve.pct_change().dropna()
                returns_df = pd.DataFrame(returns_dict).dropna()
                if not returns_df.empty:
                    try:
                        fig = plot_correlation_heatmap(returns_df)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Correlation chart error: {e}")
            else:
                st.info("Select 3 or more non-legacy results for correlation matrix.")
    else:
        st.info("Select 2 or more non-legacy results to enable comparison charts.")

# ── Legacy CSV Files ──────────────────────────────────────────────────────────
st.divider()
with st.expander("Legacy CSV Analysis Files", expanded=False):
    project_root = Path(_PROJECT_ROOT)
    csv_files = list((project_root / "backtests").glob("*.csv"))
    if csv_files:
        for csv_path in csv_files:
            st.write(f"**{csv_path.name}**")
            try:
                df = pd.read_csv(csv_path)
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Error reading {csv_path.name}: {e}")
    else:
        st.info("No CSV files found in backtests/")

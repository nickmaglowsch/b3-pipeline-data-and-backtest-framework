"""
Backtest Runner Page -- fully implemented.
"""
from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="Backtest Runner", page_icon="rocket", layout="wide")

st.title("Backtest Runner")
st.caption("Select a strategy, configure parameters, and run a full backtest with real-time log streaming.")

# ── Session state ─────────────────────────────────────────────────────────────
if "job_runner" not in st.session_state:
    from ui.services.job_runner import JobRunner
    st.session_state.job_runner = JobRunner()
if "last_backtest_result" not in st.session_state:
    st.session_state.last_backtest_result = None

try:
    from backtests.core.strategy_registry import get_registry
    from ui.services.backtest_service import run_backtest
    from ui.services.result_store import ResultStore
    from ui.components.parameter_form import render_parameter_form, render_reset_button
    from ui.components.log_stream import render_log_stream
    from ui.components.charts import (
        plot_equity_curves,
        plot_drawdown,
        plot_tax_detail,
    )
    from ui.components.metrics_table import render_metrics_table
    from ui.services.job_runner import JobStatus
    _LOADED = True
except ImportError as e:
    _LOADED = False
    st.warning(f"Some services not yet available: {e}")

if not _LOADED:
    st.info("This page will be fully functional after all task implementations are complete.")
    st.stop()

runner = st.session_state.job_runner

# ── Strategy Selection ────────────────────────────────────────────────────────
st.subheader("Strategy Selection")

try:
    registry = get_registry()
    strategy_names = registry.names()
except Exception as e:
    st.error(f"Could not load strategy registry: {e}")
    st.stop()

if not strategy_names:
    st.warning("No strategies found. Ensure backtests/strategies/ is populated.")
    st.stop()

selected_name = st.selectbox(
    "Select a strategy",
    options=strategy_names,
    key="selected_strategy",
)

strategy = registry.get(selected_name)
st.info(strategy.description)

st.divider()

# ── Parameter Form ────────────────────────────────────────────────────────────
st.subheader("Parameters")

specs = strategy.get_parameter_specs()

# Load previous params if same strategy was run
prev_params = None
if (
    st.session_state.last_backtest_result
    and st.session_state.last_backtest_result.get("strategy_name") == selected_name
):
    prev_params = st.session_state.last_backtest_result.get("params")

col_reset, _ = st.columns([1, 5])
with col_reset:
    render_reset_button(selected_name, specs)

with st.form("backtest_form"):
    params = render_parameter_form(specs, strategy_name=selected_name, defaults=prev_params)
    run_clicked = st.form_submit_button("Run Backtest", type="primary")

st.divider()

# ── Job Execution ─────────────────────────────────────────────────────────────
active_job = runner.get_active_job("backtest")

if run_clicked:
    if runner.is_running("backtest"):
        st.warning("A backtest is already running. Wait for it to complete.")
    else:
        job_id = runner.submit("backtest", run_backtest, selected_name, params)
        st.session_state["current_backtest_job_id"] = job_id
        st.rerun()

current_job_id = st.session_state.get("current_backtest_job_id")

if current_job_id:
    job = runner.get_job(current_job_id)
    if job:
        render_log_stream("backtest", runner)

        if job.status == JobStatus.COMPLETED and job.result:
            result = job.result
            st.session_state.last_backtest_result = result

            # Save result to disk
            try:
                store = ResultStore()
                result_id = store.save(result)
                st.toast(f"Result saved: {result_id}", icon="check_mark")
            except Exception as e:
                st.warning(f"Could not save result: {e}")

        elif job.status == JobStatus.FAILED:
            st.error(f"Backtest failed: {job.error}")

# ── Results Display ───────────────────────────────────────────────────────────
result = st.session_state.last_backtest_result
if result and result.get("strategy_name") == selected_name:
    st.subheader(f"Results: {result['strategy_name']}")

    tab_equity, tab_dd, tab_tax, tab_metrics = st.tabs([
        "Equity Curves", "Drawdown", "Tax Detail", "Metrics",
    ])

    with tab_equity:
        try:
            fig = plot_equity_curves(
                result["pretax_values"],
                result["aftertax_values"],
                result["ibov_ret"],
                result.get("cdi_ret"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render equity curves: {e}")

    with tab_dd:
        try:
            fig = plot_drawdown(
                result["pretax_values"],
                result["aftertax_values"],
                result["ibov_ret"],
                result.get("cdi_ret"),
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render drawdown chart: {e}")

    with tab_tax:
        try:
            fig = plot_tax_detail(
                result["tax_paid"],
                result["loss_carryforward"],
                result["turnover"],
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render tax detail: {e}")

    with tab_metrics:
        try:
            render_metrics_table(result["metrics"])
        except Exception as e:
            st.error(f"Could not render metrics: {e}")

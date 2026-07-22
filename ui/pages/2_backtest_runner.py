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
    from backtests.core.strategy_base import (
        COMMON_CONTRIBUTION,
        COMMON_INITIAL_CAPITAL,
    )
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

# Bug #23 fix: Clear stale job when strategy changes
if st.session_state.get("_prev_selected_strategy") != selected_name:
    st.session_state["_prev_selected_strategy"] = selected_name
    # Drop the previous job id so we don't show stale results
    st.session_state.pop("current_backtest_job_id", None)

strategy = registry.get(selected_name)
st.info(strategy.description)

st.divider()

# ── Parameter Form ────────────────────────────────────────────────────────────
st.subheader("Parameters")

specs = strategy.get_parameter_specs()
# Starting money and buy-ins are engine-level knobs (run_simulation), so offer
# them for every strategy without each one having to declare them.
declared = {s.name for s in specs}
specs = specs + [
    s for s in (COMMON_INITIAL_CAPITAL, COMMON_CONTRIBUTION)
    if s.name not in declared
]

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

            # Bug #22 fix: Only save once per job using a session state flag
            save_flag = f"result_saved_{current_job_id}"
            if not st.session_state.get(save_flag):
                try:
                    store = ResultStore()
                    result_id = store.save(result)
                    st.session_state[save_flag] = True
                    st.toast(f"Result saved: {result_id}", icon="\u2705")
                except Exception as e:
                    st.warning(f"Could not save result: {e}")

        elif job.status == JobStatus.FAILED:
            st.error(f"Backtest failed: {job.error}")

# ── Results Display ───────────────────────────────────────────────────────────
result = st.session_state.last_backtest_result
if result and result.get("strategy_name") == selected_name:
    # Guard: check if the simulation produced any data
    if result.get("pretax_values") is not None and len(result["pretax_values"]) == 0:
        st.warning(
            "The strategy produced no data for the selected date range. "
            "This usually means no stocks passed the filters (e.g., ADTV, price). "
            "Try relaxing the parameters."
        )
        st.stop()

    st.subheader(f"Results: {result['strategy_name']}")

    contribs = result.get("contributions")
    invested = result.get("invested")
    if contribs is not None and float(contribs.abs().sum()) > 0:
        final_nav = float(result["aftertax_values"].iloc[-1])
        total_in = float(invested.iloc[-1])
        cols = st.columns(3)
        cols[0].metric("Total Invested", f"R$ {total_in:,.0f}")
        cols[1].metric("Final NAV (after-tax)", f"R$ {final_nav:,.0f}")
        cols[2].metric("Profit", f"R$ {final_nav - total_in:,.0f}",
                       f"{(final_nav / total_in - 1) * 100:.1f}%")

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
                contributions=contribs,
                invested=invested,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not render equity curves: {e}")

    with tab_dd:
        try:
            # Contribution-neutral curves: deposits would otherwise refill the
            # peak and understate the drawdown (same series when buy-ins are off).
            # `or`-style .get is wrong here: the key can exist holding None.
            pt_dd = result.get("pretax_twr")
            at_dd = result.get("aftertax_twr")
            fig = plot_drawdown(
                result["pretax_values"] if pt_dd is None else pt_dd,
                result["aftertax_values"] if at_dd is None else at_dd,
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

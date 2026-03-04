"""
Fundamentals Data Pipeline Page
================================
Streamlit page for managing the CVM fundamentals pipeline (DFP/ITR/FRE).
Mirrors the structure of ui/pages/1_pipeline.py.
"""
from __future__ import annotations

import sys
import os
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

st.set_page_config(page_title="Fundamentals", page_icon="bar_chart", layout="wide")

st.title("Fundamentals Data Pipeline")
st.caption(
    "Download and process CVM financial statements (DFP/ITR/FRE) "
    "for point-in-time fundamental analysis."
)

# ── Session state ──────────────────────────────────────────────────────────────
if "job_runner" not in st.session_state:
    from ui.services.job_runner import JobRunner
    st.session_state.job_runner = JobRunner()

# ── Import services ────────────────────────────────────────────────────────────
try:
    from ui.services.fundamentals_service import get_fundamentals_stats, run_fundamentals_job
    from ui.components.log_stream import render_log_stream
    from ui.services.job_runner import JobStatus
    _SERVICES_LOADED = True
except ImportError as e:
    _SERVICES_LOADED = False
    st.warning(f"Fundamentals services not yet available: {e}")

if not _SERVICES_LOADED:
    st.info("This page will be fully functional after all task implementations are complete.")
    st.stop()


# ── Helper: render stats panel ─────────────────────────────────────────────────

def render_fundamentals_stats(stats: dict) -> None:
    """Render 3 metric cards for CVM fundamentals coverage."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Companies Mapped", f"{stats.get('total_cvm_companies', 0):,}")
    col2.metric("Filings Stored", f"{stats.get('total_cvm_filings', 0):,}")
    col3.metric("Fundamentals Rows", f"{stats.get('total_fundamentals_pit', 0):,}")


# ── Stats Panel ────────────────────────────────────────────────────────────────
st.subheader("Database Coverage")

try:
    stats = get_fundamentals_stats()
    render_fundamentals_stats(stats)
except FileNotFoundError:
    st.warning("Database not found. Run the main pipeline first.")
    stats = None
except Exception as e:
    st.error(f"Error loading fundamentals stats: {e}")
    stats = None

st.divider()

# ── Pipeline Runner ────────────────────────────────────────────────────────────
st.subheader("Pipeline Runner")

runner = st.session_state.job_runner
active_job = runner.get_active_job("fundamentals")

current_year = datetime.now().year
year_options = list(range(2010, current_year + 1))

if active_job and active_job.status == JobStatus.RUNNING:
    st.info("Fundamentals pipeline is currently running...")
    render_log_stream("fundamentals", runner)
else:
    with st.form("fundamentals_form"):
        col1, col2 = st.columns(2)
        with col1:
            start_year = st.selectbox(
                "Start Year",
                options=year_options,
                index=0,  # default 2010
                help="First year to download and parse.",
            )
            rebuild = st.checkbox(
                "Rebuild fundamentals tables (drops existing data)",
                value=False,
                help="Drop and recreate cvm_companies, cvm_filings, fundamentals_pit tables.",
            )
        with col2:
            end_year = st.selectbox(
                "End Year",
                options=year_options,
                index=len(year_options) - 1,  # default current year
                help="Last year to download and parse.",
            )
            skip_ratios = st.checkbox(
                "Skip valuation ratio computation (faster, ratios computed later)",
                value=False,
                help="Skip computing P/E, P/B, and EV/EBITDA ratios.",
            )

        submitted = st.form_submit_button("Run Fundamentals Pipeline", type="primary")

    # Form submission handling
    if submitted:
        if rebuild:
            # Store params and show confirmation on next rerun
            st.session_state["fundamentals_pending_rebuild"] = True
            st.session_state["fundamentals_start_year"] = start_year
            st.session_state["fundamentals_end_year"] = end_year
            st.session_state["fundamentals_skip_ratios"] = skip_ratios
            st.rerun()
        else:
            # Non-rebuild: run immediately
            job_id = runner.submit(
                "fundamentals",
                run_fundamentals_job,
                start_year=int(start_year),
                end_year=int(end_year),
                rebuild=False,
                skip_ratios=skip_ratios,
            )
            st.success(f"Fundamentals pipeline job started (ID: {job_id})")
            st.rerun()

    # Rebuild confirmation dialog
    if st.session_state.get("fundamentals_pending_rebuild"):
        st.warning(
            "You are about to REBUILD the fundamentals tables. "
            "All existing CVM data (companies, filings, fundamentals_pit) will be deleted. "
            "This action cannot be undone."
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("Confirm Rebuild", type="primary", key="confirm_fund_rebuild"):
                _start = st.session_state.pop("fundamentals_start_year", 2010)
                _end = st.session_state.pop("fundamentals_end_year", current_year)
                _skip = st.session_state.pop("fundamentals_skip_ratios", False)
                st.session_state.pop("fundamentals_pending_rebuild", None)
                job_id = runner.submit(
                    "fundamentals",
                    run_fundamentals_job,
                    start_year=int(_start),
                    end_year=int(_end),
                    rebuild=True,
                    skip_ratios=_skip,
                )
                st.success(f"Fundamentals pipeline job started (ID: {job_id})")
                st.rerun()
        with col_cancel:
            if st.button("Cancel", key="cancel_fund_rebuild"):
                st.session_state.pop("fundamentals_pending_rebuild", None)
                st.session_state.pop("fundamentals_start_year", None)
                st.session_state.pop("fundamentals_end_year", None)
                st.session_state.pop("fundamentals_skip_ratios", None)
                st.rerun()

# ── Show completed / failed job logs ──────────────────────────────────────────
if active_job and active_job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
    render_log_stream("fundamentals", runner)

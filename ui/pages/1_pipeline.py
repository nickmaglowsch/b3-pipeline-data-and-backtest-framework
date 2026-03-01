"""
Pipeline Management Page -- stub (implemented in task-06).
"""
from __future__ import annotations

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Pipeline", page_icon="wrench", layout="wide")

st.title("Data Pipeline Management")
st.caption("Manage your B3 market data pipeline: load, rebuild, and inspect the SQLite database.")

# ── Session state ─────────────────────────────────────────────────────────────
if "job_runner" not in st.session_state:
    from ui.services.job_runner import JobRunner
    st.session_state.job_runner = JobRunner()

# ── Import pipeline page implementation ───────────────────────────────────────
try:
    from ui.services.pipeline_service import (
        get_db_stats,
        get_raw_files,
        get_table_sample,
        run_pipeline_job,
    )
    from ui.components.log_stream import render_log_stream
    from ui.components.metrics_table import render_db_stats
    from ui.services.job_runner import JobStatus
    _SERVICES_LOADED = True
except ImportError as e:
    _SERVICES_LOADED = False
    st.warning(f"Pipeline services not yet available: {e}")

if not _SERVICES_LOADED:
    st.info("This page will be fully functional after all task implementations are complete.")
    st.stop()

# ── Database Overview ─────────────────────────────────────────────────────────
st.subheader("Database Overview")

try:
    stats = get_db_stats()
    render_db_stats(stats)
except FileNotFoundError:
    st.warning("Database not found. Run the pipeline first to create it.")
    stats = None
except Exception as e:
    st.error(f"Error loading database stats: {e}")
    stats = None

st.divider()

# ── Raw Data Files ────────────────────────────────────────────────────────────
with st.expander("Raw Data Files", expanded=False):
    try:
        files = get_raw_files()
        if files:
            files_df = pd.DataFrame(files)
            st.dataframe(files_df, use_container_width=True, hide_index=True)
        else:
            st.info("No raw COTAHIST files found in data/raw/")
    except Exception as e:
        st.error(f"Error listing raw files: {e}")

# ── Data Explorer ─────────────────────────────────────────────────────────────
with st.expander("Data Explorer", expanded=False):
    if stats is not None:
        tabs = st.tabs(["prices", "corporate_actions", "stock_actions", "detected_splits"])
        table_names = ["prices", "corporate_actions", "stock_actions", "detected_splits"]
        for tab, tname in zip(tabs, table_names):
            with tab:
                try:
                    df = get_table_sample(tname, limit=100)
                    if df is not None and not df.empty:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info(f"No data in {tname}")
                except Exception as e:
                    st.info(f"Table {tname} not accessible: {e}")
    else:
        st.info("Connect to a database first.")

st.divider()

# ── Pipeline Runner ───────────────────────────────────────────────────────────
st.subheader("Pipeline Runner")

runner = st.session_state.job_runner
active_job = runner.get_active_job("pipeline")

if active_job and active_job.status == JobStatus.RUNNING:
    st.info("Pipeline is currently running...")
    render_log_stream("pipeline", runner)
else:
    with st.form("pipeline_form"):
        col1, col2 = st.columns(2)
        with col1:
            rebuild = st.checkbox(
                "Rebuild database (destructive -- drops all tables!)",
                value=False,
                help="Drop and recreate all tables. All existing data will be lost.",
            )
            skip_corporate = st.checkbox(
                "Skip corporate actions",
                value=False,
                help="Skip processing corporate action events.",
            )
        with col2:
            year_options = ["All"] + [str(y) for y in range(2010, 2027)]
            year_choice = st.selectbox(
                "Year",
                options=year_options,
                index=0,
                help="Process a specific year, or All years.",
            )

        submitted = st.form_submit_button("Run Pipeline", type="primary")

    # Handle form submission: either run directly or show confirmation for rebuild
    if submitted:
        if rebuild:
            # Store params and show confirmation dialog on next rerun
            st.session_state["pipeline_pending_rebuild"] = True
            st.session_state["pipeline_year_choice"] = year_choice
            st.session_state["pipeline_skip_corporate"] = skip_corporate
            st.rerun()
        else:
            # Non-rebuild: run immediately
            year = None if year_choice == "All" else int(year_choice)
            job_id = runner.submit(
                "pipeline",
                run_pipeline_job,
                rebuild=False,
                year=year,
                skip_corporate_actions=skip_corporate,
            )
            st.success(f"Pipeline job started (ID: {job_id})")
            st.rerun()

    # Handle pending rebuild confirmation dialog
    if st.session_state.get("pipeline_pending_rebuild"):
        st.warning(
            "You are about to REBUILD the database. This will delete all existing data. "
            "This action cannot be undone.",
        )
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("Confirm Rebuild", type="primary", key="confirm_rebuild"):
                year_choice = st.session_state.pop("pipeline_year_choice", "All")
                skip_corporate = st.session_state.pop("pipeline_skip_corporate", False)
                st.session_state.pop("pipeline_pending_rebuild", None)
                year = None if year_choice == "All" else int(year_choice)
                job_id = runner.submit(
                    "pipeline",
                    run_pipeline_job,
                    rebuild=True,
                    year=year,
                    skip_corporate_actions=skip_corporate,
                )
                st.success(f"Pipeline job started (ID: {job_id})")
                st.rerun()
        with col_cancel:
            if st.button("Cancel", key="cancel_rebuild"):
                st.session_state.pop("pipeline_pending_rebuild", None)
                st.session_state.pop("pipeline_year_choice", None)
                st.session_state.pop("pipeline_skip_corporate", None)
                st.rerun()

# ── Show completed job logs ───────────────────────────────────────────────────
if active_job and active_job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
    render_log_stream("pipeline", runner)

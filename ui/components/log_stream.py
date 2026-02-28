"""
Log Stream Component
====================
Streamlit component for displaying real-time streaming logs from a background job.
Uses st.fragment for auto-refresh without blocking the entire page.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from ui.services.job_runner import JobRunner, JobStatus


@st.fragment(run_every=1.0)
def _log_poll_fragment(job_type: str, runner: JobRunner) -> None:
    """Auto-refreshing fragment that polls job logs every second."""
    job = runner.get_active_job(job_type)
    if not job:
        return

    log_key = f"log_{job.id}"
    if log_key not in st.session_state:
        st.session_state[log_key] = []

    # Drain new lines
    new_lines = runner.drain_logs(job.id)
    if new_lines:
        st.session_state[log_key].extend(new_lines)

    # Determine status display
    if job.status == JobStatus.RUNNING:
        label = f"Running {job_type}..."
        state_icon = "running"
    elif job.status == JobStatus.COMPLETED:
        label = f"{job_type.capitalize()} completed successfully"
        state_icon = "complete"
    elif job.status == JobStatus.FAILED:
        label = f"{job_type.capitalize()} FAILED"
        state_icon = "error"
    else:
        label = f"{job_type.capitalize()} {job.status.value}"
        state_icon = "running"

    with st.status(label, expanded=True, state=state_icon):
        all_logs = "\n".join(st.session_state[log_key][-500:])
        st.code(all_logs or "(no output yet)", language="text")

    if job.started_at and job.completed_at:
        elapsed = (job.completed_at - job.started_at).total_seconds()
        st.caption(f"Duration: {elapsed:.1f}s")

    if job.status == JobStatus.FAILED and job.error:
        with st.expander("Error details", expanded=True):
            st.code(job.error, language="python")


def render_log_stream(job_type: str, runner: JobRunner) -> Any:
    """
    Display a live log stream for the most recent job of the given type.

    Uses st.fragment with run_every=1.0 for non-blocking auto-refresh.

    Args:
        job_type: The job type tag (e.g. 'pipeline', 'backtest', 'research').
        runner:   The JobRunner instance from st.session_state.

    Returns:
        The job result when complete, or None.
    """
    job = runner.get_active_job(job_type)
    if not job:
        return None

    _log_poll_fragment(job_type, runner)

    return job.result if job.status == JobStatus.COMPLETED else None

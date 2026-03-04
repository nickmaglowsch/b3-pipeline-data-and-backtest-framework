"""
Fundamentals Service
====================
Thin wrapper around b3_pipeline CVM modules for the Streamlit fundamentals page.
Provides cached stats access and a job-runner-compatible pipeline function.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"


@st.cache_data(ttl=60)
def get_fundamentals_stats() -> dict:
    """
    Return CVM fundamentals statistics from the database.

    Returns a dict with keys: total_cvm_companies, total_cvm_filings, total_fundamentals_pit.
    Raises FileNotFoundError if the database does not exist.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        from b3_pipeline.cvm_storage import get_fundamentals_stats as _get_stats
        stats = _get_stats(conn)
    except Exception:
        # Fallback: basic counts if import fails
        cur = conn.cursor()
        stats = {}
        for key, table in [
            ("total_cvm_companies", "cvm_companies"),
            ("total_cvm_filings", "cvm_filings"),
            ("total_fundamentals_pit", "fundamentals_pit"),
        ]:
            try:
                cur.execute(f"SELECT COUNT(*) FROM {table}")
                stats[key] = cur.fetchone()[0]
            except Exception:
                stats[key] = 0
    finally:
        conn.close()

    return stats


def run_fundamentals_job(
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    rebuild: bool = False,
    skip_ratios: bool = False,
) -> None:
    """
    Run the CVM fundamentals pipeline. Designed to be called inside a JobRunner thread.
    Clears the get_fundamentals_stats cache after completion.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from b3_pipeline.cvm_main import run_fundamentals_pipeline
    run_fundamentals_pipeline(
        start_year=start_year,
        end_year=end_year,
        rebuild=rebuild,
        skip_ratios=skip_ratios,
    )
    # Clear Streamlit cache so stats panel refreshes on next page load
    get_fundamentals_stats.clear()

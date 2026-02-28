"""
Pipeline Service
================
Thin wrapper around b3_pipeline for the UI pipeline management page.
Provides read-only database access for stats/browsing and a function
to run the pipeline job inside a JobRunner thread.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"


@st.cache_data(ttl=60)
def get_db_stats() -> dict:
    """
    Return database summary statistics.
    Raises FileNotFoundError if the database does not exist.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found at {DB_PATH}")

    conn = sqlite3.connect(str(DB_PATH))
    try:
        from b3_pipeline.storage import get_summary_stats
        stats = get_summary_stats(conn)
    except Exception:
        # Fallback: build stats manually if import fails or function raises
        stats = _manual_stats(conn)
    finally:
        conn.close()

    return stats


def _manual_stats(conn: sqlite3.Connection) -> dict:
    """Compute basic stats when b3_pipeline.storage is not importable."""
    cur = conn.cursor()
    stats: dict = {}

    # Total prices
    try:
        cur.execute("SELECT COUNT(*) FROM prices")
        stats["total_prices"] = cur.fetchone()[0]
    except Exception:
        stats["total_prices"] = 0

    # Unique tickers
    try:
        cur.execute("SELECT COUNT(DISTINCT ticker) FROM prices")
        stats["total_tickers"] = cur.fetchone()[0]
    except Exception:
        stats["total_tickers"] = 0

    # ISINs
    try:
        cur.execute("SELECT COUNT(DISTINCT isin) FROM prices")
        stats["total_isins"] = cur.fetchone()[0]
    except Exception:
        stats["total_isins"] = 0

    # Date range
    try:
        cur.execute("SELECT MIN(date), MAX(date) FROM prices")
        row = cur.fetchone()
        stats["date_range"] = (row[0] or "N/A", row[1] or "N/A")
    except Exception:
        stats["date_range"] = ("N/A", "N/A")

    # Corporate / stock actions
    for key, table in [
        ("total_corporate_actions", "corporate_actions"),
        ("total_stock_actions", "stock_actions"),
        ("total_detected_splits", "detected_splits"),
    ]:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[key] = cur.fetchone()[0]
        except Exception:
            stats[key] = 0

    return stats


def get_raw_files() -> list[dict]:
    """Return a list of dicts describing COTAHIST ZIP files in data/raw/."""
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if not raw_dir.exists():
        return []

    files = []
    for f in sorted(raw_dir.glob("*.ZIP")) + sorted(raw_dir.glob("*.zip")):
        stat = f.stat()
        files.append({
            "filename": f.name,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified": pd.Timestamp(stat.st_mtime, unit="s").strftime("%Y-%m-%d %H:%M"),
        })
    return files


_ALLOWED_TABLES = {"prices", "corporate_actions", "stock_actions", "detected_splits"}


def get_table_sample(table_name: str, limit: int = 100) -> Optional[pd.DataFrame]:
    """
    Return the first `limit` rows of a database table.
    Uses a read-only SQLite connection. Only allows whitelisted table names.
    """
    if table_name not in _ALLOWED_TABLES:
        return None
    if not DB_PATH.exists():
        return None
    conn = None
    try:
        conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT {int(limit)}", conn)
        return df
    except Exception:
        return None
    finally:
        if conn is not None:
            conn.close()


def run_pipeline_job(rebuild: bool = False, year: Optional[int] = None, skip_corporate_actions: bool = False) -> None:
    """
    Run the B3 data pipeline. Designed to be called inside a JobRunner thread.
    Imports run_pipeline from b3_pipeline.main.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from b3_pipeline.main import run_pipeline
    run_pipeline(
        rebuild=rebuild,
        year=year,
        skip_corporate_actions=skip_corporate_actions,
    )
    # Clear Streamlit cache after pipeline run
    get_db_stats.clear()

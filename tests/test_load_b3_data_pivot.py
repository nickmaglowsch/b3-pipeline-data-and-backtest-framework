"""
Unit tests for the pivot+ffill logic inside load_b3_data in backtests/core/data.py.

Safety-net tests for the Rust rewrite in task-03. These tests must pass
before and after the Rust implementation is introduced.
"""
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from b3_pipeline import storage
from backtests.core.data import load_b3_data


# ── Fixture helpers (pattern from test_survivorship_filter.py) ─────────────────

def _setup_db(tmp_path: Path) -> str:
    """Create a temp DB with full schema. Returns db_path string."""
    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    conn.commit()
    conn.close()
    return db_path


def _insert_price(conn, ticker, date_str, close=10.0, adj_close=None, volume=100_000):
    if adj_close is None:
        adj_close = close
    conn.execute(
        """INSERT INTO prices (ticker, isin_code, date, open, high, low, close,
                               adj_close, split_adj_high, split_adj_low, split_adj_close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker, "UNKNOWN", date_str, close, close, close, close,
         adj_close, close, close, close, volume)
    )
    conn.commit()


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLoadB3DataPivot:

    def test_returns_three_dataframes(self, tmp_path):
        """load_b3_data returns a 3-tuple of DataFrames."""
        db_path = _setup_db(tmp_path)
        conn = sqlite3.connect(db_path)
        for ticker in ["PETR3", "VALE3"]:
            for date_str in ["2020-01-02", "2020-01-03", "2020-01-06"]:
                _insert_price(conn, ticker, date_str)
        conn.close()

        result = load_b3_data(db_path, "2020-01-01", "2020-01-31")
        assert isinstance(result, tuple)
        assert len(result) == 3
        adj_close, close_px, fin_vol = result
        assert isinstance(adj_close, pd.DataFrame)
        assert isinstance(close_px, pd.DataFrame)
        assert isinstance(fin_vol, pd.DataFrame)

    def test_wide_shape_dates_are_index_tickers_are_columns(self, tmp_path):
        """Output DataFrames have dates as index and tickers as columns."""
        db_path = _setup_db(tmp_path)
        conn = sqlite3.connect(db_path)
        for ticker in ["PETR3", "VALE3"]:
            for date_str in ["2020-01-02", "2020-01-03"]:
                _insert_price(conn, ticker, date_str)
        conn.close()

        adj_close, close_px, fin_vol = load_b3_data(db_path, "2020-01-01", "2020-01-31")

        # Index should contain the two dates as datetime64[ns]
        assert pd.Timestamp("2020-01-02") in adj_close.index
        assert pd.Timestamp("2020-01-03") in adj_close.index
        assert adj_close.index.dtype == "datetime64[ns]"

        # Columns should contain both tickers
        assert "PETR3" in adj_close.columns
        assert "VALE3" in adj_close.columns

    def test_fin_vol_scaled_by_100(self, tmp_path):
        """fin_vol values equal raw volume / 100.0."""
        db_path = _setup_db(tmp_path)
        conn = sqlite3.connect(db_path)
        _insert_price(conn, "PETR3", "2020-01-02", volume=100_000)
        conn.close()

        adj_close, close_px, fin_vol = load_b3_data(db_path, "2020-01-01", "2020-01-31")

        assert "PETR3" in fin_vol.columns
        val = fin_vol.loc[pd.Timestamp("2020-01-02"), "PETR3"]
        assert abs(val - 1000.0) < 1e-9  # 100_000 / 100.0

    def test_forward_fill_closes_gaps(self, tmp_path):
        """adj_close and close_px are forward-filled across date gaps."""
        db_path = _setup_db(tmp_path)
        conn = sqlite3.connect(db_path)
        # PETR3 only has 2020-01-02; VALE3 has both dates so both appear in index
        _insert_price(conn, "PETR3", "2020-01-02", close=100.0, adj_close=100.0)
        _insert_price(conn, "VALE3", "2020-01-02", close=200.0, adj_close=200.0)
        _insert_price(conn, "VALE3", "2020-01-03", close=201.0, adj_close=201.0)
        conn.close()

        adj_close, close_px, fin_vol = load_b3_data(db_path, "2020-01-01", "2020-01-31")

        # PETR3 has no row on 2020-01-03 — should be filled from 2020-01-02
        assert abs(adj_close.loc[pd.Timestamp("2020-01-03"), "PETR3"] - 100.0) < 1e-9
        assert abs(close_px.loc[pd.Timestamp("2020-01-03"), "PETR3"] - 100.0) < 1e-9

    def test_fin_vol_not_forward_filled(self, tmp_path):
        """fin_vol is NOT forward-filled; gaps remain NaN."""
        db_path = _setup_db(tmp_path)
        conn = sqlite3.connect(db_path)
        # PETR3 only on 2020-01-02; VALE3 on both dates so 2020-01-03 appears
        _insert_price(conn, "PETR3", "2020-01-02", volume=100_000)
        _insert_price(conn, "VALE3", "2020-01-02", volume=200_000)
        _insert_price(conn, "VALE3", "2020-01-03", volume=200_000)
        conn.close()

        adj_close, close_px, fin_vol = load_b3_data(db_path, "2020-01-01", "2020-01-31")

        val = fin_vol.loc[pd.Timestamp("2020-01-03"), "PETR3"]
        assert pd.isna(val), f"Expected NaN, got {val}"

    def test_non_standard_tickers_excluded(self, tmp_path):
        """Non-standard tickers (BDRs like PETR34, unknown suffix) are excluded."""
        db_path = _setup_db(tmp_path)
        conn = sqlite3.connect(db_path)
        _insert_price(conn, "PETR3", "2020-01-02")   # standard — 5 chars, suffix 3
        _insert_price(conn, "PETR34", "2020-01-02")  # BDR — 6 chars, suffix not "11"
        _insert_price(conn, "PETRX", "2020-01-02")   # 5 chars, suffix X — non-standard
        conn.close()

        adj_close, close_px, fin_vol = load_b3_data(db_path, "2020-01-01", "2020-01-31")

        assert "PETR3" in adj_close.columns
        assert "PETR34" not in adj_close.columns
        assert "PETRX" not in adj_close.columns

    def test_ticker_filter_accepts_suffix_11(self, tmp_path):
        """Tickers like BOVA11 (6 chars, suffix 11) are included."""
        db_path = _setup_db(tmp_path)
        conn = sqlite3.connect(db_path)
        _insert_price(conn, "BOVA11", "2020-01-02")
        conn.close()

        adj_close, close_px, fin_vol = load_b3_data(db_path, "2020-01-01", "2020-01-31")

        assert "BOVA11" in adj_close.columns

    def test_empty_db_returns_empty_dataframes(self, tmp_path):
        """With no price rows in the date range, all three DataFrames are empty."""
        db_path = _setup_db(tmp_path)
        # No rows inserted

        adj_close, close_px, fin_vol = load_b3_data(db_path, "2020-01-01", "2020-01-31")

        assert adj_close.empty
        assert close_px.empty
        assert fin_vol.empty

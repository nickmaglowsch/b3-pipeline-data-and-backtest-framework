"""
Tests for survivorship bias filter in backtests (Task 07 — TDD).

All tests use in-memory SQLite DBs or temp files — no real price data required.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from b3_pipeline import storage


# ── Helpers ────────────────────────────────────────────────────────────────────

def _setup_db(tmp_path: Path) -> str:
    """Create a temp DB with full schema. Returns db_path string."""
    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    conn.commit()
    conn.close()
    return db_path


def _insert_company(conn, cnpj, ticker, delisting_date=None, listing_date=None):
    conn.execute(
        """INSERT INTO cvm_companies (cnpj, ticker, listing_date, delisting_date)
           VALUES (?, ?, ?, ?)""",
        (cnpj, ticker, listing_date, delisting_date)
    )
    conn.commit()


def _insert_price(conn, ticker, date, close=10.0):
    conn.execute(
        """INSERT INTO prices (ticker, isin_code, date, open, high, low, close,
                               adj_close, split_adj_high, split_adj_low, split_adj_close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker, "UNKNOWN", date, close, close, close, close, close, close, close, close, 100000)
    )
    conn.commit()


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_load_active_tickers_returns_only_active(tmp_path):
    """Active tickers are returned; delisted-before-as_of_date tickers are excluded."""
    from backtests.core.data import load_active_tickers

    db_path = _setup_db(tmp_path)
    conn = sqlite3.connect(db_path)
    _insert_company(conn, "11111111000100", "PETR", delisting_date=None)
    _insert_company(conn, "22222222000100", "VALE", delisting_date="2005-06-30")
    conn.close()

    result = load_active_tickers(db_path, "2010-01-01")
    assert "PETR" in result
    assert "VALE" not in result


def test_load_active_tickers_keeps_company_listed_on_query_date(tmp_path):
    """Company with delisting_date == as_of_date is included (boundary: >= not >)."""
    from backtests.core.data import load_active_tickers

    db_path = _setup_db(tmp_path)
    conn = sqlite3.connect(db_path)
    _insert_company(conn, "11111111000100", "BBAS", delisting_date="2010-01-01")
    conn.close()

    result = load_active_tickers(db_path, "2010-01-01")
    assert "BBAS" in result


def test_load_active_tickers_keeps_unknown_roots(tmp_path):
    """Only cvm_companies rows are returned; unknown tickers not in cvm_companies are absent."""
    from backtests.core.data import load_active_tickers

    db_path = _setup_db(tmp_path)
    # Insert one known company
    conn = sqlite3.connect(db_path)
    _insert_company(conn, "11111111000100", "PETR")
    conn.close()

    result = load_active_tickers(db_path, "2010-01-01")
    # Only PETR is in cvm_companies; MGLU (unknown) is NOT returned by the function
    assert "PETR" in result
    assert "MGLU" not in result  # not in cvm_companies — caller handles conservatively


def test_load_active_tickers_graceful_degradation(tmp_path):
    """Returns empty set (no crash) if cvm_companies lacks delisting_date column."""
    from backtests.core.data import load_active_tickers

    db_path = _setup_db(tmp_path)
    # Drop the delisting_date column to simulate un-migrated DB
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE cvm_companies_old AS
        SELECT cnpj, ticker, company_name, cvm_code, b3_trading_name, updated_at
        FROM cvm_companies
    """)
    conn.execute("DROP TABLE cvm_companies")
    conn.execute("ALTER TABLE cvm_companies_old RENAME TO cvm_companies")
    conn.commit()
    conn.close()

    result = load_active_tickers(db_path, "2010-01-01")
    assert isinstance(result, set)
    # Should return empty set (graceful degradation) or any valid set without crashing


def test_load_b3_data_filter_delisted_false_returns_all_tickers(tmp_path):
    """With filter_delisted=False (default), behavior is unchanged."""
    from backtests.core.data import load_b3_data

    db_path = _setup_db(tmp_path)
    conn = sqlite3.connect(db_path)
    _insert_price(conn, "PETR3", "2010-01-04")
    _insert_price(conn, "VALE3", "2010-01-04")
    _insert_company(conn, "11111111000100", "PETR", delisting_date="2005-01-01")  # delisted
    conn.close()

    adj_close, close_px, fin_vol = load_b3_data(db_path, "2010-01-01", "2010-01-31", filter_delisted=False)
    # Both tickers should be present (filter off)
    assert "PETR3" in adj_close.columns
    assert "VALE3" in adj_close.columns


def test_load_b3_data_filter_delisted_drops_known_delisted_ticker(tmp_path):
    """With filter_delisted=True, tickers whose root delisted before start are dropped."""
    from backtests.core.data import load_b3_data

    db_path = _setup_db(tmp_path)
    conn = sqlite3.connect(db_path)
    _insert_price(conn, "PETR3", "2010-01-04")
    _insert_price(conn, "VALE3", "2010-01-04")
    # VALE delisted in 2005 — should be dropped when querying from 2010
    _insert_company(conn, "11111111000100", "PETR", delisting_date=None)
    _insert_company(conn, "22222222000100", "VALE", delisting_date="2005-01-01")
    conn.close()

    adj_close, close_px, fin_vol = load_b3_data(db_path, "2010-01-01", "2010-01-31", filter_delisted=True)
    # VALE3 should be dropped (VALE delisted before start)
    assert "VALE3" not in adj_close.columns
    # PETR3 should be kept (PETR still active)
    assert "PETR3" in adj_close.columns


def test_load_b3_data_filter_delisted_keeps_ticker_not_in_cvm_companies(tmp_path):
    """Tickers not in cvm_companies are kept (conservative — cannot determine if delisted)."""
    from backtests.core.data import load_b3_data

    db_path = _setup_db(tmp_path)
    conn = sqlite3.connect(db_path)
    _insert_price(conn, "MGLU3", "2010-01-04")
    # MGLU not in cvm_companies at all
    conn.close()

    adj_close, close_px, fin_vol = load_b3_data(db_path, "2010-01-01", "2010-01-31", filter_delisted=True)
    # MGLU3 should be kept (conservative: unknown root = keep)
    assert "MGLU3" in adj_close.columns

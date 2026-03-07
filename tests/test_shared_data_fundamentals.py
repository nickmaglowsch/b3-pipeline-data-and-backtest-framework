"""
Tests for build_shared_data() monthly fundamentals integration (Task 06 TDD).

All tests use a temporary SQLite DB with synthetic data.
Network calls (download_cdi_daily, download_benchmark) are mocked.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from b3_pipeline import storage


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_constant_series(value: float, start: str, end: str, freq: str = "D") -> pd.Series:
    idx = pd.date_range(start, end, freq=freq)
    return pd.Series(value, index=idx)


def _insert_price(conn, ticker, date, close, volume=10_000):
    conn.execute(
        """
        INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close,
                                       volume, split_adj_high, split_adj_low, split_adj_close, adj_close)
        VALUES (?, 'UNKNOWN', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, close, close, close, close, int(volume * 100),
         close, close, close, close),
    )


@pytest.fixture
def tmp_db(tmp_path):
    """File-based SQLite DB with a minimal set of prices and fundamentals_monthly rows."""
    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    storage.init_db(conn)

    # Insert monthly prices for two tickers
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="ME")
    for dt in dates:
        date_str = dt.strftime("%Y-%m-%d")
        _insert_price(conn, "PETR3", date_str, 30.0)
        _insert_price(conn, "VALE3", date_str, 50.0)

    # Insert fundamentals_pit rows so load_all_fundamentals() works
    conn.execute("""
        INSERT OR REPLACE INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type,
           revenue, pe_ratio, pb_ratio, ev_ebitda)
        VALUES ('f1', '33000167000101', 'PETR3', '2022-12-31', '2023-01-01', 1, 'DFP',
                1000.0, 15.0, 2.0, 8.0)
    """)

    # Insert fundamentals_monthly rows
    monthly_rows = [
        ("2023-01-31", "PETR3", 15.0, 10.0, 2.0, 1000.0, 100.0),
        ("2023-02-28", "PETR3", 14.0, 10.0, 2.0, 1000.0, 100.0),
        ("2023-01-31", "VALE3", 8.0, 20.0, 1.5, 2000.0, 200.0),
    ]
    for me, tkr, pe, ni, pb, rev, eq in monthly_rows:
        conn.execute(
            """INSERT OR REPLACE INTO fundamentals_monthly
               (month_end, ticker, pe_ratio, net_income, pb_ratio, revenue, equity)
               VALUES (?,?,?,?,?,?,?)""",
            (me, tkr, pe, ni, pb, rev, eq),
        )

    conn.commit()
    conn.close()
    yield db_path


def _mock_cdi(start, end):
    return _make_constant_series(0.0001, start, end, freq="D")


def _mock_benchmark(ticker, start, end):
    s = _make_constant_series(10000.0, start, end, freq="D")
    s.index = s.index.tz_localize(None)
    return s


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("backtests.core.shared_data.download_cdi_daily", side_effect=_mock_cdi)
@patch("backtests.core.shared_data.download_benchmark", side_effect=_mock_benchmark)
def test_build_shared_data_adds_monthly_keys(mock_bench, mock_cdi, tmp_db):
    """build_shared_data with include_fundamentals=True should add _m and _dyn keys."""
    from backtests.core.shared_data import build_shared_data

    shared = build_shared_data(
        tmp_db, "2023-01-01", "2023-06-30", freq="ME",
        include_fundamentals=True,
    )

    assert "f_pe_ratio_dyn" in shared, "f_pe_ratio_dyn key should be present"
    assert "f_net_income_m" in shared, "f_net_income_m key should be present"
    assert "f_equity_m" in shared, "f_equity_m key should be present"
    assert "f_shares_outstanding_m" in shared, "f_shares_outstanding_m key should be present"


@patch("backtests.core.shared_data.download_cdi_daily", side_effect=_mock_cdi)
@patch("backtests.core.shared_data.download_benchmark", side_effect=_mock_benchmark)
def test_build_shared_data_monthly_keys_empty_when_table_empty(mock_bench, mock_cdi, tmp_db):
    """When fundamentals_monthly is empty, _m/_dyn keys should be empty DataFrames."""
    from backtests.core.shared_data import build_shared_data
    import sqlite3 as sq

    # Empty the table
    conn = sq.connect(tmp_db)
    conn.execute("DELETE FROM fundamentals_monthly")
    conn.commit()
    conn.close()

    shared = build_shared_data(
        tmp_db, "2023-01-01", "2023-06-30", freq="ME",
        include_fundamentals=True,
    )

    assert "f_pe_ratio_dyn" in shared, "Key should still exist even when empty"
    assert isinstance(shared["f_pe_ratio_dyn"], pd.DataFrame), "Should be DataFrame"
    # Should be empty or have no data
    assert shared["f_pe_ratio_dyn"].empty or shared["f_pe_ratio_dyn"].isna().all().all()


@patch("backtests.core.shared_data.download_cdi_daily", side_effect=_mock_cdi)
@patch("backtests.core.shared_data.download_benchmark", side_effect=_mock_benchmark)
def test_build_shared_data_original_keys_unaffected(mock_bench, mock_cdi, tmp_db):
    """Original f_* keys from load_all_fundamentals() should still be present."""
    from backtests.core.shared_data import build_shared_data

    shared = build_shared_data(
        tmp_db, "2023-01-01", "2023-06-30", freq="ME",
        include_fundamentals=True,
    )

    # These must still be present (backward compat for ValueQuality)
    assert "f_pe_ratio" in shared, "Original f_pe_ratio key should still be present"
    assert "f_revenue" in shared, "Original f_revenue key should still be present"
    # Both old and new keys together
    assert "f_pe_ratio_dyn" in shared


@patch("backtests.core.shared_data.download_cdi_daily", side_effect=_mock_cdi)
@patch("backtests.core.shared_data.download_benchmark", side_effect=_mock_benchmark)
def test_build_shared_data_monthly_disabled_by_default(mock_bench, mock_cdi, tmp_db):
    """Default include_fundamentals=False should produce no f_* keys."""
    from backtests.core.shared_data import build_shared_data

    shared = build_shared_data(
        tmp_db, "2023-01-01", "2023-06-30", freq="ME",
    )

    f_keys = [k for k in shared.keys() if k.startswith("f_")]
    assert len(f_keys) == 0, f"Expected no f_* keys when include_fundamentals=False, got: {f_keys}"

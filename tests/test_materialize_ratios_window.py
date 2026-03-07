"""
Tests for the ±5 trading-day price window and highest-ADTV ticker selection
in materialize_valuation_ratios() (Task 02 TDD).

All tests use in-memory SQLite with synthetic data. No network calls.
"""
from __future__ import annotations

import sqlite3

import pytest

from b3_pipeline import storage, cvm_storage
from b3_pipeline.cvm_main import materialize_valuation_ratios


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_conn():
    """In-memory SQLite connection with full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    yield conn
    conn.close()


def _insert_price(conn, ticker, date, close, volume=100_000):
    """Helper: insert a minimal prices row (volume in financial volume units *100)."""
    conn.execute(
        """
        INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close, volume)
        VALUES (?, 'UNKNOWN', ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, close, close, close, close, int(volume * 100)),
    )
    conn.commit()


def _insert_company(conn, cnpj, ticker):
    """Helper: insert a cvm_companies row."""
    conn.execute(
        "INSERT OR REPLACE INTO cvm_companies (cnpj, ticker) VALUES (?, ?)",
        (cnpj, ticker),
    )
    conn.commit()


def _insert_pit_row(conn, **kwargs):
    """Helper: insert a fundamentals_pit row with defaults for unspecified columns."""
    defaults = {
        "filing_id": "TEST_DFP_2023-12-31_1",
        "cnpj": "33000167000101",
        "ticker": "PETR",
        "period_end": "2023-12-31",
        "filing_date": "2023-04-15",
        "filing_version": 1,
        "doc_type": "DFP",
        "fiscal_year": 2023,
        "quarter": None,
        "revenue": None,
        "net_income": 1_000.0,
        "ebitda": None,
        "total_assets": None,
        "equity": None,
        "net_debt": None,
        "shares_outstanding": 10_000_000.0,
        "pe_ratio": None,
        "pb_ratio": None,
        "ev_ebitda": None,
    }
    defaults.update(kwargs)
    conn.execute(
        """
        INSERT OR REPLACE INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version,
           doc_type, fiscal_year, quarter,
           revenue, net_income, ebitda, total_assets, equity, net_debt,
           shares_outstanding, pe_ratio, pb_ratio, ev_ebitda)
        VALUES
          (:filing_id, :cnpj, :ticker, :period_end, :filing_date, :filing_version,
           :doc_type, :fiscal_year, :quarter,
           :revenue, :net_income, :ebitda, :total_assets, :equity, :net_debt,
           :shares_outstanding, :pe_ratio, :pb_ratio, :ev_ebitda)
        """,
        defaults,
    )
    conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Window tests
# ──────────────────────────────────────────────────────────────────────────────

def test_window_finds_price_on_next_trading_day(mem_conn):
    """Filing on Saturday should find a price from the following Monday (+2 days)."""
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-04-15",  # Saturday
    )
    # Only price available: Monday April 17 (+2 days)
    _insert_price(mem_conn, "PETR3", "2023-04-17", 30.0)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    assert pe is not None, "Expected pe_ratio to be non-NULL (price found via forward window)"


def test_window_finds_price_on_prior_trading_day(mem_conn):
    """Filing on Saturday should find a price from the prior Friday (-1 day)."""
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-04-15",  # Saturday
    )
    # Only price available: Friday April 14 (-1 day)
    _insert_price(mem_conn, "PETR3", "2023-04-14", 25.0)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    assert pe is not None, "Expected pe_ratio to be non-NULL (price found via backward window)"
    # PE = (25 * 10M) / (1000 * 1000) = 250
    assert pe == pytest.approx(250.0, rel=1e-3)


def test_window_prefers_closest_date(mem_conn):
    """When two prices are available, picks the one closest to filing_date."""
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-04-15",  # Saturday
    )
    # 3 days before: price=20; 2 days after: price=30 (closer)
    _insert_price(mem_conn, "PETR3", "2023-04-12", 20.0)
    _insert_price(mem_conn, "PETR3", "2023-04-17", 30.0)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    # Should use price=30 (2 days away is closer than 3 days)
    expected = (30.0 * 10_000_000) / (1_000.0 * 1_000)  # 300.0
    assert pe == pytest.approx(expected, rel=1e-3), f"Expected pe_ratio using price=30 ({expected}), got {pe}"


def test_window_ties_prefer_earlier_date(mem_conn):
    """When two prices are equidistant, pick the earlier (backward) date."""
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-04-15",  # Saturday
    )
    # Both 1 day away: Friday (20) and Sunday... use Monday for +1 workday
    _insert_price(mem_conn, "PETR3", "2023-04-14", 20.0)  # -1 day
    _insert_price(mem_conn, "PETR3", "2023-04-16", 30.0)  # +1 day

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    # Tie broken by preferring earlier date: price=20
    expected = (20.0 * 10_000_000) / (1_000.0 * 1_000)  # 200.0
    assert pe == pytest.approx(expected, rel=1e-3), f"Expected pe_ratio using price=20 ({expected}), got {pe}"


def test_window_no_price_within_window(mem_conn):
    """No price within ±10 calendar days means pe_ratio stays NULL."""
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-04-15",
    )
    # Price far outside window
    _insert_price(mem_conn, "PETR3", "2023-01-01", 30.0)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    assert pe is None, f"Expected pe_ratio NULL when no price in ±10 days, got {pe}"


# ──────────────────────────────────────────────────────────────────────────────
# ADTV picker tests
# ──────────────────────────────────────────────────────────────────────────────

def test_adtv_picks_highest_volume_ticker(mem_conn):
    """When two tickers exist for same company, highest-ADTV ticker is used for ratio."""
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-04-15",
    )
    # PETR3: low volume, price=20; PETR4: high volume, price=40
    _insert_price(mem_conn, "PETR3", "2023-04-15", 20.0, volume=100)      # low ADTV
    _insert_price(mem_conn, "PETR4", "2023-04-15", 40.0, volume=10_000)   # high ADTV

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    assert pe is not None, "Expected pe_ratio to be computed"
    # Should use PETR4 price (40), not PETR3 (20)
    expected_using_petr4 = (40.0 * 10_000_000) / (1_000.0 * 1_000)  # 400.0
    assert pe == pytest.approx(expected_using_petr4, rel=1e-3), (
        f"Expected pe_ratio using PETR4 price=40 ({expected_using_petr4}), got {pe}"
    )


def test_adtv_uses_standard_lot_tickers_only(mem_conn):
    """Non-standard lot tickers (e.g. length-6 not ending in '11') should be excluded."""
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-04-15",
    )
    # PETR31 is 6 chars but doesn't end in '11' — should be excluded
    _insert_price(mem_conn, "PETR31", "2023-04-15", 100.0, volume=999_999)  # very high but non-standard
    # PETR3 is standard lot, lower volume
    _insert_price(mem_conn, "PETR3", "2023-04-15", 30.0, volume=1_000)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    assert pe is not None, "Expected pe_ratio to be computed using standard-lot PETR3"
    # Should use PETR3 price=30 (PETR31 excluded by filter)
    expected_using_petr3 = (30.0 * 10_000_000) / (1_000.0 * 1_000)  # 300.0
    assert pe == pytest.approx(expected_using_petr3, rel=1e-3), (
        f"Expected pe_ratio using PETR3 price=30 ({expected_using_petr3}), got {pe}"
    )

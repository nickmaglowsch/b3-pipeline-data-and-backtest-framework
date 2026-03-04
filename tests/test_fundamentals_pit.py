"""
Tests for CVM pipeline orchestration and valuation ratio materialization (Task 05 — TDD).
Also contains the Task 06 acceptance test: test_load_fundamentals_pit_forward_fills.

All tests use in-memory SQLite (or tmp_path file DB) with synthetic data.
No network calls or real CVM files are used.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from b3_pipeline import storage, cvm_storage
from b3_pipeline.cvm_main import materialize_valuation_ratios


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_conn():
    """In-memory SQLite connection with full schema including fundamentals tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    yield conn
    conn.close()


def _insert_price(conn, ticker, date, close):
    """Helper: insert a minimal prices row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close, volume)
        VALUES (?, 'UNKNOWN', ?, ?, ?, ?, ?, 0)
        """,
        (ticker, date, close, close, close, close),
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
        "net_income": None,
        "ebitda": None,
        "total_assets": None,
        "equity": None,
        "net_debt": None,
        "shares_outstanding": None,
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
# 1. Basic P/E ratio
# ──────────────────────────────────────────────────────────────────────────────

def test_materialize_pe_ratio_basic(mem_conn):
    """P/E = (price * shares) / (net_income * 1000) — CVM values are in thousands BRL."""
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        ticker="PETR",
        net_income=1_000.0,  # 1,000 thousands BRL = 1,000,000 BRL
        shares_outstanding=10_000_000.0,
        filing_date="2023-04-15",
        doc_type="DFP",
    )
    _insert_price(mem_conn, "PETR3", "2023-04-14", 30.0)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    # market_cap = 30 * 10M = 300M BRL; net_income = 1_000 * 1_000 = 1M BRL; PE = 300
    expected = (30.0 * 10_000_000) / (1_000.0 * 1_000)  # = 300.0
    assert pe == pytest.approx(expected, rel=1e-3), f"Expected pe_ratio ~{expected}, got {pe}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. P/E is NULL for loss-making companies
# ──────────────────────────────────────────────────────────────────────────────

def test_materialize_pe_ratio_null_for_loss(mem_conn):
    """net_income <= 0 should result in pe_ratio = NULL."""
    _insert_pit_row(
        mem_conn,
        filing_id="LOSS_DFP_2023-12-31_1",
        ticker="PETR",
        net_income=-500_000.0,
        shares_outstanding=10_000_000.0,
        filing_date="2023-04-15",
        doc_type="DFP",
    )
    _insert_price(mem_conn, "PETR3", "2023-04-14", 30.0)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'LOSS_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    assert pe is None, f"Expected pe_ratio = NULL for loss-making company, got {pe}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. ITR Q1 annualizes income by factor of 4
# ──────────────────────────────────────────────────────────────────────────────

def test_materialize_itr_annualizes_income(mem_conn):
    """Q1 ITR net_income should be annualized (×4) and converted from thousands BRL."""
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_ITR_2023-03-31_1",
        ticker="PETR",
        net_income=250.0,   # 250 thousands BRL Q1 actual = 250,000 BRL
        shares_outstanding=10_000_000.0,
        filing_date="2023-05-15",
        doc_type="ITR",
        quarter=1,
        period_end="2023-03-31",
    )
    _insert_price(mem_conn, "PETR3", "2023-05-15", 30.0)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PETR_ITR_2023-03-31_1'")
    pe = cursor.fetchone()[0]
    # annualized NI = 250 * 1_000 * 4 = 1_000_000 BRL; PE = (30 * 10M) / 1M = 300
    assert pe == pytest.approx(300.0, rel=1e-3), f"Expected pe_ratio ~300.0, got {pe}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Uses nearest prior trading day (not future price)
# ──────────────────────────────────────────────────────────────────────────────

def test_materialize_uses_nearest_prior_trading_day(mem_conn):
    """Price lookup should use the latest price ON OR BEFORE filing_date."""
    _insert_pit_row(
        mem_conn,
        filing_id="PRIOR_DFP_2023-12-31_1",
        ticker="PETR",
        net_income=1_000.0,  # 1,000 thousands BRL = 1,000,000 BRL
        shares_outstanding=10_000_000.0,
        filing_date="2023-04-15",  # Saturday
        doc_type="DFP",
    )
    # Friday before: price = 25
    _insert_price(mem_conn, "PETR3", "2023-04-14", 25.0)
    # Monday after: price = 35 — should NOT be used
    _insert_price(mem_conn, "PETR3", "2023-04-17", 35.0)

    materialize_valuation_ratios(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'PRIOR_DFP_2023-12-31_1'")
    pe = cursor.fetchone()[0]
    # market_cap = 25 * 10M = 250M BRL; net_income = 1_000 * 1_000 = 1M BRL; PE = 250
    expected = (25.0 * 10_000_000) / (1_000.0 * 1_000)  # 250.0
    assert pe == pytest.approx(expected, rel=1e-3), (
        f"Expected pe_ratio computed with Friday price (25.0), got {pe}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. PIT correctness — only v1 returned as of 2023-05-01
# ──────────────────────────────────────────────────────────────────────────────

def test_pit_query_respects_filing_date(mem_conn):
    """
    Only filings whose filing_date <= as_of_date should be considered.

    v1 filed 2023-04-01 (revenue=1000), v2 filed 2023-07-01 (revenue=1200).
    As of 2023-05-01 => only v1. As of 2023-08-01 => v2.
    """
    # Version 1
    mem_conn.execute("""
        INSERT INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type,
           revenue, net_income, ebitda, total_assets, equity, net_debt, shares_outstanding)
        VALUES ('v1', '33000167000101', 'PETR', '2022-12-31', '2023-04-01', 1, 'DFP',
                1000, NULL, NULL, NULL, NULL, NULL, NULL)
    """)
    # Version 2 (restatement)
    mem_conn.execute("""
        INSERT INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type,
           revenue, net_income, ebitda, total_assets, equity, net_debt, shares_outstanding)
        VALUES ('v2', '33000167000101', 'PETR', '2022-12-31', '2023-07-01', 2, 'DFP',
                1200, NULL, NULL, NULL, NULL, NULL, NULL)
    """)
    mem_conn.commit()

    cursor = mem_conn.cursor()

    # As of 2023-05-01 — only v1 visible
    cursor.execute("""
        SELECT filing_id, revenue FROM fundamentals_pit
        WHERE cnpj = '33000167000101' AND period_end = '2022-12-31'
          AND filing_date <= '2023-05-01'
        ORDER BY filing_date DESC, filing_version DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "v1", f"Expected v1 as of 2023-05-01, got {row[0]}"
    assert row[1] == 1000

    # As of 2023-08-01 — v2 is now visible
    cursor.execute("""
        SELECT filing_id, revenue FROM fundamentals_pit
        WHERE cnpj = '33000167000101' AND period_end = '2022-12-31'
          AND filing_date <= '2023-08-01'
        ORDER BY filing_date DESC, filing_version DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "v2", f"Expected v2 as of 2023-08-01, got {row[0]}"
    assert row[1] == 1200


# ──────────────────────────────────────────────────────────────────────────────
# 6. PIT returns latest version per period when multiple available
# ──────────────────────────────────────────────────────────────────────────────

def test_pit_query_returns_latest_version_per_period(mem_conn):
    """As of 2023-10-01, the latest version (v3) should be selected."""
    for version, filing_date, revenue in [
        (1, "2023-04-01", 1000),
        (2, "2023-06-01", 1100),
        (3, "2023-09-01", 1200),
    ]:
        mem_conn.execute(
            """
            INSERT INTO fundamentals_pit
              (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type, revenue)
            VALUES (?, '33000167000101', 'PETR', '2022-12-31', ?, ?, 'DFP', ?)
            """,
            (f"v{version}", filing_date, version, revenue),
        )
    mem_conn.commit()

    cursor = mem_conn.cursor()
    cursor.execute("""
        SELECT filing_id, revenue FROM fundamentals_pit
        WHERE cnpj = '33000167000101' AND period_end = '2022-12-31'
          AND filing_date <= '2023-10-01'
        ORDER BY filing_date DESC, filing_version DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "v3", f"Expected v3 as the latest version, got {row[0]}"
    assert row[1] == 1200


# ──────────────────────────────────────────────────────────────────────────────
# Task 06 acceptance test: load_fundamentals_pit forward-fill
# ──────────────────────────────────────────────────────────────────────────────

def test_load_fundamentals_pit_forward_fills(tmp_path):
    """
    load_fundamentals_pit() must forward-fill values across rebalance dates.

    Setup:
    - Two fundamentals_pit rows for PETR with revenue filed on 2023-01-31 (rev=1000)
      and 2023-06-30 (rev=2000).
    - Prices available monthly from 2023-01 through 2023-09.

    Assertion:
    - For all rebalance dates between 2023-01-31 and before 2023-06-30,
      PETR's revenue should be 1000 (forward-filled from the first filing).
    - From 2023-06-30 onwards, PETR's revenue should be 2000.
    """
    from backtests.core.data import load_fundamentals_pit

    db_path = str(tmp_path / "test.sqlite")

    # Create DB with schema
    conn = sqlite3.connect(db_path)
    storage.init_db(conn)

    # Insert prices monthly Jan–Sep 2023 for PETR3
    dates = pd.date_range("2023-01-01", "2023-09-30", freq="ME")
    for dt in dates:
        conn.execute(
            "INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close, volume) "
            "VALUES ('PETR3', 'UNKNOWN', ?, 30, 30, 30, 30, 0)",
            (dt.strftime("%Y-%m-%d"),),
        )

    # Insert two fundamentals_pit rows
    conn.execute("""
        INSERT INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type, revenue)
        VALUES ('f1', '33000167000101', 'PETR', '2022-12-31', '2023-01-31', 1, 'DFP', 1000.0)
    """)
    conn.execute("""
        INSERT INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type, revenue)
        VALUES ('f2', '33000167000101', 'PETR', '2023-06-30', '2023-06-30', 1, 'ITR', 2000.0)
    """)
    conn.commit()
    conn.close()

    # Load with load_fundamentals_pit
    wide = load_fundamentals_pit(db_path, "revenue", "2023-01-01", "2023-09-30", freq="ME")

    assert "PETR" in wide.columns, "PETR should be a column in the wide DataFrame"

    petr = wide["PETR"]

    # Between filing 1 (Jan) and filing 2 (Jun), value should be 1000 (forward-filled)
    mar_val = petr.get(pd.Timestamp("2023-03-31"))
    assert mar_val == pytest.approx(1000.0), (
        f"Expected 1000.0 at 2023-03-31 (forward-fill from Jan filing), got {mar_val}"
    )

    # From Jun 30 onwards, value should be 2000
    jun_val = petr.get(pd.Timestamp("2023-06-30"))
    assert jun_val == pytest.approx(2000.0), (
        f"Expected 2000.0 at 2023-06-30 (second filing), got {jun_val}"
    )

    sep_val = petr.get(pd.Timestamp("2023-09-30"))
    assert sep_val == pytest.approx(2000.0), (
        f"Expected 2000.0 at 2023-09-30 (forward-fill from Jun filing), got {sep_val}"
    )

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


def _insert_price(conn, ticker, date, close, volume=0):
    """Helper: insert a minimal prices row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close, volume)
        VALUES (?, 'UNKNOWN', ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, close, close, close, close, int(volume)),
    )
    conn.commit()


def _insert_company(conn, cnpj, ticker):
    """Helper: insert a cvm_companies row so ADTV ticker map can find it."""
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
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2023-12-31_1",
        ticker="PETR",
        net_income=1_000.0,  # 1,000 thousands BRL = 1,000,000 BRL
        shares_outstanding=10_000_000.0,
        filing_date="2023-04-15",
        doc_type="DFP",
    )
    _insert_price(mem_conn, "PETR3", "2023-04-14", 30.0, volume=10_000)

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
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="LOSS_DFP_2023-12-31_1",
        ticker="PETR",
        net_income=-500_000.0,
        shares_outstanding=10_000_000.0,
        filing_date="2023-04-15",
        doc_type="DFP",
    )
    _insert_price(mem_conn, "PETR3", "2023-04-14", 30.0, volume=10_000)

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
    _insert_company(mem_conn, "33000167000101", "PETR")
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
    _insert_price(mem_conn, "PETR3", "2023-05-15", 30.0, volume=10_000)

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
    """Price lookup should prefer the nearest date; Friday (-1d) beats Monday (+2d)."""
    _insert_company(mem_conn, "33000167000101", "PETR")
    _insert_pit_row(
        mem_conn,
        filing_id="PRIOR_DFP_2023-12-31_1",
        ticker="PETR",
        net_income=1_000.0,  # 1,000 thousands BRL = 1,000,000 BRL
        shares_outstanding=10_000_000.0,
        filing_date="2023-04-15",  # Saturday
        doc_type="DFP",
    )
    # Friday before: price = 25, distance = 1 day
    _insert_price(mem_conn, "PETR3", "2023-04-14", 25.0, volume=10_000)
    # Monday after: price = 35, distance = 2 days — farther, should NOT be preferred
    _insert_price(mem_conn, "PETR3", "2023-04-17", 35.0, volume=10_000)

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


# ──────────────────────────────────────────────────────────────────────────────
# Task 03: Fix filing_date >= start bug
# ──────────────────────────────────────────────────────────────────────────────

def test_load_fundamentals_pit_uses_pre_period_filing_as_seed(tmp_path):
    """A filing before the backtest start should be forward-filled as the initial seed."""
    from backtests.core.data import load_fundamentals_pit

    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    storage.init_db(conn)

    # Insert prices monthly Jan–Jun 2020 for PETR3
    dates = pd.date_range("2020-01-01", "2020-06-30", freq="ME")
    for dt in dates:
        conn.execute(
            "INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close, volume) "
            "VALUES ('PETR3', 'UNKNOWN', ?, 30, 30, 30, 30, 0)",
            (dt.strftime("%Y-%m-%d"),),
        )

    # Filing BEFORE the backtest start
    conn.execute("""
        INSERT INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type, revenue)
        VALUES ('pre_f1', '33000167000101', 'PETR', '2019-12-31', '2019-12-31', 1, 'DFP', 5000.0)
    """)
    conn.commit()
    conn.close()

    wide = load_fundamentals_pit(db_path, "revenue", "2020-01-01", "2020-06-30", freq="ME")

    assert "PETR" in wide.columns, "PETR should be a column in the result"
    # Each month-end should have revenue=5000 forward-filled from the 2019 filing
    for dt in pd.date_range("2020-01-31", "2020-06-30", freq="ME"):
        val = wide.loc[dt, "PETR"] if dt in wide.index else None
        assert val == pytest.approx(5000.0), (
            f"Expected revenue=5000.0 at {dt.date()} (forward-filled from 2019 filing), got {val}"
        )


def test_load_fundamentals_pit_result_starts_at_start_date(tmp_path):
    """Returned DataFrame should not contain dates before start."""
    from backtests.core.data import load_fundamentals_pit

    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    storage.init_db(conn)

    dates = pd.date_range("2020-01-01", "2020-06-30", freq="ME")
    for dt in dates:
        conn.execute(
            "INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close, volume) "
            "VALUES ('PETR3', 'UNKNOWN', ?, 30, 30, 30, 30, 0)",
            (dt.strftime("%Y-%m-%d"),),
        )
    conn.execute("""
        INSERT INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type, revenue)
        VALUES ('pre_f1', '33000167000101', 'PETR', '2019-12-31', '2019-12-31', 1, 'DFP', 5000.0)
    """)
    conn.commit()
    conn.close()

    wide = load_fundamentals_pit(db_path, "revenue", "2020-01-01", "2020-06-30", freq="ME")

    assert wide.index.min() >= pd.Timestamp("2020-01-01"), (
        f"Index min {wide.index.min()} is before start date 2020-01-01"
    )


def test_load_fundamentals_pit_pre_period_overridden_by_in_period_filing(tmp_path):
    """Pre-period filing seeds, then in-period filing overrides from its date."""
    from backtests.core.data import load_fundamentals_pit

    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    storage.init_db(conn)

    dates = pd.date_range("2020-01-01", "2020-06-30", freq="ME")
    for dt in dates:
        conn.execute(
            "INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close, volume) "
            "VALUES ('PETR3', 'UNKNOWN', ?, 30, 30, 30, 30, 0)",
            (dt.strftime("%Y-%m-%d"),),
        )
    # Pre-period filing
    conn.execute("""
        INSERT INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type, revenue)
        VALUES ('pre_f1', '33000167000101', 'PETR', '2019-12-31', '2019-12-31', 1, 'DFP', 5000.0)
    """)
    # In-period filing
    conn.execute("""
        INSERT INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type, revenue)
        VALUES ('in_f2', '33000167000101', 'PETR', '2020-03-31', '2020-03-31', 1, 'ITR', 7000.0)
    """)
    conn.commit()
    conn.close()

    wide = load_fundamentals_pit(db_path, "revenue", "2020-01-01", "2020-06-30", freq="ME")

    assert "PETR" in wide.columns
    petr = wide["PETR"]

    # Jan and Feb should be seeded from the 2019 filing (5000)
    jan_val = petr.get(pd.Timestamp("2020-01-31"))
    feb_val = petr.get(pd.Timestamp("2020-02-29"))
    assert jan_val == pytest.approx(5000.0), f"Expected 5000.0 at Jan, got {jan_val}"
    assert feb_val == pytest.approx(5000.0), f"Expected 5000.0 at Feb, got {feb_val}"

    # Mar–Jun should be from in-period filing (7000)
    mar_val = petr.get(pd.Timestamp("2020-03-31"))
    jun_val = petr.get(pd.Timestamp("2020-06-30"))
    assert mar_val == pytest.approx(7000.0), f"Expected 7000.0 at Mar, got {mar_val}"
    assert jun_val == pytest.approx(7000.0), f"Expected 7000.0 at Jun, got {jun_val}"

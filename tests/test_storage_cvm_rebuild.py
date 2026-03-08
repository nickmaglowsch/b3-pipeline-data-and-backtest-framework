"""
Tests for the cvm_only=True parameter in storage.init_db() (regression guard for
the bug where `cvm_main --rebuild` wiped the prices table).

TDD: these tests are written BEFORE the fix so they fail on the current codebase.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from b3_pipeline import storage


@pytest.fixture
def mem_conn():
    """In-memory SQLite connection with full schema and one price row pre-inserted."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    yield conn
    conn.close()


def _insert_price_row(conn: sqlite3.Connection) -> None:
    """Insert a single synthetic price row to verify preservation."""
    df = pd.DataFrame([{
        "ticker": "PETR4",
        "isin_code": "BRPETRACNPB3",
        "date": "2024-01-02",
        "open": 36.0,
        "high": 37.0,
        "low": 35.5,
        "close": 36.5,
        "volume": 100000,
        "quotation_factor": 1,
    }])
    storage.upsert_prices(conn, df)


def _insert_fundamentals_pit_row(conn: sqlite3.Connection) -> None:
    """Insert a single synthetic fundamentals_pit row to verify it gets wiped."""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO fundamentals_pit
            (filing_id, cnpj, ticker, period_end, filing_date, filing_version, doc_type,
             fiscal_year, quarter, revenue, net_income, ebitda, total_assets, equity,
             net_debt, shares_outstanding)
        VALUES
            ('TEST_FID_001', '33000167000101', 'PETR4', '2023-12-31', '2024-03-15',
             1, 'DFP', 2023, NULL, 500000.0, 50000.0, 80000.0, 1000000.0,
             300000.0, 200000.0, 13000000.0)
    """)
    conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: cvm_only=True must NOT drop prices (the core bug regression)
# ──────────────────────────────────────────────────────────────────────────────

def test_cvm_only_rebuild_preserves_prices(mem_conn):
    """
    init_db(rebuild=True, cvm_only=True) must leave prices intact.

    This is the direct regression test: running the CVM pipeline with --rebuild
    should never touch COTAHIST-owned tables.
    """
    _insert_price_row(mem_conn)
    cursor = mem_conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM prices")
    assert cursor.fetchone()[0] == 1, "Pre-condition: prices must have 1 row"

    storage.init_db(mem_conn, rebuild=True, cvm_only=True)

    cursor.execute("SELECT COUNT(*) FROM prices")
    count = cursor.fetchone()[0]
    assert count == 1, (
        f"prices table was destroyed by cvm_only rebuild (expected 1 row, got {count}). "
        "This is the regression: init_db(cvm_only=True) must not drop COTAHIST tables."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: cvm_only=True MUST wipe CVM tables (fundamentals_pit)
# ──────────────────────────────────────────────────────────────────────────────

def test_cvm_only_rebuild_wipes_fundamentals_pit(mem_conn):
    """
    init_db(rebuild=True, cvm_only=True) must drop and recreate CVM tables,
    leaving fundamentals_pit empty.
    """
    _insert_fundamentals_pit_row(mem_conn)
    cursor = mem_conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM fundamentals_pit")
    assert cursor.fetchone()[0] == 1, "Pre-condition: fundamentals_pit must have 1 row"

    storage.init_db(mem_conn, rebuild=True, cvm_only=True)

    cursor.execute("SELECT COUNT(*) FROM fundamentals_pit")
    count = cursor.fetchone()[0]
    assert count == 0, (
        f"fundamentals_pit was not wiped by cvm_only rebuild (expected 0 rows, got {count})."
    )

    # Table must still exist (dropped and recreated, not just truncated)
    cursor.execute("PRAGMA table_info(fundamentals_pit)")
    cols = cursor.fetchall()
    assert len(cols) > 0, "fundamentals_pit table must still exist after rebuild"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: full rebuild (cvm_only=False) drops everything including prices
# ──────────────────────────────────────────────────────────────────────────────

def test_full_rebuild_drops_prices(mem_conn):
    """
    init_db(rebuild=True, cvm_only=False) must drop ALL tables, including prices.

    This verifies that the existing full-rebuild behaviour is preserved when
    cvm_only defaults to False.
    """
    _insert_price_row(mem_conn)
    _insert_fundamentals_pit_row(mem_conn)
    cursor = mem_conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM prices")
    assert cursor.fetchone()[0] == 1
    cursor.execute("SELECT COUNT(*) FROM fundamentals_pit")
    assert cursor.fetchone()[0] == 1

    storage.init_db(mem_conn, rebuild=True, cvm_only=False)

    cursor.execute("SELECT COUNT(*) FROM prices")
    assert cursor.fetchone()[0] == 0, "Full rebuild must empty prices"

    cursor.execute("SELECT COUNT(*) FROM fundamentals_pit")
    assert cursor.fetchone()[0] == 0, "Full rebuild must empty fundamentals_pit"

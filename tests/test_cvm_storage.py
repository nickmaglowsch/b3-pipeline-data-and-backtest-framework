"""
Tests for CVM storage layer (Task 01 — TDD).

Tests are written against an in-memory SQLite DB to avoid touching the
real b3_market_data.sqlite file.
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from b3_pipeline import storage
from b3_pipeline import cvm_storage


@pytest.fixture
def mem_conn():
    """In-memory SQLite connection with full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")  # allow testing without FK cascades
    storage.init_db(conn)
    yield conn
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Table creation
# ──────────────────────────────────────────────────────────────────────────────

def test_init_db_creates_new_tables(mem_conn):
    """init_db() should create cvm_companies, cvm_filings, fundamentals_pit."""
    cursor = mem_conn.cursor()

    for table in ("cvm_companies", "cvm_filings", "fundamentals_pit"):
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [row[1] for row in cursor.fetchall()]
        assert len(cols) > 0, f"Table {table} has no columns — was not created"

    # Spot-check specific columns
    cursor.execute("PRAGMA table_info(cvm_companies)")
    cols = [row[1] for row in cursor.fetchall()]
    assert "cnpj" in cols
    assert "ticker" in cols

    cursor.execute("PRAGMA table_info(fundamentals_pit)")
    cols = [row[1] for row in cursor.fetchall()]
    assert "filing_id" in cols
    assert "pe_ratio" in cols
    assert "net_debt" in cols


# ──────────────────────────────────────────────────────────────────────────────
# 2. Rebuild drops and recreates
# ──────────────────────────────────────────────────────────────────────────────

def test_init_db_rebuild_drops_and_recreates(mem_conn):
    """rebuild=True should drop all data but leave tables intact."""
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj="12345678000100", ticker="TEST",
        company_name="Test Co", cvm_code="999", b3_trading_name="TEST"
    )
    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cvm_companies")
    assert cursor.fetchone()[0] == 1

    storage.init_db(mem_conn, rebuild=True)

    cursor.execute("SELECT COUNT(*) FROM cvm_companies")
    assert cursor.fetchone()[0] == 0, "Table should be empty after rebuild"

    # Table must still exist
    cursor.execute("PRAGMA table_info(cvm_companies)")
    cols = cursor.fetchall()
    assert len(cols) > 0, "cvm_companies table should still exist after rebuild"


# ──────────────────────────────────────────────────────────────────────────────
# 3. upsert_cvm_company — insert and update
# ──────────────────────────────────────────────────────────────────────────────

def test_upsert_cvm_company_insert_and_update(mem_conn):
    """upsert_cvm_company() should update on conflict, not duplicate."""
    cnpj = "33000167000101"

    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj=cnpj, ticker="PETR",
        company_name="Petrobras", cvm_code="9512", b3_trading_name="PETROBRAS"
    )
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj=cnpj, ticker="PETR4",  # updated ticker
        company_name="Petrobras S.A.", cvm_code="9512", b3_trading_name="PETROBRAS"
    )

    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*), ticker FROM cvm_companies WHERE cnpj = ?", (cnpj,))
    row = cursor.fetchone()
    assert row[0] == 1, "Should have exactly 1 row, not duplicated"
    assert row[1] == "PETR4", "Ticker should be updated to new value"


# ──────────────────────────────────────────────────────────────────────────────
# 4. upsert_fundamentals_pit — batch insert
# ──────────────────────────────────────────────────────────────────────────────

def test_upsert_fundamentals_pit_batch(mem_conn):
    """upsert_fundamentals_pit() should insert all rows and return the count."""
    df = pd.DataFrame([
        {
            "filing_id": "CNPJ1_DFP_2023-12-31_1",
            "cnpj": "11111111000100",
            "ticker": "AAAA",
            "period_end": "2023-12-31",
            "filing_date": "2024-03-15",
            "filing_version": 1,
            "doc_type": "DFP",
            "fiscal_year": 2023,
            "quarter": None,
            "revenue": 1_000_000.0,
            "net_income": 200_000.0,
            "ebitda": 300_000.0,
            "total_assets": 5_000_000.0,
            "equity": 2_000_000.0,
            "net_debt": 500_000.0,
            "shares_outstanding": 1_000_000.0,
            "pe_ratio": None,
            "pb_ratio": None,
            "ev_ebitda": None,
        },
        {
            "filing_id": "CNPJ2_DFP_2023-12-31_1",
            "cnpj": "22222222000100",
            "ticker": "BBBB",
            "period_end": "2023-12-31",
            "filing_date": "2024-03-20",
            "filing_version": 1,
            "doc_type": "DFP",
            "fiscal_year": 2023,
            "quarter": None,
            "revenue": 500_000.0,
            "net_income": 50_000.0,
            "ebitda": 80_000.0,
            "total_assets": 2_000_000.0,
            "equity": 800_000.0,
            "net_debt": 200_000.0,
            "shares_outstanding": 500_000.0,
            "pe_ratio": None,
            "pb_ratio": None,
            "ev_ebitda": None,
        },
        {
            "filing_id": "CNPJ3_DFP_2023-12-31_1",
            "cnpj": "33333333000100",
            "ticker": "CCCC",
            "period_end": "2023-12-31",
            "filing_date": "2024-03-25",
            "filing_version": 1,
            "doc_type": "DFP",
            "fiscal_year": 2023,
            "quarter": None,
            "revenue": 750_000.0,
            "net_income": 100_000.0,
            "ebitda": 150_000.0,
            "total_assets": 3_000_000.0,
            "equity": 1_200_000.0,
            "net_debt": 300_000.0,
            "shares_outstanding": 750_000.0,
            "pe_ratio": None,
            "pb_ratio": None,
            "ev_ebitda": None,
        },
    ])

    result = cvm_storage.upsert_fundamentals_pit(mem_conn, df)
    assert result == 3, f"Expected 3 rows inserted, got {result}"

    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fundamentals_pit")
    count = cursor.fetchone()[0]
    assert count == 3, f"DB should have 3 rows, found {count}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. get_summary_stats includes fundamentals keys
# ──────────────────────────────────────────────────────────────────────────────

def test_get_summary_stats_includes_fundamentals_keys(mem_conn):
    """get_summary_stats() must include the three CVM count keys (values = 0 when empty)."""
    stats = storage.get_summary_stats(mem_conn)
    assert "total_cvm_companies" in stats, "Missing total_cvm_companies"
    assert "total_cvm_filings" in stats, "Missing total_cvm_filings"
    assert "total_fundamentals_pit" in stats, "Missing total_fundamentals_pit"
    assert stats["total_cvm_companies"] == 0
    assert stats["total_cvm_filings"] == 0
    assert stats["total_fundamentals_pit"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# 6. get_cvm_company_map
# ──────────────────────────────────────────────────────────────────────────────

def test_get_cvm_company_map_returns_dict(mem_conn):
    """get_cvm_company_map() should return {cnpj: ticker} for all mapped rows."""
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj="12345", ticker="PETR",
        company_name="Petrobras", cvm_code="1", b3_trading_name="PETROBRAS"
    )
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj="67890", ticker="VALE",
        company_name="Vale", cvm_code="2", b3_trading_name="VALE"
    )

    result = cvm_storage.get_cvm_company_map(mem_conn)
    assert result == {"12345": "PETR", "67890": "VALE"}, (
        f"Unexpected map: {result}"
    )

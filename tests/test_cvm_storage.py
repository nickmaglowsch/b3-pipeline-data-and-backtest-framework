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


# ──────────────────────────────────────────────────────────────────────────────
# Task 03: CAD storage — listing_date / delisting_date schema migration
# ──────────────────────────────────────────────────────────────────────────────

def test_migrate_schema_adds_listing_date_columns():
    """init_db() should add listing_date and delisting_date to cvm_companies."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    # Create cvm_companies WITHOUT the new columns (simulate old DB)
    conn.execute("""
        CREATE TABLE cvm_companies (
            cnpj TEXT NOT NULL PRIMARY KEY,
            ticker TEXT,
            company_name TEXT,
            cvm_code TEXT,
            b3_trading_name TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # Running init_db should migrate the schema
    storage.init_db(conn)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(cvm_companies)")
    cols = {row[1] for row in cursor.fetchall()}
    assert "listing_date" in cols, "listing_date column should exist after migration"
    assert "delisting_date" in cols, "delisting_date column should exist after migration"
    conn.close()


def test_upsert_cad_company_dates_inserts_new_row(mem_conn):
    """upsert_cad_company_dates() inserts a new row with listing/delisting dates."""
    df = pd.DataFrame([{
        "cnpj": "12345678000100",
        "cvm_code": "001",
        "company_name": "Test Corp",
        "listing_date": "2000-01-15",
        "delisting_date": None,
    }])
    count = cvm_storage.upsert_cad_company_dates(mem_conn, df)
    assert count == 1
    cursor = mem_conn.cursor()
    cursor.execute("SELECT listing_date, delisting_date FROM cvm_companies WHERE cnpj = ?",
                   ("12345678000100",))
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "2000-01-15"
    assert row[1] is None


def test_upsert_cad_company_dates_does_not_overwrite_ticker(mem_conn):
    """upsert_cad_company_dates() must NOT overwrite an existing ticker."""
    # Insert company with ticker via the regular upsert
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj="12345678000100", ticker="PETR",
        company_name="Petrobras", cvm_code="9512", b3_trading_name="PETROBRAS"
    )
    # Now upsert CAD dates for same CNPJ
    df = pd.DataFrame([{
        "cnpj": "12345678000100",
        "cvm_code": "9512",
        "company_name": "Petrobras",
        "listing_date": "1994-04-08",
        "delisting_date": None,
    }])
    cvm_storage.upsert_cad_company_dates(mem_conn, df)
    cursor = mem_conn.cursor()
    cursor.execute("SELECT ticker FROM cvm_companies WHERE cnpj = ?", ("12345678000100",))
    assert cursor.fetchone()[0] == "PETR", "Ticker should be preserved"


def test_upsert_cad_company_dates_preserves_existing_listing_date(mem_conn):
    """listing_date uses COALESCE — first known date wins."""
    df1 = pd.DataFrame([{
        "cnpj": "12345678000100",
        "cvm_code": "001",
        "company_name": "A Corp",
        "listing_date": "2005-01-01",
        "delisting_date": None,
    }])
    cvm_storage.upsert_cad_company_dates(mem_conn, df1)

    df2 = pd.DataFrame([{
        "cnpj": "12345678000100",
        "cvm_code": "001",
        "company_name": "A Corp",
        "listing_date": "2010-01-01",  # later date — should NOT win
        "delisting_date": None,
    }])
    cvm_storage.upsert_cad_company_dates(mem_conn, df2)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT listing_date FROM cvm_companies WHERE cnpj = ?", ("12345678000100",))
    assert cursor.fetchone()[0] == "2005-01-01", "First listing_date should be preserved"


def test_upsert_cad_company_dates_overwrites_delisting_date(mem_conn):
    """delisting_date is always overwritten (cancellations can be updated)."""
    df1 = pd.DataFrame([{
        "cnpj": "12345678000100",
        "cvm_code": "001",
        "company_name": "A Corp",
        "listing_date": "2000-01-01",
        "delisting_date": "2015-06-30",
    }])
    cvm_storage.upsert_cad_company_dates(mem_conn, df1)

    df2 = pd.DataFrame([{
        "cnpj": "12345678000100",
        "cvm_code": "001",
        "company_name": "A Corp",
        "listing_date": "2000-01-01",
        "delisting_date": "2016-03-15",  # updated date
    }])
    cvm_storage.upsert_cad_company_dates(mem_conn, df2)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT delisting_date FROM cvm_companies WHERE cnpj = ?", ("12345678000100",))
    assert cursor.fetchone()[0] == "2016-03-15", "delisting_date should be updated"


def test_get_fundamentals_stats_includes_listing_counts(mem_conn):
    """get_fundamentals_stats() returns companies_with_listing_date count."""
    df = pd.DataFrame([
        {
            "cnpj": "11111111000100",
            "cvm_code": "001",
            "company_name": "Alpha",
            "listing_date": "2000-01-01",
            "delisting_date": None,
        },
        {
            "cnpj": "22222222000100",
            "cvm_code": "002",
            "company_name": "Beta",
            "listing_date": None,
            "delisting_date": None,
        },
    ])
    cvm_storage.upsert_cad_company_dates(mem_conn, df)
    stats = cvm_storage.get_fundamentals_stats(mem_conn)
    assert stats["companies_with_listing_date"] == 1
    assert "companies_with_delisting_date" in stats

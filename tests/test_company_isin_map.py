"""
Tests for company_isin_map table and related population functions.

TDD: these tests are written BEFORE the implementation and should initially FAIL.

Fix 3 coverage:
- company_isin_map table created by init_db()
- populate_company_isin_map() fills rows from prices + cvm_companies join
- share_class inference from ticker suffix
- is_primary flag set for ON share class (suffix 3)
- materialize_valuation_ratios uses ISIN-based join when company_isin_map has data
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from b3_pipeline import storage, cvm_storage


@pytest.fixture
def mem_conn():
    """In-memory SQLite connection with full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    yield conn
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# 1. Table creation
# ──────────────────────────────────────────────────────────────────────────────

def test_company_isin_map_table_exists(mem_conn):
    """init_db() should create the company_isin_map table."""
    cursor = mem_conn.cursor()
    cursor.execute("PRAGMA table_info(company_isin_map)")
    cols = [row[1] for row in cursor.fetchall()]
    assert len(cols) > 0, "company_isin_map table was not created"
    assert "cnpj" in cols, "Missing 'cnpj' column"
    assert "isin_code" in cols, "Missing 'isin_code' column"
    assert "ticker" in cols, "Missing 'ticker' column"
    assert "share_class" in cols, "Missing 'share_class' column"
    assert "is_primary" in cols, "Missing 'is_primary' column"
    assert "first_seen" in cols, "Missing 'first_seen' column"
    assert "last_seen" in cols, "Missing 'last_seen' column"


def test_company_isin_map_primary_key(mem_conn):
    """company_isin_map primary key is (cnpj, isin_code)."""
    cursor = mem_conn.cursor()
    # Insert a row
    mem_conn.execute("""
        INSERT INTO company_isin_map (cnpj, isin_code, ticker, share_class, is_primary, first_seen, last_seen)
        VALUES ('33000167000101', 'BRPETRACNOR9', 'PETR3', 'ON', 1, '2020-01-01', '2024-12-31')
    """)
    mem_conn.commit()
    # Inserting again with same PK should fail (or replace with INSERT OR REPLACE)
    with pytest.raises(Exception):
        mem_conn.execute("""
            INSERT INTO company_isin_map (cnpj, isin_code, ticker, share_class, is_primary, first_seen, last_seen)
            VALUES ('33000167000101', 'BRPETRACNOR9', 'PETR3', 'ON', 1, '2020-01-01', '2024-12-31')
        """)
        mem_conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# 2. populate_company_isin_map function
# ──────────────────────────────────────────────────────────────────────────────

def test_populate_company_isin_map_is_importable():
    """populate_company_isin_map must exist in cvm_storage."""
    assert hasattr(cvm_storage, "populate_company_isin_map"), (
        "populate_company_isin_map not found in cvm_storage"
    )
    assert callable(cvm_storage.populate_company_isin_map)


def test_populate_company_isin_map_fills_rows(mem_conn):
    """
    Given cvm_companies with ticker='PETR' and prices with PETR3/PETR4 ISINs,
    populate_company_isin_map should create rows linking CNPJ to ISINs.
    """
    # Seed cvm_companies
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    # Seed prices with two share classes
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2020-01-02', 25.0)"
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-12-31', 35.0)"
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR4', 'BRPETRACNPR6', '2020-01-02', 24.0)"
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR4', 'BRPETRACNPR6', '2024-12-31', 34.0)"
    )
    mem_conn.commit()

    count = cvm_storage.populate_company_isin_map(mem_conn)
    assert count >= 2, f"Expected at least 2 rows, got {count}"

    cursor = mem_conn.cursor()
    cursor.execute(
        "SELECT isin_code, ticker, share_class FROM company_isin_map WHERE cnpj = '33000167000101' ORDER BY isin_code"
    )
    rows = cursor.fetchall()
    isin_set = {r[0] for r in rows}
    assert "BRPETRACNOR9" in isin_set, "Missing PETR3 ISIN row"
    assert "BRPETRACNPR6" in isin_set, "Missing PETR4 ISIN row"


def test_populate_company_isin_map_share_class_on(mem_conn):
    """Ticker ending in '3' should map to share_class='ON'."""
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    cvm_storage.populate_company_isin_map(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT share_class FROM company_isin_map WHERE ticker = 'PETR3'")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "ON", f"Expected share_class='ON', got {row[0]!r}"


def test_populate_company_isin_map_share_class_pn(mem_conn):
    """Ticker ending in '4' should map to share_class='PN'."""
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR4', 'BRPETRACNPR6', '2024-01-02', 29.0)"
    )
    mem_conn.commit()

    cvm_storage.populate_company_isin_map(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT share_class FROM company_isin_map WHERE ticker = 'PETR4'")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "PN", f"Expected share_class='PN', got {row[0]!r}"


def test_populate_company_isin_map_share_class_unit(mem_conn):
    """Ticker with suffix '11' should map to share_class='UNT'."""
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="00000000000191",
        ticker="SANB",
        company_name="Santander BR",
        cvm_code="020121",
        b3_trading_name="SANTANDER BR",
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('SANB11', 'BRSANBCTF004', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    cvm_storage.populate_company_isin_map(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT share_class FROM company_isin_map WHERE ticker = 'SANB11'")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "UNT", f"Expected share_class='UNT', got {row[0]!r}"


def test_populate_company_isin_map_is_primary_for_on(mem_conn):
    """is_primary should be 1 for the ON share class (ticker suffix 3)."""
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-01-02', 30.0)"
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR4', 'BRPETRACNPR6', '2024-01-02', 29.0)"
    )
    mem_conn.commit()

    cvm_storage.populate_company_isin_map(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute(
        "SELECT ticker, is_primary FROM company_isin_map WHERE cnpj = '33000167000101' ORDER BY ticker"
    )
    rows = {r[0]: r[1] for r in cursor.fetchall()}
    assert rows.get("PETR3") == 1, f"PETR3 (ON) should be primary, got is_primary={rows.get('PETR3')}"
    assert rows.get("PETR4") == 0, f"PETR4 (PN) should not be primary, got is_primary={rows.get('PETR4')}"


def test_populate_company_isin_map_first_last_seen(mem_conn):
    """first_seen and last_seen should track the date range in prices."""
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    for dt in ["2020-01-02", "2022-06-15", "2024-12-31"]:
        mem_conn.execute(
            "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', ?, 30.0)",
            (dt,)
        )
    mem_conn.commit()

    cvm_storage.populate_company_isin_map(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute(
        "SELECT first_seen, last_seen FROM company_isin_map WHERE ticker = 'PETR3'"
    )
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "2020-01-02", f"Expected first_seen='2020-01-02', got {row[0]!r}"
    assert row[1] == "2024-12-31", f"Expected last_seen='2024-12-31', got {row[1]!r}"


def test_populate_company_isin_map_skips_unknown_isin(mem_conn):
    """Rows with isin_code='UNKNOWN' should not be inserted."""
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'UNKNOWN', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    count = cvm_storage.populate_company_isin_map(mem_conn)
    assert count == 0, f"Expected 0 rows (UNKNOWN ISINs skipped), got {count}"


def test_populate_company_isin_map_no_cvm_company_skips(mem_conn):
    """Prices without a matching cvm_companies row (no ticker) should be skipped."""
    # No cvm_companies row
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    count = cvm_storage.populate_company_isin_map(mem_conn)
    assert count == 0, f"Expected 0 rows (no cvm_companies match), got {count}"


def test_populate_company_isin_map_is_idempotent(mem_conn):
    """Calling populate_company_isin_map twice should not duplicate rows."""
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    cvm_storage.populate_company_isin_map(mem_conn)
    cvm_storage.populate_company_isin_map(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM company_isin_map WHERE cnpj = '33000167000101'")
    count = cursor.fetchone()[0]
    assert count == 1, f"Expected 1 row after 2 calls, got {count}"


# ──────────────────────────────────────────────────────────────────────────────
# 3. materialize_valuation_ratios uses ISIN-based join when possible
# ──────────────────────────────────────────────────────────────────────────────

def test_materialize_uses_isin_join_when_company_isin_map_populated(mem_conn):
    """
    When company_isin_map has a primary share row, materialize_valuation_ratios
    should find the price using the ISIN join (not the LIKE 'ticker%' fallback).
    This test verifies ratio is computed correctly.
    """
    from b3_pipeline import cvm_main

    cnpj = "33000167000101"

    # Seed cvm_companies
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj=cnpj,
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    # Seed prices
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-03-14', 36.50)"
    )
    mem_conn.commit()

    # Populate company_isin_map so materialize can use ISIN join
    cvm_storage.populate_company_isin_map(mem_conn)

    # Seed fundamentals_pit row that needs ratio computation
    mem_conn.execute("""
        INSERT INTO fundamentals_pit
            (filing_id, cnpj, ticker, period_end, filing_date, filing_version,
             doc_type, fiscal_year, quarter, net_income, equity, ebitda, net_debt, shares_outstanding)
        VALUES
            ('CNPJ_DFP_2023-12-31_1', ?, 'PETR', '2023-12-31', '2024-03-15', 1,
             'DFP', 2023, NULL, 50000.0, 200000.0, 80000.0, 10000.0, 13000.0)
    """, (cnpj,))
    mem_conn.commit()

    updated = cvm_main.materialize_valuation_ratios(mem_conn)
    assert updated == 1, f"Expected 1 row updated, got {updated}"

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio, pb_ratio FROM fundamentals_pit WHERE filing_id = 'CNPJ_DFP_2023-12-31_1'")
    row = cursor.fetchone()
    assert row is not None
    pe, pb = row
    assert pe is not None, "pe_ratio should be computed"
    assert pb is not None, "pb_ratio should be computed"
    # market_cap = 36.50 * 13000 = 474500
    # net_income = 50000 * 1000 = 50_000_000
    # pe = 474500 / 50_000_000 ≈ 0.00949
    assert abs(pe - (36.50 * 13000.0) / (50000.0 * 1000.0)) < 0.001


def test_materialize_uses_ticker_like_fallback_when_no_isin_map(mem_conn):
    """
    When company_isin_map is empty, materialize_valuation_ratios falls back to
    the LIKE 'ticker%' price lookup and still computes ratios correctly.
    """
    from b3_pipeline import cvm_main

    cnpj = "19526477000175"

    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj=cnpj,
        ticker="VALE",
        company_name="Vale",
        cvm_code="004170",
        b3_trading_name="VALE",
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('VALE3', 'BRVALEACNOR0', '2024-03-14', 65.00)"
    )
    mem_conn.commit()

    # Do NOT call populate_company_isin_map — leave company_isin_map empty

    mem_conn.execute("""
        INSERT INTO fundamentals_pit
            (filing_id, cnpj, ticker, period_end, filing_date, filing_version,
             doc_type, fiscal_year, quarter, net_income, equity, ebitda, net_debt, shares_outstanding)
        VALUES
            ('VALE_DFP_2023-12-31_1', ?, 'VALE', '2023-12-31', '2024-03-15', 1,
             'DFP', 2023, NULL, 70000.0, 300000.0, 100000.0, 20000.0, 5000.0)
    """, (cnpj,))
    mem_conn.commit()

    updated = cvm_main.materialize_valuation_ratios(mem_conn)
    assert updated == 1, f"Expected 1 row updated with LIKE fallback, got {updated}"

    cursor = mem_conn.cursor()
    cursor.execute("SELECT pe_ratio FROM fundamentals_pit WHERE filing_id = 'VALE_DFP_2023-12-31_1'")
    row = cursor.fetchone()
    assert row is not None and row[0] is not None, "pe_ratio should be computed via LIKE fallback"


# ──────────────────────────────────────────────────────────────────────────────
# 4. company_isin_map survives rebuild
# ──────────────────────────────────────────────────────────────────────────────

def test_company_isin_map_dropped_on_rebuild():
    """rebuild=True should drop company_isin_map data but keep the table."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)

    conn.execute("""
        INSERT INTO company_isin_map (cnpj, isin_code, ticker, share_class, is_primary, first_seen, last_seen)
        VALUES ('33000167000101', 'BRPETRACNOR9', 'PETR3', 'ON', 1, '2020-01-01', '2024-12-31')
    """)
    conn.commit()

    storage.init_db(conn, rebuild=True)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM company_isin_map")
    count = cursor.fetchone()[0]
    assert count == 0, f"company_isin_map should be empty after rebuild, got {count}"

    # Table should still exist
    cursor.execute("PRAGMA table_info(company_isin_map)")
    cols = cursor.fetchall()
    assert len(cols) > 0, "company_isin_map table should still exist after rebuild"
    conn.close()

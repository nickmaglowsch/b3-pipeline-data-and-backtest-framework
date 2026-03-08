"""
Tests for company_isin_map table and related population functions.

TDD: these tests are written BEFORE the implementation and should initially FAIL.

Fix 3 coverage:
- company_isin_map table created by init_db()
- populate_company_isin_map() fills rows from prices + cvm_companies join
- share_class inference from ticker suffix
- is_primary flag set for ON share class (suffix 3)
- company_isin_map enables ISIN-based price joins for dynamic ratio computation
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
# 3. company_isin_map enables price lookup for fundamentals (ISIN join)
#    Note: materialize_valuation_ratios() was removed in the refactor.
#    Ratios are now computed dynamically at query time in build_shared_data().
#    These tests verify the company_isin_map data can be queried for price joins.
# ──────────────────────────────────────────────────────────────────────────────

def test_company_isin_map_can_join_prices_via_isin(mem_conn):
    """
    When company_isin_map is populated, a CNPJ can be joined to a price row via ISIN.
    This is the join pattern used for dynamic ratio computation at query time.
    """
    cnpj = "33000167000101"

    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj=cnpj,
        ticker="PETR",
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name="PETROBRAS",
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-03-14', 36.50)"
    )
    mem_conn.commit()

    cvm_storage.populate_company_isin_map(mem_conn)

    # Verify that a CNPJ → price join via company_isin_map works
    cursor = mem_conn.cursor()
    cursor.execute("""
        SELECT p.close
        FROM prices p
        JOIN company_isin_map m ON m.isin_code = p.isin_code
        JOIN cvm_companies c ON c.cnpj = m.cnpj
        WHERE c.cnpj = ? AND m.is_primary = 1 AND p.date = '2024-03-14'
    """, (cnpj,))
    row = cursor.fetchone()
    assert row is not None, "Expected a price row via ISIN join"
    assert abs(row[0] - 36.50) < 0.01, f"Expected close=36.50, got {row[0]}"


def test_company_isin_map_fundamentals_pit_has_no_ratio_columns(mem_conn):
    """
    fundamentals_pit no longer contains pe_ratio / pb_ratio / ev_ebitda columns.
    Ratios are computed dynamically at query time by build_shared_data().
    """
    cursor = mem_conn.cursor()
    cursor.execute("PRAGMA table_info(fundamentals_pit)")
    cols = {row[1] for row in cursor.fetchall()}
    assert "pe_ratio" not in cols, "pe_ratio should not be a column in fundamentals_pit"
    assert "pb_ratio" not in cols, "pb_ratio should not be a column in fundamentals_pit"
    assert "ev_ebitda" not in cols, "ev_ebitda should not be a column in fundamentals_pit"
    # Raw inputs for dynamic computation must still be present
    assert "net_income" in cols
    assert "equity" in cols
    assert "shares_outstanding" in cols


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

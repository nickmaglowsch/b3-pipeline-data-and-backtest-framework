"""
Tests for CNPJ mapping functions (Task 02 — TDD).
"""
from __future__ import annotations

import sqlite3

import pytest

from b3_pipeline import storage
from b3_pipeline import cvm_storage
from b3_pipeline.b3_corporate_actions import (
    extract_cnpj_from_company_data,
    build_cnpj_ticker_map,
)


@pytest.fixture
def mem_conn():
    """In-memory SQLite connection with full schema."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    yield conn
    conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# 1. extract_cnpj_from_company_data — formatting stripped
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_cnpj_strips_formatting():
    """CNPJ with dots/slashes/dashes should be returned as 14-digit string."""
    result = extract_cnpj_from_company_data({"cnpj": "33.000.167/0001-01"})
    assert result == "33000167000101", f"Expected '33000167000101', got {result!r}"


def test_extract_cnpj_returns_none_for_missing():
    """Missing 'cnpj' key in dict should return None."""
    result = extract_cnpj_from_company_data({})
    assert result is None, f"Expected None, got {result!r}"


def test_extract_cnpj_returns_none_for_wrong_length():
    """CNPJ with wrong digit count should return None."""
    result = extract_cnpj_from_company_data({"cnpj": "123"})
    assert result is None, f"Expected None for short CNPJ, got {result!r}"


# ──────────────────────────────────────────────────────────────────────────────
# 4. build_cnpj_ticker_map — round trip
# ──────────────────────────────────────────────────────────────────────────────

def test_build_cnpj_ticker_map_round_trip(mem_conn):
    """Inserted companies should appear in build_cnpj_ticker_map output."""
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj="33000167000101", ticker="PETR",
        company_name="Petrobras", cvm_code="9512", b3_trading_name="PETROBRAS"
    )
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj="19526477000175", ticker="VALE",
        company_name="Vale", cvm_code="4170", b3_trading_name="VALE"
    )

    result = build_cnpj_ticker_map(mem_conn)
    assert result == {
        "33000167000101": "PETR",
        "19526477000175": "VALE",
    }, f"Unexpected map: {result}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. upsert_cvm_company — idempotent
# ──────────────────────────────────────────────────────────────────────────────

def test_upsert_cvm_company_idempotent(mem_conn):
    """Calling upsert twice with same CNPJ but different ticker should update, not duplicate."""
    cnpj = "33000167000101"

    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj=cnpj, ticker="PETR",
        company_name="Petrobras", cvm_code="9512", b3_trading_name="PETROBRAS"
    )
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj=cnpj, ticker="PETR4",  # updated
        company_name="Petrobras", cvm_code="9512", b3_trading_name="PETROBRAS"
    )

    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*), ticker FROM cvm_companies WHERE cnpj = ?", (cnpj,))
    count, ticker = cursor.fetchone()
    assert count == 1, "Should have exactly 1 row after 2 upserts with same CNPJ"
    assert ticker == "PETR4", f"Ticker should be updated; got {ticker!r}"

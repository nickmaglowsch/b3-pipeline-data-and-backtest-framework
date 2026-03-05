"""
Tests for fetch_all_b3_listed_companies() in b3_corporate_actions.py (Fix 2)
and match-rate warning in _fetch_ticker_mappings() (Fix 4).

TDD: these tests are written BEFORE the implementation and should initially FAIL.
"""
from __future__ import annotations

import sqlite3
import logging
from unittest.mock import MagicMock, patch, call

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
# Fix 2: fetch_all_b3_listed_companies
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_all_b3_listed_companies_is_importable():
    """fetch_all_b3_listed_companies must exist in b3_corporate_actions."""
    from b3_pipeline import b3_corporate_actions
    assert hasattr(b3_corporate_actions, "fetch_all_b3_listed_companies"), (
        "fetch_all_b3_listed_companies not found in b3_corporate_actions"
    )
    assert callable(b3_corporate_actions.fetch_all_b3_listed_companies)


def test_fetch_all_b3_listed_companies_single_page():
    """
    When the B3 API returns a single page of companies, all records are returned.
    """
    from b3_pipeline import b3_corporate_actions

    fake_page = {
        "page": {"pageNumber": 1, "pageSize": 3, "totalPages": 1, "totalRecords": 3},
        "results": [
            {"codeCVM": "9512", "issuingCompany": "PETR", "companyName": "PETROBRAS",
             "tradingName": "PETROBRAS", "cnpj": "33000167000101"},
            {"codeCVM": "4170", "issuingCompany": "VALE", "companyName": "VALE",
             "tradingName": "VALE", "cnpj": "19526477000175"},
            {"codeCVM": "906", "issuingCompany": "BBDC", "companyName": "BRADESCO",
             "tradingName": "BRADESCO", "cnpj": "60746948000112"},
        ],
    }

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = fake_page
        mock_get.return_value = mock_resp

        result = b3_corporate_actions.fetch_all_b3_listed_companies(page_size=3)

    assert len(result) == 3, f"Expected 3 companies, got {len(result)}"
    cvm_codes = {r["codeCVM"] for r in result}
    assert "9512" in cvm_codes
    assert "4170" in cvm_codes
    assert "906" in cvm_codes


def test_fetch_all_b3_listed_companies_paginates():
    """
    When totalPages > 1, fetch_all_b3_listed_companies should make multiple requests.
    """
    from b3_pipeline import b3_corporate_actions

    page1 = {
        "page": {"pageNumber": 1, "pageSize": 2, "totalPages": 2, "totalRecords": 3},
        "results": [
            {"codeCVM": "9512", "issuingCompany": "PETR", "cnpj": "33000167000101"},
            {"codeCVM": "4170", "issuingCompany": "VALE", "cnpj": "19526477000175"},
        ],
    }
    page2 = {
        "page": {"pageNumber": 2, "pageSize": 2, "totalPages": 2, "totalRecords": 3},
        "results": [
            {"codeCVM": "906", "issuingCompany": "BBDC", "cnpj": "60746948000112"},
        ],
    }

    with patch("requests.get") as mock_get:
        mock_resp1 = MagicMock()
        mock_resp1.raise_for_status.return_value = None
        mock_resp1.json.return_value = page1

        mock_resp2 = MagicMock()
        mock_resp2.raise_for_status.return_value = None
        mock_resp2.json.return_value = page2

        mock_get.side_effect = [mock_resp1, mock_resp2]

        result = b3_corporate_actions.fetch_all_b3_listed_companies(page_size=2)

    assert len(result) == 3, f"Expected 3 total companies across 2 pages, got {len(result)}"
    assert mock_get.call_count == 2, f"Expected 2 HTTP calls for 2 pages, got {mock_get.call_count}"


def test_fetch_all_b3_listed_companies_returns_empty_on_request_error():
    """On HTTP error, fetch_all_b3_listed_companies should return empty list (not raise)."""
    import requests
    from b3_pipeline import b3_corporate_actions

    with patch("requests.get", side_effect=requests.RequestException("timeout")):
        result = b3_corporate_actions.fetch_all_b3_listed_companies()

    assert result == [], f"Expected [] on request error, got {result}"


def test_fetch_all_b3_listed_companies_returns_empty_on_json_error():
    """On JSON decode error, should return empty list."""
    import json
    from b3_pipeline import b3_corporate_actions

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
        mock_get.return_value = mock_resp

        result = b3_corporate_actions.fetch_all_b3_listed_companies()

    assert result == [], f"Expected [] on JSON error, got {result}"


def test_fetch_all_b3_listed_companies_records_have_expected_keys():
    """Each returned dict should have at least codeCVM, issuingCompany, cnpj."""
    from b3_pipeline import b3_corporate_actions

    fake_page = {
        "page": {"pageNumber": 1, "pageSize": 1, "totalPages": 1, "totalRecords": 1},
        "results": [
            {"codeCVM": "9512", "issuingCompany": "PETR", "companyName": "PETROBRAS",
             "tradingName": "PETROBRAS", "cnpj": "33000167000101", "market": "N2"},
        ],
    }

    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = fake_page
        mock_get.return_value = mock_resp

        result = b3_corporate_actions.fetch_all_b3_listed_companies()

    assert len(result) == 1
    rec = result[0]
    for key in ("codeCVM", "issuingCompany", "cnpj"):
        assert key in rec, f"Missing key '{key}' in result record"


def test_b3_company_list_url_constant_exists():
    """config.py should have B3_COMPANY_LIST_URL constant."""
    from b3_pipeline import config
    assert hasattr(config, "B3_COMPANY_LIST_URL"), (
        "B3_COMPANY_LIST_URL not found in config"
    )
    assert "{payload}" in config.B3_COMPANY_LIST_URL, (
        "B3_COMPANY_LIST_URL should contain {payload} placeholder"
    )
    assert "GetInitialCompanies" in config.B3_COMPANY_LIST_URL, (
        "B3_COMPANY_LIST_URL should point to GetInitialCompanies endpoint"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fix 1 (codeCVM path): _fetch_ticker_mappings uses update_ticker_by_cvm_code
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_ticker_mappings_uses_cvm_code_not_cnpj(mem_conn):
    """
    The new implementation must call update_ticker_by_cvm_code (via codeCVM),
    NOT upsert_cvm_company (the old CNPJ path that always got None).
    """
    from b3_pipeline import cvm_main, cvm_storage as _cvm_storage

    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    # Pre-seed cvm_companies with cvm_code so update_ticker_by_cvm_code can match
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker=None,
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name=None,
    )

    fake_company = {
        "codeCVM": 9512,  # integer, as B3 returns it
        "tradingName": "PETROBRAS",
        "cnpj": None,  # B3 API NEVER returns CNPJ here
    }

    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_all_b3_listed_companies", return_value=[]), \
         patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_company_data", return_value=fake_company):
        result = cvm_main._fetch_ticker_mappings(mem_conn)

    # Should have updated 1 row via codeCVM
    assert result == 1, f"Expected 1 updated via codeCVM path, got {result}"

    # Verify cvm_companies.ticker is now set
    cursor = mem_conn.cursor()
    cursor.execute("SELECT ticker FROM cvm_companies WHERE cnpj = '33000167000101'")
    row = cursor.fetchone()
    assert row is not None and row[0] == "PETR", f"Expected ticker='PETR', got {row!r}"


def test_fetch_ticker_mappings_uses_bulk_list_primary_path(mem_conn):
    """
    The new implementation should use fetch_all_b3_listed_companies as primary
    path and avoid per-company API calls for companies found in the bulk list.
    """
    from b3_pipeline import cvm_main

    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-01-02', 30.0)"
    )
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('VALE3', 'BRVALEACNOR0', '2024-01-02', 60.0)"
    )
    mem_conn.commit()

    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj="33000167000101", ticker=None, company_name="Petrobras",
        cvm_code="009512", b3_trading_name=None,
    )
    cvm_storage.upsert_cvm_company(
        mem_conn, cnpj="19526477000175", ticker=None, company_name="Vale",
        cvm_code="004170", b3_trading_name=None,
    )

    bulk_list = [
        {"codeCVM": "9512", "issuingCompany": "PETR", "cnpj": "33000167000101", "tradingName": "PETROBRAS"},
        {"codeCVM": "4170", "issuingCompany": "VALE", "cnpj": "19526477000175", "tradingName": "VALE"},
    ]

    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_all_b3_listed_companies", return_value=bulk_list) as mock_bulk, \
         patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_company_data") as mock_per_company:
        result = cvm_main._fetch_ticker_mappings(mem_conn)

    # bulk fetch should have been called
    mock_bulk.assert_called_once()
    # Both tickers should be mapped from bulk list — no per-company calls needed
    # (per-company calls only happen for roots NOT in the bulk list)
    assert result == 2, f"Expected 2 companies mapped from bulk list, got {result}"


# ──────────────────────────────────────────────────────────────────────────────
# Fix 4: match rate warning
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_ticker_mappings_warns_when_match_rate_zero(mem_conn, caplog):
    """
    When no tickers are matched at all (match rate = 0%), a WARNING should be logged.
    """
    from b3_pipeline import cvm_main

    # Seed prices but NO cvm_companies rows to match against
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNOR9', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    # Return empty bulk list and return None from per-company fallback
    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_all_b3_listed_companies", return_value=[]), \
         patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_company_data", return_value=None), \
         caplog.at_level(logging.WARNING, logger="b3_pipeline.cvm_main"):
        result = cvm_main._fetch_ticker_mappings(mem_conn)

    assert result == 0
    warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("match rate" in msg.lower() or "no tickers" in msg.lower() or "0" in msg
               for msg in warning_messages), (
        f"Expected a warning about low match rate, got: {warning_messages}"
    )


def test_fetch_ticker_mappings_no_warning_when_match_rate_ok(mem_conn, caplog):
    """
    When match rate is acceptable (>= 10%), no WARNING about match rate should appear.
    """
    from b3_pipeline import cvm_main

    # Seed 10 tickers, all will be matched
    for i in range(10):
        ticker = f"T{i:03d}3"
        isin = f"BRTICKER{i:04d}"
        mem_conn.execute(
            "INSERT INTO prices (ticker, isin_code, date, close) VALUES (?, ?, '2024-01-02', 10.0)",
            (ticker, isin),
        )
        cvm_storage.upsert_cvm_company(
            mem_conn, cnpj=f"0000000000{i:04d}", ticker=None,
            company_name=f"Co {i}", cvm_code=f"{i:06d}", b3_trading_name=None,
        )
    mem_conn.commit()

    bulk_list = [
        {"codeCVM": f"{i}", "issuingCompany": f"T{i:03d}", "cnpj": f"0000000000{i:04d}", "tradingName": f"T{i:03d}"}
        for i in range(10)
    ]

    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_all_b3_listed_companies", return_value=bulk_list), \
         caplog.at_level(logging.WARNING, logger="b3_pipeline.cvm_main"):
        result = cvm_main._fetch_ticker_mappings(mem_conn)

    # No "match rate" warning expected when rate is high
    match_rate_warnings = [
        r for r in caplog.records
        if r.levelno >= logging.WARNING and "match rate" in r.message.lower()
    ]
    assert len(match_rate_warnings) == 0, (
        f"Unexpected match rate warning when rate is OK: {[r.message for r in match_rate_warnings]}"
    )

"""
Tests for _fetch_ticker_mappings() in cvm_main.py (Task: ticker-fetch step).

TDD: these tests are written BEFORE the implementation and should initially FAIL.
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

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
# 1. _fetch_ticker_mappings exists and is importable
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_ticker_mappings_is_importable():
    """_fetch_ticker_mappings must exist in cvm_main."""
    from b3_pipeline import cvm_main
    assert hasattr(cvm_main, "_fetch_ticker_mappings"), (
        "_fetch_ticker_mappings not found in cvm_main"
    )
    assert callable(cvm_main._fetch_ticker_mappings)


# ──────────────────────────────────────────────────────────────────────────────
# 2. _fetch_ticker_mappings updates cvm_companies.ticker from B3 API
# ──────────────────────────────────────────────────────────────────────────────

def test_fetch_ticker_mappings_populates_ticker(mem_conn):
    """
    Given a prices table with a ticker and a cvm_companies row with matching
    CNPJ but no ticker, _fetch_ticker_mappings should set ticker via B3 API.
    """
    from b3_pipeline import cvm_main

    # Seed prices with one ticker so get_all_tickers returns something
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNPR6', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    # Seed cvm_companies with matching CNPJ but no ticker
    cvm_storage.upsert_cvm_company(
        mem_conn,
        cnpj="33000167000101",
        ticker=None,
        company_name="Petrobras",
        cvm_code="009512",
        b3_trading_name=None,
    )

    # Mock B3 API: fetch_company_data returns company dict, extract_cnpj returns CNPJ
    fake_company = {
        "companyName": "PETROBRAS",
        "cnpj": "33.000.167/0001-01",
        "codeCVM": "9512",
        "tradingName": "PETROBRAS",
    }

    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_company_data", return_value=fake_company) as mock_fetch, \
         patch("b3_pipeline.cvm_main.b3_corporate_actions.extract_cnpj_from_company_data", return_value="33000167000101") as mock_cnpj:

        result = cvm_main._fetch_ticker_mappings(mem_conn)

    # Should have updated 1 company
    assert result == 1, f"Expected 1 company updated, got {result}"

    # ticker in cvm_companies should now be set to 'PETR' (4-char root)
    cursor = mem_conn.cursor()
    cursor.execute("SELECT ticker FROM cvm_companies WHERE cnpj = '33000167000101'")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == "PETR", f"Expected ticker='PETR', got {row[0]!r}"


def test_fetch_ticker_mappings_returns_zero_when_no_tickers(mem_conn):
    """When prices table is empty, _fetch_ticker_mappings should return 0 and not crash."""
    from b3_pipeline import cvm_main

    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_company_data") as mock_fetch:
        result = cvm_main._fetch_ticker_mappings(mem_conn)

    assert result == 0, f"Expected 0 when no tickers, got {result}"
    mock_fetch.assert_not_called()


def test_fetch_ticker_mappings_skips_none_company_data(mem_conn):
    """When fetch_company_data returns None, the ticker root should be silently skipped."""
    from b3_pipeline import cvm_main

    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('VALE3', 'BRVALEACNOR0', '2024-01-02', 60.0)"
    )
    mem_conn.commit()

    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_company_data", return_value=None):
        result = cvm_main._fetch_ticker_mappings(mem_conn)

    assert result == 0, f"Expected 0 when API returns None, got {result}"


def test_fetch_ticker_mappings_skips_none_cnpj(mem_conn):
    """When extract_cnpj returns None, the company should be skipped (no upsert)."""
    from b3_pipeline import cvm_main

    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('VALE3', 'BRVALEACNOR0', '2024-01-02', 60.0)"
    )
    mem_conn.commit()

    fake_company = {"companyName": "Vale", "cnpj": "", "codeCVM": "", "tradingName": "VALE"}

    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_company_data", return_value=fake_company), \
         patch("b3_pipeline.cvm_main.b3_corporate_actions.extract_cnpj_from_company_data", return_value=None):
        result = cvm_main._fetch_ticker_mappings(mem_conn)

    assert result == 0, f"Expected 0 when CNPJ is None, got {result}"


def test_fetch_ticker_mappings_deduplicates_ticker_roots(mem_conn):
    """Multiple tickers with same 4-char root should only generate one API call."""
    from b3_pipeline import cvm_main

    for ticker, isin in [("PETR3", "BRPETRACNPR6"), ("PETR4", "BRPETRACNPR7")]:
        mem_conn.execute(
            "INSERT INTO prices (ticker, isin_code, date, close) VALUES (?, ?, '2024-01-02', 30.0)",
            (ticker, isin),
        )
    mem_conn.commit()

    fake_company = {
        "companyName": "Petrobras",
        "cnpj": "33.000.167/0001-01",
        "codeCVM": "9512",
        "tradingName": "PETROBRAS",
    }

    with patch("b3_pipeline.cvm_main.b3_corporate_actions.fetch_company_data", return_value=fake_company) as mock_fetch, \
         patch("b3_pipeline.cvm_main.b3_corporate_actions.extract_cnpj_from_company_data", return_value="33000167000101"):
        cvm_main._fetch_ticker_mappings(mem_conn)

    # PETR3 and PETR4 both have root 'PETR' -- only one call expected
    assert mock_fetch.call_count == 1, (
        f"Expected 1 API call for deduplicated roots, got {mock_fetch.call_count}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. run_fundamentals_pipeline accepts skip_ticker_fetch parameter
# ──────────────────────────────────────────────────────────────────────────────

def test_run_fundamentals_pipeline_accepts_skip_ticker_fetch():
    """run_fundamentals_pipeline must accept skip_ticker_fetch kwarg without error."""
    import inspect
    from b3_pipeline import cvm_main

    sig = inspect.signature(cvm_main.run_fundamentals_pipeline)
    assert "skip_ticker_fetch" in sig.parameters, (
        "run_fundamentals_pipeline must have a skip_ticker_fetch parameter"
    )


def test_run_fundamentals_pipeline_calls_ticker_fetch_by_default(mem_conn):
    """
    When skip_ticker_fetch is False (default), _fetch_ticker_mappings should be called.
    We mock out all I/O-heavy steps and verify the ticker-fetch step is invoked.
    """
    from b3_pipeline import cvm_main

    with patch("b3_pipeline.cvm_main.storage.get_connection", return_value=mem_conn), \
         patch("b3_pipeline.cvm_main.storage.init_db"), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_dfp_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_itr_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_fre_file", return_value=None), \
         patch("b3_pipeline.cvm_main._fetch_ticker_mappings", return_value=0) as mock_ticker_fetch, \
         patch("b3_pipeline.cvm_main.materialize_valuation_ratios", return_value=0), \
         patch("b3_pipeline.cvm_main._propagate_fre_shares"), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_tickers_from_cvm_companies", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_fundamentals_stats", return_value={
             "total_cvm_filings": 0,
             "total_cvm_companies": 0,
             "total_fundamentals_pit": 0,
         }):
        cvm_main.run_fundamentals_pipeline(
            start_year=2024, end_year=2024, skip_ticker_fetch=False
        )

    mock_ticker_fetch.assert_called_once_with(mem_conn)


def test_run_fundamentals_pipeline_skips_ticker_fetch_when_flag_set(mem_conn):
    """
    When skip_ticker_fetch=True, _fetch_ticker_mappings must NOT be called.
    """
    from b3_pipeline import cvm_main

    with patch("b3_pipeline.cvm_main.storage.get_connection", return_value=mem_conn), \
         patch("b3_pipeline.cvm_main.storage.init_db"), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_dfp_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_itr_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_fre_file", return_value=None), \
         patch("b3_pipeline.cvm_main._fetch_ticker_mappings", return_value=0) as mock_ticker_fetch, \
         patch("b3_pipeline.cvm_main.materialize_valuation_ratios", return_value=0), \
         patch("b3_pipeline.cvm_main._propagate_fre_shares"), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_tickers_from_cvm_companies", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_fundamentals_stats", return_value={
             "total_cvm_filings": 0,
             "total_cvm_companies": 0,
             "total_fundamentals_pit": 0,
         }):
        cvm_main.run_fundamentals_pipeline(
            start_year=2024, end_year=2024, skip_ticker_fetch=True
        )

    mock_ticker_fetch.assert_not_called()


# ──────────────────────────────────────────────────────────────────────────────
# 4. main.py's _run_update_cnpj_map delegates to _fetch_ticker_mappings
# ──────────────────────────────────────────────────────────────────────────────

def test_run_update_cnpj_map_delegates_to_fetch_ticker_mappings(mem_conn):
    """
    _run_update_cnpj_map in main.py should call cvm_main._fetch_ticker_mappings()
    rather than re-implementing the ticker-fetching logic itself.
    """
    from b3_pipeline import main as b3_main
    from b3_pipeline import cvm_main

    # Seed prices so get_all_tickers returns something (otherwise the function
    # early-returns before reaching _fetch_ticker_mappings)
    mem_conn.execute(
        "INSERT INTO prices (ticker, isin_code, date, close) VALUES ('PETR3', 'BRPETRACNPR6', '2024-01-02', 30.0)"
    )
    mem_conn.commit()

    # Patch sqlite3.connect to return our in-memory connection so no file I/O occurs
    import sqlite3 as _sqlite3

    class _FakeCtx:
        def __enter__(self):
            return mem_conn
        def __exit__(self, *args):
            return False

    with patch("b3_pipeline.main.storage.init_db"), \
         patch("b3_pipeline.cvm_main._fetch_ticker_mappings", return_value=3) as mock_fetch, \
         patch("sqlite3.connect", return_value=_FakeCtx()):
        b3_main._run_update_cnpj_map()

    mock_fetch.assert_called_once_with(mem_conn)

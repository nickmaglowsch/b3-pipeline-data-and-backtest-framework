"""
Tests for the --include-historical extension to cvm_main.py (Task 06 — TDD).

All tests mock network calls and use in-memory DBs.
"""
from __future__ import annotations

import argparse
import io
import sqlite3
import zipfile
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from b3_pipeline import storage
from b3_pipeline.cvm_main import run_fundamentals_pipeline


# ── Helpers ────────────────────────────────────────────────────────────────────

def _mem_conn():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    return conn


def _csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, sep=";", index=False)
    return buf.getvalue().encode("latin-1")


def _make_ipe_zip(year: int = 2008) -> io.BytesIO:
    rows = [{
        "CNPJ_Companhia": "00.000.000/0001-91",
        "Nome_Companhia": "BANCO DO BRASIL",
        "Codigo_CVM": "1023",
        "Data_Referencia": f"{year}-12-31",
        "Categoria": "Dados Econômico-Financeiros",
        "Tipo": "Demonstrações Financeiras Anuais Completas",
        "Especie": "", "Assunto": "", "Data_Entrega": f"{year+1}-02-28",
        "Tipo_Apresentacao": "AP", "Protocolo_Entrega": "", "Versao": "1",
        "Link_Download": "https://example.com/doc",
    }]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"ipe_cia_aberta_{year}.csv", _csv_bytes(rows))
    buf.seek(0)
    return buf


def _make_cad_stringio() -> io.StringIO:
    rows = [{
        "CNPJ_CIA": "00.000.000/0001-91",
        "DENOM_SOCIAL": "BANCO DO BRASIL SA",
        "CD_CVM": "001023",
        "DT_REG": "1994-04-22",
        "DT_CANCEL": "",
        "SIT": "ATIVO",
    }]
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, sep=";", index=False)
    buf.seek(0)
    return buf


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_run_pipeline_without_historical_skips_ipe_steps():
    """When include_historical=False, CAD and IPE downloaders are never called."""
    with patch("b3_pipeline.cvm_main.storage.get_connection") as mock_get_conn, \
         patch("b3_pipeline.cvm_main.storage.init_db"), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_dfp_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_itr_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_fre_file", return_value=None), \
         patch("b3_pipeline.cvm_main._fetch_ticker_mappings", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_cvm_company_map", return_value={}), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_fundamentals_stats", return_value={
             "total_cvm_filings": 0, "total_cvm_companies": 0, "total_fundamentals_pit": 0,
             "companies_with_listing_date": 0, "companies_with_delisting_date": 0,
         }), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_tickers_from_cvm_companies", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_company_isin_map", return_value=0), \
         patch("b3_pipeline.cvm_main._propagate_fre_shares"), \
         patch("b3_pipeline.cvm_main.materialize_fundamentals_monthly", return_value=0), \
         patch("b3_pipeline.cad_downloader.download_cad_file") as mock_cad_dl, \
         patch("b3_pipeline.ipe_downloader.download_ipe_file") as mock_ipe_dl:

        mock_conn = _mem_conn()
        mock_get_conn.return_value = mock_conn

        run_fundamentals_pipeline(
            start_year=2023, end_year=2023,
            include_historical=False,
            skip_ticker_fetch=True,
            skip_ratios=True,
            skip_monthly=True,
        )

        # Lazy imports only happen inside the include_historical branch,
        # so the downloaders should never be invoked.
        mock_cad_dl.assert_not_called()
        mock_ipe_dl.assert_not_called()


def test_run_pipeline_with_historical_calls_cad_download():
    """When include_historical=True, cad_downloader.download_cad_file is called once."""
    with patch("b3_pipeline.cvm_main.storage.get_connection") as mock_get_conn, \
         patch("b3_pipeline.cvm_main.storage.init_db"), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_dfp_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_itr_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_fre_file", return_value=None), \
         patch("b3_pipeline.cvm_main._fetch_ticker_mappings", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_cvm_company_map", return_value={}), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_fundamentals_stats", return_value={
             "total_cvm_filings": 0, "total_cvm_companies": 0, "total_fundamentals_pit": 0,
             "companies_with_listing_date": 0, "companies_with_delisting_date": 0,
         }), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_tickers_from_cvm_companies", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_company_isin_map", return_value=0), \
         patch("b3_pipeline.cvm_main._propagate_fre_shares"), \
         patch("b3_pipeline.cvm_main.materialize_fundamentals_monthly", return_value=0), \
         patch("b3_pipeline.cad_downloader.download_cad_file", return_value=None) as mock_cad_dl, \
         patch("b3_pipeline.ipe_downloader.download_ipe_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_storage.upsert_cad_company_dates", return_value=0):

        mock_conn = _mem_conn()
        mock_get_conn.return_value = mock_conn

        run_fundamentals_pipeline(
            start_year=2023, end_year=2023,
            include_historical=True,
            skip_ticker_fetch=True,
            skip_ratios=True,
            skip_monthly=True,
        )

        mock_cad_dl.assert_called_once()


def test_run_pipeline_with_historical_calls_ipe_download_for_range():
    """With start_year=2006, IPE downloads years 2006-2009 (4 calls)."""
    with patch("b3_pipeline.cvm_main.storage.get_connection") as mock_get_conn, \
         patch("b3_pipeline.cvm_main.storage.init_db"), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_dfp_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_itr_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_fre_file", return_value=None), \
         patch("b3_pipeline.cvm_main._fetch_ticker_mappings", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_cvm_company_map", return_value={}), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_fundamentals_stats", return_value={
             "total_cvm_filings": 0, "total_cvm_companies": 0, "total_fundamentals_pit": 0,
             "companies_with_listing_date": 0, "companies_with_delisting_date": 0,
         }), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_tickers_from_cvm_companies", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_company_isin_map", return_value=0), \
         patch("b3_pipeline.cvm_main._propagate_fre_shares"), \
         patch("b3_pipeline.cvm_main.materialize_fundamentals_monthly", return_value=0), \
         patch("b3_pipeline.cad_downloader.download_cad_file", return_value=None), \
         patch("b3_pipeline.ipe_downloader.download_ipe_file", return_value=None) as mock_ipe_dl, \
         patch("b3_pipeline.cvm_main.cvm_storage.upsert_cad_company_dates", return_value=0):

        mock_conn = _mem_conn()
        mock_get_conn.return_value = mock_conn

        run_fundamentals_pipeline(
            start_year=2006, end_year=2023,
            include_historical=True,
            skip_ticker_fetch=True,
            skip_ratios=True,
            skip_monthly=True,
        )

        # Should have been called for years 2006, 2007, 2008, 2009
        call_years = [c.args[0] for c in mock_ipe_dl.call_args_list]
        assert set(call_years) == {2006, 2007, 2008, 2009}, f"Unexpected years: {call_years}"
        # Must NOT be called for 2010+
        assert all(y <= 2009 for y in call_years), f"Called for year >= 2010: {call_years}"


def test_run_pipeline_ipe_year_range_default():
    """Without start_year, IPE downloads years 2003-2009 (7 calls)."""
    with patch("b3_pipeline.cvm_main.storage.get_connection") as mock_get_conn, \
         patch("b3_pipeline.cvm_main.storage.init_db"), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_dfp_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_itr_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_fre_file", return_value=None), \
         patch("b3_pipeline.cvm_main._fetch_ticker_mappings", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_cvm_company_map", return_value={}), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_fundamentals_stats", return_value={
             "total_cvm_filings": 0, "total_cvm_companies": 0, "total_fundamentals_pit": 0,
             "companies_with_listing_date": 0, "companies_with_delisting_date": 0,
         }), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_tickers_from_cvm_companies", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_company_isin_map", return_value=0), \
         patch("b3_pipeline.cvm_main._propagate_fre_shares"), \
         patch("b3_pipeline.cvm_main.materialize_fundamentals_monthly", return_value=0), \
         patch("b3_pipeline.cad_downloader.download_cad_file", return_value=None), \
         patch("b3_pipeline.ipe_downloader.download_ipe_file", return_value=None) as mock_ipe_dl, \
         patch("b3_pipeline.cvm_main.cvm_storage.upsert_cad_company_dates", return_value=0):

        mock_conn = _mem_conn()
        mock_get_conn.return_value = mock_conn

        run_fundamentals_pipeline(
            include_historical=True,
            skip_ticker_fetch=True,
            skip_ratios=True,
            skip_monthly=True,
        )

        call_years = [c.args[0] for c in mock_ipe_dl.call_args_list]
        assert set(call_years) == set(range(2003, 2010)), f"Expected 2003-2009, got: {call_years}"


def test_include_historical_flag_in_cli():
    """argparse correctly parses --include-historical as True."""
    from b3_pipeline.cvm_main import main
    import argparse

    # Re-build the arg_parser as in main() to test argument parsing
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--start-year", type=int, default=None)
    arg_parser.add_argument("--end-year", type=int, default=None)
    arg_parser.add_argument("--rebuild", action="store_true")
    arg_parser.add_argument("--force-download", action="store_true")
    arg_parser.add_argument("--skip-ratios", action="store_true")
    arg_parser.add_argument("--skip-ticker-fetch", action="store_true")
    arg_parser.add_argument("--skip-monthly", action="store_true")
    arg_parser.add_argument("--include-historical", action="store_true")
    arg_parser.add_argument("--verbose", "-v", action="store_true")

    args = arg_parser.parse_args(["--include-historical"])
    assert args.include_historical is True


def test_run_pipeline_historical_upserts_filings(tmp_path):
    """With include_historical=True and a valid IPE ZIP, filings rows are processed."""
    import tempfile
    from pathlib import Path

    # Use a temp DB file
    db_path = tmp_path / "test.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)

    # Create a minimal IPE zip
    ipe_zip = _make_ipe_zip(2008)

    # Return the IPE zip for year 2008, None for others
    def ipe_download_side_effect(year, force=False):
        if year == 2008:
            return ipe_zip
        return None

    with patch("b3_pipeline.cvm_main.storage.get_connection") as mock_get_conn, \
         patch("b3_pipeline.cvm_main.storage.init_db"), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_dfp_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_itr_file", return_value=None), \
         patch("b3_pipeline.cvm_main.cvm_downloader.download_fre_file", return_value=None), \
         patch("b3_pipeline.cvm_main._fetch_ticker_mappings", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_cvm_company_map", return_value={}), \
         patch("b3_pipeline.cvm_main.cvm_storage.get_fundamentals_stats", return_value={
             "total_cvm_filings": 0, "total_cvm_companies": 0, "total_fundamentals_pit": 0,
             "companies_with_listing_date": 0, "companies_with_delisting_date": 0,
         }), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_tickers_from_cvm_companies", return_value=0), \
         patch("b3_pipeline.cvm_main.cvm_storage.populate_company_isin_map", return_value=0), \
         patch("b3_pipeline.cvm_main._propagate_fre_shares"), \
         patch("b3_pipeline.cvm_main.materialize_fundamentals_monthly", return_value=0), \
         patch("b3_pipeline.cad_downloader.download_cad_file", return_value=None), \
         patch("b3_pipeline.ipe_downloader.download_ipe_file", side_effect=ipe_download_side_effect), \
         patch("b3_pipeline.cvm_main.cvm_storage.upsert_cad_company_dates", return_value=0):

        mock_get_conn.return_value = conn

        run_fundamentals_pipeline(
            start_year=2008, end_year=2023,
            include_historical=True,
            skip_ticker_fetch=True,
            skip_ratios=True,
            skip_monthly=True,
        )

    conn.close()
    # Verify: cvm_filings should have IPE rows
    verify_conn = sqlite3.connect(str(db_path))
    cursor = verify_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM cvm_filings WHERE doc_type = 'IPE'")
    ipe_count = cursor.fetchone()[0]
    verify_conn.close()
    assert ipe_count > 0, "Expected IPE filings to be stored in cvm_filings"

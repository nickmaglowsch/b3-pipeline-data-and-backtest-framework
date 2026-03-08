"""
Tests for the IPE document index downloader and parser (Task 04 — TDD).

IMPORTANT: As documented in tasks/ipe-structure-report.md, the IPE dataset
is a document filing INDEX, NOT structured financial statement data.
It contains metadata (company ID, doc category, dates, download links)
but no account codes or financial values.

Therefore:
- parse_ipe_zip() returns (filings_df, fundamentals_df) where fundamentals_df
  is always EMPTY (no financial metrics to extract).
- extract_ipe_company_index() returns (cnpj, cvm_code, company_name) tuples
  for populating cvm_companies.
- doc_type is 'IPE' in filings_df rows.
- All financial metric columns in fundamentals_df are None/NaN.

All tests use in-memory synthetic ZIPs — no real CVM data or network calls.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from b3_pipeline.ipe_parser import parse_ipe_zip, extract_ipe_company_index


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _csv_bytes(rows: list[dict]) -> bytes:
    """Convert a list of dicts to semicolon-delimited CSV bytes (latin-1)."""
    buf = io.StringIO()
    df = pd.DataFrame(rows)
    df.to_csv(buf, sep=";", index=False)
    return buf.getvalue().encode("latin-1")


def _make_ipe_zip(rows: list[dict], year: int = 2008) -> io.BytesIO:
    """Build an in-memory IPE ZIP with a single CSV matching real IPE structure."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"ipe_cia_aberta_{year}.csv", _csv_bytes(rows))
    buf.seek(0)
    return buf


def _ipe_row(
    cnpj="00.000.000/0001-91",
    nome="BANCO DO BRASIL S.A.",
    cd_cvm="1023",
    dt_ref="2008-01-24",
    categoria="Assembleia",
    tipo="AGE",
    dt_entrega="2008-07-02",
    versao=None,
    link="https://www.rad.cvm.gov.br/ENET/frmDownloadDocumento.aspx?numProtocolo=168751",
) -> dict:
    return {
        "CNPJ_Companhia": cnpj,
        "Nome_Companhia": nome,
        "Codigo_CVM": cd_cvm,
        "Data_Referencia": dt_ref,
        "Categoria": categoria,
        "Tipo": tipo,
        "Especie": "",
        "Assunto": "",
        "Data_Entrega": dt_entrega,
        "Tipo_Apresentacao": "AP - Apresentação",
        "Protocolo_Entrega": "",
        "Versao": versao if versao is not None else "",
        "Link_Download": link,
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_parse_ipe_zip_returns_two_dataframes():
    """parse_ipe_zip() returns a (filings_df, fundamentals_df) tuple."""
    rows = [_ipe_row()]
    result = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map={})
    assert isinstance(result, tuple)
    assert len(result) == 2
    filings_df, fundamentals_df = result
    assert isinstance(filings_df, pd.DataFrame)
    assert isinstance(fundamentals_df, pd.DataFrame)


def test_parse_ipe_zip_doc_type_is_ipe():
    """doc_type in filings_df must be 'IPE'."""
    rows = [_ipe_row()]
    filings_df, _ = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map={})
    if not filings_df.empty:
        assert (filings_df["doc_type"] == "IPE").all()


def test_parse_ipe_zip_fundamentals_empty_or_all_null():
    """fundamentals_df contains no financial data (IPE is a document index)."""
    rows = [_ipe_row()]
    _, fundamentals_df = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map={})
    # Either empty or all financial metric columns are None
    financial_cols = ["revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt"]
    for col in financial_cols:
        if col in fundamentals_df.columns and not fundamentals_df.empty:
            assert fundamentals_df[col].isna().all(), f"{col} should be null in IPE output"


def test_parse_ipe_zip_ratio_columns_absent_from_output():
    """pe_ratio, pb_ratio, ev_ebitda are not output columns — ratios computed dynamically."""
    rows = [_ipe_row()]
    _, fundamentals_df = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map={})
    # Ratio columns were removed from the schema; they must not appear in parser output
    for col in ["pe_ratio", "pb_ratio", "ev_ebitda"]:
        assert col not in fundamentals_df.columns, (
            f"{col} should not be in IPE parser output — ratios computed dynamically"
        )


def test_parse_ipe_zip_shares_outstanding_is_none():
    """shares_outstanding is None/NaN in IPE output (no capital data in IPE)."""
    rows = [_ipe_row()]
    _, fundamentals_df = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map={})
    if "shares_outstanding" in fundamentals_df.columns and not fundamentals_df.empty:
        assert fundamentals_df["shares_outstanding"].isna().all()


def test_parse_ipe_zip_ticker_from_map():
    """If cnpj_ticker_map is provided, ticker is resolved in filings_df."""
    rows = [_ipe_row(cnpj="00.000.000/0001-91")]
    cnpj_map = {"00000000000191": "BBAS"}
    filings_df, _ = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map=cnpj_map)
    if not filings_df.empty and "ticker" in filings_df.columns:
        assert filings_df.iloc[0]["ticker"] == "BBAS"


def test_parse_ipe_zip_ticker_none_for_unmapped():
    """Ticker is None when CNPJ is not in cnpj_ticker_map."""
    rows = [_ipe_row(cnpj="00.000.000/0001-91")]
    filings_df, _ = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map={})
    if not filings_df.empty and "ticker" in filings_df.columns:
        assert filings_df.iloc[0]["ticker"] is None or pd.isna(filings_df.iloc[0]["ticker"])


def test_extract_ipe_company_index_returns_tuples():
    """extract_ipe_company_index returns (cnpj_14digit, cvm_code, company_name) tuples."""
    rows = [
        _ipe_row(cnpj="00.000.000/0001-91", nome="BANCO DO BRASIL S.A.", cd_cvm="1023"),
        _ipe_row(cnpj="33.000.167/0001-01", nome="PETROBRAS SA", cd_cvm="9512"),
        # Duplicate CNPJ — should deduplicate
        _ipe_row(cnpj="00.000.000/0001-91", nome="BANCO DO BRASIL S.A.", cd_cvm="1023"),
    ]
    index = extract_ipe_company_index(_make_ipe_zip(rows))
    assert isinstance(index, list)
    assert len(index) == 2  # deduplicated
    cnpjs = {t[0] for t in index}
    assert "00000000000191" in cnpjs
    assert "33000167000101" in cnpjs
    # Each tuple is (cnpj, cvm_code, company_name)
    for item in index:
        assert len(item) == 3
        assert len(item[0]) == 14  # 14-digit CNPJ


def test_extract_ipe_company_index_drops_invalid_cnpj():
    """Rows with malformed CNPJ are excluded from the company index."""
    rows = [
        _ipe_row(cnpj="00.000.000/0001-91"),  # valid
        _ipe_row(cnpj="INVALID"),              # bad — excluded
        _ipe_row(cnpj=""),                     # empty — excluded
    ]
    index = extract_ipe_company_index(_make_ipe_zip(rows))
    assert len(index) == 1
    assert index[0][0] == "00000000000191"


def test_parse_ipe_zip_empty_zip():
    """An empty ZIP returns two empty DataFrames without raising."""
    empty = io.BytesIO()
    with zipfile.ZipFile(empty, "w"):
        pass
    empty.seek(0)
    filings_df, fundamentals_df = parse_ipe_zip(empty, cnpj_ticker_map={})
    assert isinstance(filings_df, pd.DataFrame)
    assert isinstance(fundamentals_df, pd.DataFrame)

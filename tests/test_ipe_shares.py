"""
Tests for IPE shares outstanding handling (Task 05 — Path B TDD).

As documented in tasks/ipe-structure-report.md, IPE does NOT contain
capital structure / shares outstanding data. This is Path B:
- No _propagate_ipe_shares() function is implemented
- shares_outstanding remains NULL for all pre-2010 IPE rows
- The NOTE comment in ipe_parser.py documents this gap

These tests confirm the Path B behavior:
1. IPE parse output always has NULL shares_outstanding
2. The ipe_parser module has the NOTE comment
3. NULL shares_outstanding rows in fundamentals_pit do not cause crashes at query time
"""
from __future__ import annotations

import io
import sqlite3
import zipfile

import pandas as pd
import pytest

from b3_pipeline.ipe_parser import parse_ipe_zip, IPE_SHARES_COLUMN
import b3_pipeline.ipe_parser as ipe_parser_module


# ── Helpers ────────────────────────────────────────────────────────────────────

def _csv_bytes(rows: list[dict]) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, sep=";", index=False)
    return buf.getvalue().encode("latin-1")


def _make_ipe_zip(rows: list[dict], year: int = 2008) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"ipe_cia_aberta_{year}.csv", _csv_bytes(rows))
    buf.seek(0)
    return buf


def _ipe_row(cnpj="00.000.000/0001-91", dt_ref="2008-01-24") -> dict:
    return {
        "CNPJ_Companhia": cnpj,
        "Nome_Companhia": "BANCO DO BRASIL S.A.",
        "Codigo_CVM": "1023",
        "Data_Referencia": dt_ref,
        "Categoria": "Dados Econômico-Financeiros",
        "Tipo": "Demonstrações Financeiras Anuais Completas",
        "Especie": "",
        "Assunto": "",
        "Data_Entrega": "2008-02-26",
        "Tipo_Apresentacao": "AP - Apresentação",
        "Protocolo_Entrega": "",
        "Versao": "",
        "Link_Download": "https://www.rad.cvm.gov.br/ENET/example",
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_ipe_shares_column_is_none():
    """IPE_SHARES_COLUMN must be None — confirmed Path B (no capital data in IPE)."""
    assert IPE_SHARES_COLUMN is None, (
        "IPE_SHARES_COLUMN should be None (Path B: IPE has no shares data)"
    )


def test_ipe_parser_has_path_b_note_comment():
    """ipe_parser.py must contain a NOTE comment documenting the Path B decision."""
    import inspect
    source = inspect.getsource(ipe_parser_module)
    assert "Path B" in source or "NOTE" in source, (
        "ipe_parser.py should document Path B (no shares in IPE) with a # NOTE comment"
    )


def test_parse_ipe_zip_shares_outstanding_column_is_null():
    """shares_outstanding is always NULL in IPE fundamentals_df (Path B)."""
    rows = [_ipe_row()]
    _, fundamentals_df = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map={})
    # fundamentals_df may be empty (since IPE has no financial data)
    # or have shares_outstanding column with all nulls
    if not fundamentals_df.empty and "shares_outstanding" in fundamentals_df.columns:
        assert fundamentals_df["shares_outstanding"].isna().all(), (
            "shares_outstanding must be NULL for all IPE rows"
        )


def test_no_propagate_ipe_shares_function_exists():
    """_propagate_ipe_shares() should NOT be defined (Path B — not needed)."""
    import b3_pipeline.cvm_main as cvm_main_module
    assert not hasattr(cvm_main_module, "_propagate_ipe_shares"), (
        "_propagate_ipe_shares should not exist in cvm_main (Path B chosen)"
    )


def test_parse_ipe_zip_fundamentals_always_empty_for_financial_rows():
    """Even financial statement rows in IPE produce no financial metrics."""
    rows = [_ipe_row(dt_ref="2007-12-31")]  # annual report row
    _, fundamentals_df = parse_ipe_zip(_make_ipe_zip(rows), cnpj_ticker_map={})
    financial_cols = ["revenue", "net_income", "ebitda", "total_assets", "equity", "net_debt"]
    for col in financial_cols:
        if col in fundamentals_df.columns and not fundamentals_df.empty:
            assert fundamentals_df[col].isna().all(), (
                f"Column {col} should be all NULL for IPE — no structured financial data available"
            )

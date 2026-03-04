"""
Tests for the CVM CSV parser (Task 04 — TDD).

All tests use in-memory synthetic ZIPs — no real CVM data or network calls.
"""
from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from b3_pipeline.cvm_parser import parse_dfp_zip, parse_itr_zip, parse_fre_zip


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _csv_bytes(rows: list[dict]) -> bytes:
    """Convert a list of dicts to semicolon-delimited CSV bytes (latin-1)."""
    buf = io.StringIO()
    df = pd.DataFrame(rows)
    df.to_csv(buf, sep=";", index=False)
    return buf.getvalue().encode("latin-1")


def _make_fake_dfp_zip(
    dre_rows: list[dict],
    bpa_rows: list[dict],
    bpp_rows: list[dict],
) -> io.BytesIO:
    """Build an in-memory ZIP with synthetic DRE/BPA/BPP CSVs."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("dfp_cia_aberta_DRE_con_2023.csv", _csv_bytes(dre_rows))
        zf.writestr("dfp_cia_aberta_BPA_con_2023.csv", _csv_bytes(bpa_rows))
        zf.writestr("dfp_cia_aberta_BPP_con_2023.csv", _csv_bytes(bpp_rows))
    buf.seek(0)
    return buf


def _make_fake_itr_zip(dre_rows, bpa_rows, bpp_rows) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("itr_cia_aberta_DRE_con_2023.csv", _csv_bytes(dre_rows))
        zf.writestr("itr_cia_aberta_BPA_con_2023.csv", _csv_bytes(bpa_rows))
        zf.writestr("itr_cia_aberta_BPP_con_2023.csv", _csv_bytes(bpp_rows))
    buf.seek(0)
    return buf


def _make_fake_fre_zip(capital_rows: list[dict]) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("fre_cia_aberta_capital_social_2023.csv", _csv_bytes(capital_rows))
    buf.seek(0)
    return buf


# ── Shared synthetic data ──────────────────────────────────────────────────────

BASE_DRE = [
    {
        "CNPJ_CIA": "33.000.167/0001-01",
        "DT_REFER": "2023-12-31",
        "DT_RECEB": "2024-03-15",
        "VERSAO": "1",
        "ORDEM_EXERC": "ÚLTIMO",
        "CD_CONTA": "3.01",
        "VL_CONTA": "1000000",
    }
]

BASE_BPA = [
    {
        "CNPJ_CIA": "33.000.167/0001-01",
        "DT_REFER": "2023-12-31",
        "DT_RECEB": "2024-03-15",
        "VERSAO": "1",
        "ORDEM_EXERC": "ÚLTIMO",
        "CD_CONTA": "1",
        "VL_CONTA": "5000000",
    }
]

BASE_BPP = [
    {
        "CNPJ_CIA": "33.000.167/0001-01",
        "DT_REFER": "2023-12-31",
        "DT_RECEB": "2024-03-15",
        "VERSAO": "1",
        "ORDEM_EXERC": "ÚLTIMO",
        "CD_CONTA": "2.03",
        "VL_CONTA": "2000000",
    }
]


# ──────────────────────────────────────────────────────────────────────────────
# 1. Revenue extraction
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_zip_extracts_revenue():
    """Revenue from account 3.01 should be parsed correctly."""
    fake_zip = _make_fake_dfp_zip(BASE_DRE, BASE_BPA, BASE_BPP)
    cnpj_map = {"33000167000101": "PETR"}
    filings_df, fundamentals_df = parse_dfp_zip(fake_zip, cnpj_map)

    assert not fundamentals_df.empty, "fundamentals_df should not be empty"
    assert "revenue" in fundamentals_df.columns
    assert fundamentals_df["revenue"].iloc[0] == pytest.approx(1_000_000.0), (
        f"Expected revenue 1000000, got {fundamentals_df['revenue'].iloc[0]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Ticker mapping
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_zip_sets_ticker_from_map():
    """Ticker should be resolved from cnpj_ticker_map."""
    fake_zip = _make_fake_dfp_zip(BASE_DRE, BASE_BPA, BASE_BPP)
    filings_df, fundamentals_df = parse_dfp_zip(fake_zip, {"33000167000101": "PETR"})

    assert fundamentals_df["ticker"].iloc[0] == "PETR", (
        f"Expected ticker PETR, got {fundamentals_df['ticker'].iloc[0]!r}"
    )


def test_parse_dfp_zip_ticker_none_for_unmapped():
    """Unmapped CNPJ should result in None/NaN ticker."""
    fake_zip = _make_fake_dfp_zip(BASE_DRE, BASE_BPA, BASE_BPP)
    filings_df, fundamentals_df = parse_dfp_zip(fake_zip, {})  # empty map

    ticker_val = fundamentals_df["ticker"].iloc[0]
    assert ticker_val is None or pd.isna(ticker_val), (
        f"Expected None/NaN ticker for unmapped CNPJ, got {ticker_val!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Net debt computation
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_zip_computes_net_debt():
    """net_debt = short_debt + long_debt - cash."""
    dre_rows = BASE_DRE
    bpa_rows = [
        {
            "CNPJ_CIA": "33.000.167/0001-01",
            "DT_REFER": "2023-12-31",
            "DT_RECEB": "2024-03-15",
            "VERSAO": "1",
            "ORDEM_EXERC": "ÚLTIMO",
            "CD_CONTA": "1.01.01",  # cash = 500
            "VL_CONTA": "500",
        }
    ]
    bpp_rows = [
        {
            "CNPJ_CIA": "33.000.167/0001-01",
            "DT_REFER": "2023-12-31",
            "DT_RECEB": "2024-03-15",
            "VERSAO": "1",
            "ORDEM_EXERC": "ÚLTIMO",
            "CD_CONTA": "2.01.04",  # short_debt = 200
            "VL_CONTA": "200",
        },
        {
            "CNPJ_CIA": "33.000.167/0001-01",
            "DT_REFER": "2023-12-31",
            "DT_RECEB": "2024-03-15",
            "VERSAO": "1",
            "ORDEM_EXERC": "ÚLTIMO",
            "CD_CONTA": "2.02.01",  # long_debt = 300
            "VL_CONTA": "300",
        },
    ]
    fake_zip = _make_fake_dfp_zip(dre_rows, bpa_rows, bpp_rows)
    filings_df, fundamentals_df = parse_dfp_zip(fake_zip, {"33000167000101": "PETR"})

    assert "net_debt" in fundamentals_df.columns
    net_debt = fundamentals_df["net_debt"].iloc[0]
    assert net_debt == pytest.approx(0.0), (
        f"Expected net_debt = 200 + 300 - 500 = 0.0, got {net_debt}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Filter PENÚLTIMO rows
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_filters_penultimo():
    """ORDEM_EXERC == PENÚLTIMO rows should be excluded."""
    dre_rows = [
        {
            "CNPJ_CIA": "33.000.167/0001-01",
            "DT_REFER": "2023-12-31",
            "DT_RECEB": "2024-03-15",
            "VERSAO": "1",
            "ORDEM_EXERC": "ÚLTIMO",
            "CD_CONTA": "3.01",
            "VL_CONTA": "1000",
        },
        {
            "CNPJ_CIA": "33.000.167/0001-01",
            "DT_REFER": "2023-12-31",
            "DT_RECEB": "2024-03-15",
            "VERSAO": "1",
            "ORDEM_EXERC": "PENÚLTIMO",
            "CD_CONTA": "3.01",
            "VL_CONTA": "800",
        },
    ]
    fake_zip = _make_fake_dfp_zip(dre_rows, BASE_BPA, BASE_BPP)
    filings_df, fundamentals_df = parse_dfp_zip(fake_zip, {"33000167000101": "PETR"})

    assert len(fundamentals_df) == 1, (
        f"Expected 1 row after filtering PENÚLTIMO, got {len(fundamentals_df)}"
    )
    assert fundamentals_df["revenue"].iloc[0] == pytest.approx(1000.0), (
        "Should keep ÚLTIMO value (1000), not PENÚLTIMO (800)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6. filing_id format
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_dfp_filing_id_format():
    """filing_id should be f'{cnpj}_DFP_{period_end}_{version}'."""
    fake_zip = _make_fake_dfp_zip(BASE_DRE, BASE_BPA, BASE_BPP)
    filings_df, fundamentals_df = parse_dfp_zip(fake_zip, {"33000167000101": "PETR"})

    expected_id = "33000167000101_DFP_2023-12-31_1"
    assert fundamentals_df["filing_id"].iloc[0] == expected_id, (
        f"Expected filing_id {expected_id!r}, got {fundamentals_df['filing_id'].iloc[0]!r}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7. ITR quarter inference
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_itr_sets_quarter():
    """For ITR with period_end in March, quarter should be 1."""
    itr_dre = [
        {
            "CNPJ_CIA": "33.000.167/0001-01",
            "DT_REFER": "2023-03-31",  # Q1
            "DT_RECEB": "2023-05-15",
            "VERSAO": "1",
            "ORDEM_EXERC": "ÚLTIMO",
            "CD_CONTA": "3.01",
            "VL_CONTA": "250000",
        }
    ]
    itr_bpa = [{
        "CNPJ_CIA": "33.000.167/0001-01",
        "DT_REFER": "2023-03-31",
        "DT_RECEB": "2023-05-15",
        "VERSAO": "1",
        "ORDEM_EXERC": "ÚLTIMO",
        "CD_CONTA": "1",
        "VL_CONTA": "5000000",
    }]
    itr_bpp = [{
        "CNPJ_CIA": "33.000.167/0001-01",
        "DT_REFER": "2023-03-31",
        "DT_RECEB": "2023-05-15",
        "VERSAO": "1",
        "ORDEM_EXERC": "ÚLTIMO",
        "CD_CONTA": "2.03",
        "VL_CONTA": "2000000",
    }]
    fake_zip = _make_fake_itr_zip(itr_dre, itr_bpa, itr_bpp)
    filings_df, fundamentals_df = parse_itr_zip(fake_zip, {"33000167000101": "PETR"})

    assert not fundamentals_df.empty
    assert fundamentals_df["quarter"].iloc[0] == 1, (
        f"Expected quarter=1 for March period_end, got {fundamentals_df['quarter'].iloc[0]}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# 8. FRE shares outstanding
# ──────────────────────────────────────────────────────────────────────────────

def test_parse_fre_zip_extracts_shares():
    """Quantidade_Total_Acoes should be parsed as shares_outstanding float."""
    capital_rows = [
        {
            # Real CVM capital_social CSV column names
            "CNPJ_Companhia": "33.000.167/0001-01",
            "Data_Referencia": "2023-12-31",
            "DT_RECEB": "2024-02-10",
            "Versao": "1",
            "Quantidade_Total_Acoes": "1000000000",
            "Quantidade_Acoes_Ordinarias": "800000000",
            "Quantidade_Acoes_Preferenciais": "200000000",
        }
    ]
    fake_zip = _make_fake_fre_zip(capital_rows)
    filings_df, fundamentals_df = parse_fre_zip(fake_zip, {"33000167000101": "PETR"})

    assert not fundamentals_df.empty, "fundamentals_df should not be empty"
    assert "shares_outstanding" in fundamentals_df.columns
    shares = fundamentals_df["shares_outstanding"].iloc[0]
    assert shares == pytest.approx(1_000_000_000.0), (
        f"Expected 1_000_000_000.0, got {shares}"
    )

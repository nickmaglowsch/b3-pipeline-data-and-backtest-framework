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


def _cap_row(tipo, total, ord_, pref, aut_date, versao="1"):
    return {
        "CNPJ_Companhia": "33.000.167/0001-01",
        "Data_Referencia": "2023-12-31",
        "DT_RECEB": "2024-02-10",
        "Versao": versao,
        "Tipo_Capital": tipo,
        "Data_Autorizacao_Aprovacao": aut_date,
        "Quantidade_Total_Acoes": str(total),
        "Quantidade_Acoes_Ordinarias": str(ord_),
        "Quantidade_Acoes_Preferenciais": str(pref),
    }


def test_parse_fre_zip_ignores_capital_autorizado():
    """Regression: COCE/ITUB stored 'Capital Autorizado' (an issuance ceiling,
    e.g. 300bn shares) instead of the paid-in count. Integralizado must win."""
    rows = [
        _cap_row("Capital Emitido", 77_855_299, 48_067_937, 29_787_362, "2017-04-01"),
        _cap_row("Capital Subscrito", 77_855_299, 48_067_937, 29_787_362, "2017-04-01"),
        _cap_row("Capital Integralizado", 77_855_299, 48_067_937, 29_787_362, "2017-04-01"),
        _cap_row("Capital Autorizado", 300_000_000_000, 100_000_000_000, 200_000_000_000, "2017-04-01"),
    ]
    _, fund = parse_fre_zip(_make_fake_fre_zip(rows), {"33000167000101": "PETR"})
    assert len(fund) == 1
    assert fund["shares_outstanding"].iloc[0] == pytest.approx(77_855_299.0)
    assert fund["shares_on"].iloc[0] == pytest.approx(48_067_937.0)
    assert fund["shares_pn"].iloc[0] == pytest.approx(29_787_362.0)


def test_parse_fre_zip_duplicate_integralizado_latest_authorization_wins():
    """Regression: TELB filed its capital twice in one FRE — the stale entry
    fat-fingered x10000. The row with the latest authorization date is current."""
    rows = [
        _cap_row("Capital Integralizado", 1_096_989_129_010, 886_959_131_950,
                 210_029_997_060, "2009-02-19", versao="19"),
        _cap_row("Capital Integralizado", 109_698_912, 88_695_913,
                 21_002_999, "2010-12-03", versao="19"),
    ]
    _, fund = parse_fre_zip(_make_fake_fre_zip(rows), {"33000167000101": "TELB"})
    assert len(fund) == 1
    assert fund["shares_outstanding"].iloc[0] == pytest.approx(109_698_912.0)


# ──────────────────────────────────────────────────────────────────────────────
# 9. Chart-of-accounts drift: EBIT by description, bank chart → NULLs
# ──────────────────────────────────────────────────────────────────────────────

def _dre_row(cd, ds, val, cnpj="33.000.167/0001-01"):
    return {
        "CNPJ_CIA": cnpj,
        "DT_REFER": "2023-12-31",
        "DT_RECEB": "2024-03-15",
        "VERSAO": "1",
        "ORDEM_EXERC": "ÚLTIMO",
        "CD_CONTA": cd,
        "DS_CONTA": ds,
        "VL_CONTA": str(val),
    }


def test_corporate_ebitda_selected_by_description_at_3_05():
    """Corporate chart: EBIT line sits at 3.05."""
    dre = [
        _dre_row("3.01", "Receita de Venda de Bens e/ou Serviços", 32503601),
        _dre_row("3.05", "Resultado Antes do Resultado Financeiro e dos Tributos", 6462125),
    ]
    _, fund = parse_dfp_zip(_make_fake_dfp_zip(dre, BASE_BPA, BASE_BPP), {})
    assert fund["ebitda"].iloc[0] == pytest.approx(6_462_125.0)


def test_insurer_chart_ebitda_at_3_07():
    """Insurer chart (BBSE): same EBIT description but at code 3.07."""
    dre = [
        _dre_row("3.01", "Receitas das Atividades Seguradoras/Resseguradoras", 0),
        _dre_row("3.07", "Resultado Antes do Resultado Financeiro e dos Tributos", 8905767),
        _dre_row("3.13", "Lucro/Prejuízo Consolidado do Período", 7947203),
    ]
    _, fund = parse_dfp_zip(_make_fake_dfp_zip(dre, BASE_BPA, BASE_BPP), {})
    assert fund["ebitda"].iloc[0] == pytest.approx(8_905_767.0)
    assert fund["net_income"].iloc[0] == pytest.approx(7_947_203.0)


def test_bank_chart_ebitda_and_net_debt_are_null():
    """Bank chart (ITUB): 3.05 is PRE-TAX income, not EBIT — must not be stored
    as ebitda. Banks also report none of the cash/debt accounts, so net_debt
    must be NULL, not a fake 0."""
    dre = [
        _dre_row("3.01", "Receitas da Intermediação Financeira", 313221000),
        _dre_row("3.05", "Resultado Antes dos Tributos sobre o Lucro", 39700000),
        _dre_row("3.09", "Lucro/Prejuízo Consolidado do Período", 33877000),
    ]
    bpa = [dict(BASE_BPA[0])]  # only account "1" — banks have no 1.01.01
    bpp = [{
        "CNPJ_CIA": "33.000.167/0001-01",
        "DT_REFER": "2023-12-31",
        "DT_RECEB": "2024-03-15",
        "VERSAO": "1",
        "ORDEM_EXERC": "ÚLTIMO",
        "CD_CONTA": "2.08",
        "DS_CONTA": "Patrimônio Líquido Consolidado",
        "VL_CONTA": "218224000",
    }]
    _, fund = parse_dfp_zip(_make_fake_dfp_zip(dre, bpa, bpp), {})
    row = fund.iloc[0]
    assert row["revenue"] == pytest.approx(313_221_000.0)
    assert row["net_income"] == pytest.approx(33_877_000.0)
    assert row["equity"] == pytest.approx(218_224_000.0)
    assert pd.isna(row.get("ebitda")), f"bank ebitda must be NULL, got {row.get('ebitda')}"
    assert pd.isna(row["net_debt"]), f"bank net_debt must be NULL, got {row['net_debt']}"


def test_parse_fre_zip_zero_shares_stored_as_null():
    """A row with all share count columns = 0 must produce shares_outstanding = NaN."""
    import math

    capital_rows = [
        {
            "CNPJ_Companhia": "33.000.167/0001-01",
            "Data_Referencia": "2023-12-31",
            "DT_RECEB": "2024-02-10",
            "Versao": "1",
            "Quantidade_Total_Acoes": "0",
            "Quantidade_Acoes_Ordinarias": "0",
            "Quantidade_Acoes_Preferenciais": "0",
        }
    ]
    fake_zip = _make_fake_fre_zip(capital_rows)
    filings_df, fundamentals_df = parse_fre_zip(fake_zip, {"33000167000101": "PETR"})

    assert not fundamentals_df.empty, "fundamentals_df should not be empty even with zero shares"
    shares = fundamentals_df["shares_outstanding"].iloc[0]
    assert shares is None or (isinstance(shares, float) and math.isnan(shares)), (
        f"Expected NaN/None for zero shares, got {shares}"
    )

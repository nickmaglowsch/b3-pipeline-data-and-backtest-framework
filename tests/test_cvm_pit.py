"""
Point-in-time correctness tests for the CVM fundamentals pipeline.

Covers:
  (a) ITR DRE year-to-date row selection (DT_INI_EXERC disambiguation)
  (b) net_income_ttm computation incl. missing-prior-year -> NULL
  (c) filing_date fallback = period_end + legal deadline (45/90 days)
  (d) FCA valor_mobiliario parsing into company_tickers_pit
  (e) staleness expiry in the monthly materialization

All tests use in-memory synthetic ZIPs / in-memory SQLite — no network, no real DB.
"""
from __future__ import annotations

import io
import sqlite3
import zipfile

import pandas as pd
import pytest

from b3_pipeline import cvm_storage, fca_parser, storage
from b3_pipeline.cvm_main import compute_net_income_ttm, materialize_fundamentals_monthly
from b3_pipeline.cvm_parser import parse_dfp_zip, parse_fre_zip, parse_itr_zip


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _csv_bytes(rows: list[dict]) -> bytes:
    """Convert a list of dicts to semicolon-delimited CSV bytes (latin-1)."""
    buf = io.StringIO()
    pd.DataFrame(rows).to_csv(buf, sep=";", index=False)
    return buf.getvalue().encode("latin-1")


def _zip_with(files: dict[str, list[dict]]) -> io.BytesIO:
    """Build an in-memory ZIP with {csv_name: rows}."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, rows in files.items():
            zf.writestr(name, _csv_bytes(rows))
    buf.seek(0)
    return buf


@pytest.fixture
def mem_conn():
    conn = sqlite3.connect(":memory:")
    storage.init_db(conn)
    yield conn
    conn.close()


def _insert_pit_row(conn, **kwargs):
    defaults = {
        "filing_id": "X",
        "cnpj": "33000167000101",
        "ticker": None,
        "period_end": "2023-12-31",
        "filing_date": "2024-03-01",
        "filing_version": 1,
        "doc_type": "DFP",
        "fiscal_year": 2023,
        "quarter": None,
        "revenue": None,
        "net_income": None,
        "ebitda": None,
        "total_assets": None,
        "equity": None,
        "net_debt": None,
        "shares_outstanding": None,
        "net_income_ttm": None,
    }
    defaults.update(kwargs)
    conn.execute(
        """
        INSERT OR REPLACE INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version,
           doc_type, fiscal_year, quarter, revenue, net_income, ebitda,
           total_assets, equity, net_debt, shares_outstanding, net_income_ttm)
        VALUES
          (:filing_id, :cnpj, :ticker, :period_end, :filing_date, :filing_version,
           :doc_type, :fiscal_year, :quarter, :revenue, :net_income, :ebitda,
           :total_assets, :equity, :net_debt, :shares_outstanding, :net_income_ttm)
        """,
        defaults,
    )
    conn.commit()


def _dre_row(**kwargs):
    row = {
        "CNPJ_CIA": "33.000.167/0001-01",
        "DT_REFER": "2023-06-30",
        "DT_RECEB": "2023-08-01",
        "VERSAO": "1",
        "ORDEM_EXERC": "ÚLTIMO",
        "CD_CONTA": "3.11",
        "VL_CONTA": "100",
    }
    row.update(kwargs)
    # net_income is selected by description (bank vs corporate charts differ)
    row.setdefault("DS_CONTA", {
        "3.11": "Lucro/Prejuízo Consolidado do Período",
        "3.01": "Receita de Venda de Bens e/ou Serviços",
    }.get(row["CD_CONTA"], "Outra Conta"))
    return row


# ──────────────────────────────────────────────────────────────────────────────
# (a) ITR YTD row selection
# ──────────────────────────────────────────────────────────────────────────────

def test_itr_dre_selects_ytd_rows_over_quarterly():
    """When both quarter-only and YTD rows exist for the same DT_REFER,
    the YTD row (earliest DT_INI_EXERC) must be selected."""
    dre_rows = [
        # quarter-only row (Q2: Apr-Jun)
        _dre_row(DT_INI_EXERC="2023-04-01", CD_CONTA="3.11", VL_CONTA="100"),
        # year-to-date row (Jan-Jun) — this one must win
        _dre_row(DT_INI_EXERC="2023-01-01", CD_CONTA="3.11", VL_CONTA="250"),
        _dre_row(DT_INI_EXERC="2023-04-01", CD_CONTA="3.01", VL_CONTA="400"),
        _dre_row(DT_INI_EXERC="2023-01-01", CD_CONTA="3.01", VL_CONTA="900"),
    ]
    fake_zip = _zip_with({"itr_cia_aberta_DRE_con_2023.csv": dre_rows})
    _, fund_df = parse_itr_zip(fake_zip, {"33000167000101": "PETR"})

    assert len(fund_df) == 1
    assert fund_df["net_income"].iloc[0] == pytest.approx(250.0), (
        f"Expected YTD net_income 250, got {fund_df['net_income'].iloc[0]}"
    )
    assert fund_df["revenue"].iloc[0] == pytest.approx(900.0), (
        f"Expected YTD revenue 900, got {fund_df['revenue'].iloc[0]}"
    )


def test_dre_without_dt_ini_exerc_still_parses():
    """Files without DT_INI_EXERC (e.g. older layouts) must parse unchanged."""
    dre_rows = [_dre_row(CD_CONTA="3.11", VL_CONTA="123")]
    fake_zip = _zip_with({"itr_cia_aberta_DRE_con_2023.csv": dre_rows})
    _, fund_df = parse_itr_zip(fake_zip, {})
    assert fund_df["net_income"].iloc[0] == pytest.approx(123.0)


# ──────────────────────────────────────────────────────────────────────────────
# (b) TTM computation
# ──────────────────────────────────────────────────────────────────────────────

def _ttm(conn, filing_id):
    row = conn.execute(
        "SELECT net_income_ttm FROM fundamentals_pit WHERE filing_id = ?", (filing_id,)
    ).fetchone()
    assert row is not None, f"missing row {filing_id}"
    return row[0]


def test_ttm_dfp_equals_annual(mem_conn):
    _insert_pit_row(mem_conn, filing_id="C_DFP_2022-12-31_1", doc_type="DFP",
                    period_end="2022-12-31", fiscal_year=2022, net_income=1000.0)
    compute_net_income_ttm(mem_conn)
    assert _ttm(mem_conn, "C_DFP_2022-12-31_1") == pytest.approx(1000.0)


def test_ttm_itr_uses_prior_annual_and_prior_ytd(mem_conn):
    # Annual 2022 = 1000; YTD 2022 Q1 = 200; YTD 2023 Q1 = 300
    # TTM(2023 Q1) = 300 + 1000 - 200 = 1100
    _insert_pit_row(mem_conn, filing_id="C_DFP_2022-12-31_1", doc_type="DFP",
                    period_end="2022-12-31", fiscal_year=2022, net_income=1000.0)
    _insert_pit_row(mem_conn, filing_id="C_ITR_2022-03-31_1", doc_type="ITR",
                    period_end="2022-03-31", fiscal_year=2022, quarter=1, net_income=200.0)
    _insert_pit_row(mem_conn, filing_id="C_ITR_2023-03-31_1", doc_type="ITR",
                    period_end="2023-03-31", fiscal_year=2023, quarter=1, net_income=300.0)
    compute_net_income_ttm(mem_conn)
    assert _ttm(mem_conn, "C_ITR_2023-03-31_1") == pytest.approx(1100.0)


def test_ttm_uses_latest_filing_version(mem_conn):
    """A restated prior-year ITR (v2) must be used instead of v1."""
    _insert_pit_row(mem_conn, filing_id="C_DFP_2022-12-31_1", doc_type="DFP",
                    period_end="2022-12-31", fiscal_year=2022, net_income=1000.0)
    _insert_pit_row(mem_conn, filing_id="C_ITR_2022-03-31_1", doc_type="ITR",
                    period_end="2022-03-31", fiscal_year=2022, quarter=1,
                    filing_version=1, net_income=999.0)
    _insert_pit_row(mem_conn, filing_id="C_ITR_2022-03-31_2", doc_type="ITR",
                    period_end="2022-03-31", fiscal_year=2022, quarter=1,
                    filing_version=2, net_income=200.0)
    _insert_pit_row(mem_conn, filing_id="C_ITR_2023-03-31_1", doc_type="ITR",
                    period_end="2023-03-31", fiscal_year=2023, quarter=1, net_income=300.0)
    compute_net_income_ttm(mem_conn)
    assert _ttm(mem_conn, "C_ITR_2023-03-31_1") == pytest.approx(1100.0)


def test_ttm_ignores_prior_restatement_filed_after_current_row(mem_conn):
    """A prior-year DFP restatement filed AFTER the current ITR's filing_date
    must not leak into the TTM stamped with the ITR's PIT date (lookahead)."""
    _insert_pit_row(mem_conn, filing_id="C_DFP_2022-12-31_1", doc_type="DFP",
                    period_end="2022-12-31", fiscal_year=2022,
                    filing_version=1, filing_date="2023-03-01", net_income=1000.0)
    # Restated a year later -- after the 2023 Q1 ITR below was filed
    _insert_pit_row(mem_conn, filing_id="C_DFP_2022-12-31_2", doc_type="DFP",
                    period_end="2022-12-31", fiscal_year=2022,
                    filing_version=2, filing_date="2023-08-01", net_income=2000.0)
    _insert_pit_row(mem_conn, filing_id="C_ITR_2022-03-31_1", doc_type="ITR",
                    period_end="2022-03-31", fiscal_year=2022, quarter=1,
                    filing_date="2022-05-01", net_income=200.0)
    _insert_pit_row(mem_conn, filing_id="C_ITR_2023-03-31_1", doc_type="ITR",
                    period_end="2023-03-31", fiscal_year=2023, quarter=1,
                    filing_date="2023-05-01", net_income=300.0)
    compute_net_income_ttm(mem_conn)
    # As of 2023-05-01 only DFP v1 (1000) existed: 300 + 1000 - 200 = 1100
    assert _ttm(mem_conn, "C_ITR_2023-03-31_1") == pytest.approx(1100.0)


def test_ttm_null_when_prior_year_missing(mem_conn):
    """Missing Annual(Y-1) or YTD(Y-1, q) must leave net_income_ttm NULL."""
    # No DFP 2022 at all -> NULL
    _insert_pit_row(mem_conn, filing_id="C_ITR_2023-03-31_1", doc_type="ITR",
                    period_end="2023-03-31", fiscal_year=2023, quarter=1, net_income=300.0)
    # DFP 2022 present but no ITR 2022 Q2 -> Q2 2023 also NULL
    _insert_pit_row(mem_conn, filing_id="D_DFP_2022-12-31_1", cnpj="99999999000199",
                    doc_type="DFP", period_end="2022-12-31", fiscal_year=2022,
                    net_income=500.0)
    _insert_pit_row(mem_conn, filing_id="D_ITR_2023-06-30_1", cnpj="99999999000199",
                    doc_type="ITR", period_end="2023-06-30", fiscal_year=2023,
                    quarter=2, net_income=50.0)
    compute_net_income_ttm(mem_conn)
    assert _ttm(mem_conn, "C_ITR_2023-03-31_1") is None
    assert _ttm(mem_conn, "D_ITR_2023-06-30_1") is None


# ──────────────────────────────────────────────────────────────────────────────
# (c) filing_date fallback = period_end + legal deadline
# ──────────────────────────────────────────────────────────────────────────────

def test_itr_filing_date_fallback_45_days():
    dre_rows = [_dre_row(DT_REFER="2023-06-30")]
    del dre_rows[0]["DT_RECEB"]
    fake_zip = _zip_with({"itr_cia_aberta_DRE_con_2023.csv": dre_rows})
    filings_df, fund_df = parse_itr_zip(fake_zip, {})
    # 2023-06-30 + 45 days = 2023-08-14 (NOT the period end itself)
    assert filings_df["filing_date"].iloc[0] == "2023-08-14"
    assert fund_df["filing_date"].iloc[0] == "2023-08-14"


def test_dfp_filing_date_fallback_90_days():
    dre_rows = [_dre_row(DT_REFER="2023-12-31")]
    del dre_rows[0]["DT_RECEB"]
    fake_zip = _zip_with({"dfp_cia_aberta_DRE_con_2023.csv": dre_rows})
    filings_df, _ = parse_dfp_zip(fake_zip, {})
    # 2023-12-31 + 90 days = 2024-03-30
    assert filings_df["filing_date"].iloc[0] == "2024-03-30"


def test_dt_receb_kept_when_present():
    dre_rows = [_dre_row(DT_REFER="2023-06-30", DT_RECEB="2023-08-01")]
    fake_zip = _zip_with({"itr_cia_aberta_DRE_con_2023.csv": dre_rows})
    filings_df, _ = parse_itr_zip(fake_zip, {})
    assert filings_df["filing_date"].iloc[0] == "2023-08-01"


def test_main_metadata_csv_preferred_over_subtable():
    """The main metadata CSV (itr_cia_aberta_YYYY.csv, the only one with
    DT_RECEB) must be matched exactly -- a sub-table like
    itr_cia_aberta_parecer_YYYY.csv appearing earlier in the ZIP must not
    win and silently drop the real filing receipt date."""
    dre_rows = [_dre_row(DT_REFER="2023-06-30")]
    del dre_rows[0]["DT_RECEB"]  # DT_RECEB only lives in the metadata CSV
    parecer_rows = [{  # sub-table: same keys but NO DT_RECEB
        "CNPJ_CIA": "33.000.167/0001-01", "DT_REFER": "2023-06-30", "VERSAO": "1",
    }]
    meta_rows = [{
        "CNPJ_CIA": "33.000.167/0001-01", "DT_REFER": "2023-06-30", "VERSAO": "1",
        "DT_RECEB": "2023-08-01",
    }]
    fake_zip = _zip_with({
        "itr_cia_aberta_parecer_2023.csv": parecer_rows,  # listed first on purpose
        "itr_cia_aberta_2023.csv": meta_rows,
        "itr_cia_aberta_DRE_con_2023.csv": dre_rows,
    })
    filings_df, _ = parse_itr_zip(fake_zip, {})
    assert filings_df["filing_date"].iloc[0] == "2023-08-01", (
        "filing_date must come from the main metadata CSV's DT_RECEB, "
        "not fall back to the legal-deadline estimate"
    )


def test_fre_filing_date_fallback_90_days():
    capital_rows = [{
        "CNPJ_Companhia": "33.000.167/0001-01",
        "Data_Referencia": "2023-12-31",
        "Versao": "1",
        "Quantidade_Total_Acoes": "1000000",
    }]
    fake_zip = _zip_with({"fre_cia_aberta_capital_social_2023.csv": capital_rows})
    filings_df, _ = parse_fre_zip(fake_zip, {})
    assert filings_df["filing_date"].iloc[0] == "2024-03-30"


# ──────────────────────────────────────────────────────────────────────────────
# (d) FCA valor_mobiliario -> company_tickers_pit
# ──────────────────────────────────────────────────────────────────────────────

FCA_ROWS = [
    {
        "CNPJ_Companhia": "11.111.111/0001-91",
        "Codigo_Negociacao": "XPTO3",
        "Mercado": "Bolsa",
        "Data_Inicio_Negociacao": "2015-01-02",
        "Data_Fim_Negociacao": "2020-05-10",  # delisted — B3 API would miss it
    },
    {
        "CNPJ_Companhia": "11.111.111/0001-91",
        "Codigo_Negociacao": "N/A",  # invalid ticker — dropped
        "Mercado": "Balcão",
        "Data_Inicio_Negociacao": "",
        "Data_Fim_Negociacao": "",
    },
]


def _fca_zip(rows=FCA_ROWS) -> io.BytesIO:
    return _zip_with({"fca_cia_aberta_valor_mobiliario_2023.csv": rows})


def test_parse_fca_zip_extracts_ticker_mappings():
    df = fca_parser.parse_fca_zip(_fca_zip())
    assert len(df) == 1, f"Expected 1 valid row, got {len(df)}"
    row = df.iloc[0]
    assert row["cnpj"] == "11111111000191"
    assert row["ticker"] == "XPTO3"
    assert row["ticker_root"] == "XPTO"
    assert row["market"] == "Bolsa"
    assert row["start_date"] == "2015-01-02"
    assert row["end_date"] == "2020-05-10"
    assert row["source"] == "FCA"


def test_fca_rows_upsert_into_company_tickers_pit(mem_conn):
    df = fca_parser.parse_fca_zip(_fca_zip())
    n = cvm_storage.upsert_company_tickers_pit(mem_conn, df)
    assert n == 1
    row = mem_conn.execute(
        "SELECT cnpj, ticker, ticker_root, market, start_date, end_date, source "
        "FROM company_tickers_pit"
    ).fetchone()
    assert row == ("11111111000191", "XPTO3", "XPTO", "Bolsa",
                   "2015-01-02", "2020-05-10", "FCA")


def test_fca_fills_tickers_for_delisted_companies(mem_conn):
    """A delisted company (no B3 API ticker) gets its root from company_tickers_pit,
    and the fill propagates into fundamentals_pit via the existing UPDATE."""
    # Delisted company: known to CVM but B3 API left ticker NULL
    mem_conn.execute(
        "INSERT INTO cvm_companies (cnpj, company_name) VALUES (?, ?)",
        ("11111111000191", "Extinta SA"),
    )
    # A listed company whose B3-provided ticker must NOT be overwritten
    mem_conn.execute(
        "INSERT INTO cvm_companies (cnpj, ticker, ticker_root) VALUES (?, ?, ?)",
        ("33000167000101", "PETR", "PETR"),
    )
    mem_conn.commit()
    cvm_storage.upsert_company_tickers_pit(mem_conn, fca_parser.parse_fca_zip(_fca_zip()))

    updated = cvm_storage.populate_tickers_from_company_tickers_pit(mem_conn)
    assert updated == 1

    ticker, root = mem_conn.execute(
        "SELECT ticker, ticker_root FROM cvm_companies WHERE cnpj='11111111000191'"
    ).fetchone()
    assert ticker == "XPTO" and root == "XPTO", "ticker must be the 4-char root (join-key invariant)"
    assert mem_conn.execute(
        "SELECT ticker FROM cvm_companies WHERE cnpj='33000167000101'"
    ).fetchone()[0] == "PETR", "existing B3 ticker must not be overwritten"

    # fundamentals_pit rows for the delisted company now join to prices by root
    _insert_pit_row(mem_conn, filing_id="X_DFP_2019-12-31_1", cnpj="11111111000191",
                    period_end="2019-12-31", fiscal_year=2019, net_income=10.0)
    cvm_storage.populate_tickers_from_cvm_companies(mem_conn)
    assert mem_conn.execute(
        "SELECT ticker FROM fundamentals_pit WHERE filing_id='X_DFP_2019-12-31_1'"
    ).fetchone()[0] == "XPTO"


# ──────────────────────────────────────────────────────────────────────────────
# (e) staleness expiry in monthly materialization
# ──────────────────────────────────────────────────────────────────────────────

def test_monthly_materialization_expires_stale_values(mem_conn):
    """A company that stopped filing must not keep its last values forever:
    values expire FUNDAMENTALS_MAX_STALENESS_DAYS after their filing_date."""
    _insert_pit_row(
        mem_conn,
        filing_id="A_DFP_2019-12-31_1",
        cnpj="11111111000191",
        ticker="AAAA",
        period_end="2019-12-31",
        filing_date="2020-03-01",
        fiscal_year=2019,
        equity=100.0,
        net_income=10.0,
    )
    materialize_fundamentals_monthly(mem_conn)

    # Within the 400-day window (2020-03-01 + 400d = 2021-04-05): value present
    row = mem_conn.execute(
        "SELECT equity FROM fundamentals_monthly WHERE ticker='AAAA' AND month_end='2020-06-30'"
    ).fetchone()
    assert row is not None and row[0] == pytest.approx(100.0)
    row = mem_conn.execute(
        "SELECT equity FROM fundamentals_monthly WHERE ticker='AAAA' AND month_end='2021-03-31'"
    ).fetchone()
    assert row is not None and row[0] == pytest.approx(100.0)

    # Beyond the window: expired — no forward-filled rows at all
    count = mem_conn.execute(
        "SELECT COUNT(*) FROM fundamentals_monthly WHERE ticker='AAAA' AND month_end > '2021-04-05'"
    ).fetchone()[0]
    assert count == 0, f"Expected stale values to be expired, found {count} rows past cutoff"


def test_monthly_materialization_refreshes_staleness_on_new_filing(mem_conn):
    """A newer filing restarts the staleness clock."""
    _insert_pit_row(mem_conn, filing_id="A_DFP_2019-12-31_1", cnpj="11111111000191",
                    ticker="AAAA", period_end="2019-12-31", filing_date="2020-03-01",
                    fiscal_year=2019, equity=100.0)
    _insert_pit_row(mem_conn, filing_id="A_DFP_2020-12-31_1", cnpj="11111111000191",
                    ticker="AAAA", period_end="2020-12-31", filing_date="2021-03-01",
                    fiscal_year=2020, equity=200.0)
    materialize_fundamentals_monthly(mem_conn)

    # 2021-12-31 is > 400d after the 2020 filing but < 400d after the 2021 filing
    row = mem_conn.execute(
        "SELECT equity FROM fundamentals_monthly WHERE ticker='AAAA' AND month_end='2021-12-31'"
    ).fetchone()
    assert row is not None and row[0] == pytest.approx(200.0)


def test_monthly_materialization_carries_net_income_ttm(mem_conn):
    """net_income_ttm flows from fundamentals_pit into fundamentals_monthly."""
    _insert_pit_row(mem_conn, filing_id="A_DFP_2022-12-31_1", cnpj="11111111000191",
                    ticker="AAAA", period_end="2022-12-31", filing_date="2023-03-01",
                    fiscal_year=2022, net_income=1000.0, net_income_ttm=1000.0)
    materialize_fundamentals_monthly(mem_conn)
    row = mem_conn.execute(
        "SELECT net_income_ttm FROM fundamentals_monthly WHERE ticker='AAAA' AND month_end='2023-03-31'"
    ).fetchone()
    assert row is not None and row[0] == pytest.approx(1000.0)

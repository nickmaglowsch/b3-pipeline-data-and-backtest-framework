"""
Tests for fundamentals_monthly schema and storage functions (Task 01 TDD).
Extended by Task 04 (materialize_fundamentals_monthly tests).
Extended by Task 05 (load_fundamentals_monthly and dynamic ratio helper tests).

All tests use in-memory SQLite with synthetic data.
No network calls or real CVM files are used.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from b3_pipeline import storage, cvm_storage


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mem_conn():
    """In-memory SQLite connection with full schema including fundamentals tables."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    yield conn
    conn.close()


def _insert_price(conn, ticker, date, close, volume=1000):
    """Helper: insert a minimal prices row."""
    conn.execute(
        """
        INSERT OR REPLACE INTO prices (ticker, isin_code, date, open, high, low, close, volume)
        VALUES (?, 'UNKNOWN', ?, ?, ?, ?, ?, ?)
        """,
        (ticker, date, close, close, close, close, int(volume * 100)),
    )
    conn.commit()


def _insert_pit_row(conn, **kwargs):
    """Helper: insert a fundamentals_pit row with defaults for unspecified columns."""
    defaults = {
        "filing_id": "TEST_DFP_2023-12-31_1",
        "cnpj": "33000167000101",
        "ticker": "PETR",
        "period_end": "2023-12-31",
        "filing_date": "2023-01-31",
        "filing_version": 1,
        "doc_type": "DFP",
        "fiscal_year": 2022,
        "quarter": None,
        "revenue": None,
        "net_income": None,
        "ebitda": None,
        "total_assets": None,
        "equity": None,
        "net_debt": None,
        "shares_outstanding": None,
    }
    defaults.update(kwargs)
    conn.execute(
        """
        INSERT OR REPLACE INTO fundamentals_pit
          (filing_id, cnpj, ticker, period_end, filing_date, filing_version,
           doc_type, fiscal_year, quarter,
           revenue, net_income, ebitda, total_assets, equity, net_debt,
           shares_outstanding)
        VALUES
          (:filing_id, :cnpj, :ticker, :period_end, :filing_date, :filing_version,
           :doc_type, :fiscal_year, :quarter,
           :revenue, :net_income, :ebitda, :total_assets, :equity, :net_debt,
           :shares_outstanding)
        """,
        defaults,
    )
    conn.commit()


# ── Task 01: Schema tests ──────────────────────────────────────────────────────

def test_fundamentals_monthly_table_exists(mem_conn):
    """init_db() should create fundamentals_monthly with raw financial columns (no ratio columns)."""
    cursor = mem_conn.cursor()
    cursor.execute("PRAGMA table_info(fundamentals_monthly)")
    cols = {row[1] for row in cursor.fetchall()}
    assert "month_end" in cols
    assert "ticker" in cols
    assert "revenue" in cols
    assert "net_income" in cols
    assert "ebitda" in cols
    assert "total_assets" in cols
    assert "equity" in cols
    assert "net_debt" in cols
    assert "shares_outstanding" in cols
    # Ratio columns are intentionally excluded — computed dynamically at query time
    assert "pe_ratio" not in cols
    assert "pb_ratio" not in cols
    assert "ev_ebitda" not in cols


def test_fundamentals_monthly_primary_key(mem_conn):
    """INSERT OR REPLACE with same (month_end, ticker) should keep row count at 1."""
    for _ in range(2):
        mem_conn.execute(
            "INSERT OR REPLACE INTO fundamentals_monthly (month_end, ticker, net_income) VALUES (?,?,?)",
            ("2023-01-31", "PETR", 1000.0),
        )
        mem_conn.commit()
    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fundamentals_monthly")
    assert cursor.fetchone()[0] == 1


def test_upsert_fundamentals_monthly_inserts(mem_conn):
    """upsert_fundamentals_monthly should insert all rows from a DataFrame."""
    df = pd.DataFrame([
        {"month_end": "2023-01-31", "ticker": "PETR", "net_income": 1000.0},
        {"month_end": "2023-01-31", "ticker": "VALE", "net_income": 2000.0},
    ])
    cvm_storage.upsert_fundamentals_monthly(mem_conn, df)
    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fundamentals_monthly")
    assert cursor.fetchone()[0] == 2


def test_upsert_fundamentals_monthly_replaces(mem_conn):
    """Second upsert with same key should replace net_income."""
    df1 = pd.DataFrame([{"month_end": "2023-01-31", "ticker": "PETR", "net_income": 1000.0}])
    df2 = pd.DataFrame([{"month_end": "2023-01-31", "ticker": "PETR", "net_income": 2000.0}])
    cvm_storage.upsert_fundamentals_monthly(mem_conn, df1)
    cvm_storage.upsert_fundamentals_monthly(mem_conn, df2)
    cursor = mem_conn.cursor()
    cursor.execute("SELECT net_income FROM fundamentals_monthly WHERE month_end='2023-01-31' AND ticker='PETR'")
    assert cursor.fetchone()[0] == pytest.approx(2000.0)


def test_truncate_fundamentals_monthly(mem_conn):
    """truncate_fundamentals_monthly should delete all rows."""
    for i, ticker in enumerate(["PETR", "VALE", "ITUB"]):
        mem_conn.execute(
            "INSERT OR REPLACE INTO fundamentals_monthly (month_end, ticker, net_income) VALUES (?,?,?)",
            ("2023-01-31", ticker, float((i + 1) * 1000)),
        )
    mem_conn.commit()
    cvm_storage.truncate_fundamentals_monthly(mem_conn)
    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fundamentals_monthly")
    assert cursor.fetchone()[0] == 0


def test_migrate_schema_safe_on_existing_db(mem_conn):
    """_migrate_schema() called a second time should not raise."""
    storage._migrate_schema(mem_conn)  # second call
    cursor = mem_conn.cursor()
    cursor.execute("PRAGMA table_info(fundamentals_monthly)")
    cols = {row[1] for row in cursor.fetchall()}
    assert "month_end" in cols


def test_rebuild_drops_fundamentals_monthly(tmp_path):
    """init_db(rebuild=True) should drop and recreate fundamentals_monthly."""
    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    storage.init_db(conn)
    conn.execute(
        "INSERT OR REPLACE INTO fundamentals_monthly (month_end, ticker, net_income) VALUES (?,?,?)",
        ("2023-01-31", "PETR", 1000.0),
    )
    conn.commit()
    storage.init_db(conn, rebuild=True)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fundamentals_monthly")
    assert cursor.fetchone()[0] == 0
    conn.close()


# ── Task 04: materialize_fundamentals_monthly tests ────────────────────────────

def _setup_petr_company(conn):
    """Helper: insert cvm_companies row for PETR."""
    conn.execute(
        "INSERT OR REPLACE INTO cvm_companies (cnpj, ticker, company_name) VALUES (?,?,?)",
        ("33000167000101", "PETR", "Petrobras"),
    )
    conn.commit()


def test_materialize_monthly_basic_snapshot(mem_conn):
    """materialize_fundamentals_monthly writes raw net_income at each month-end.

    When no prices exist to build an ADTV map, the ticker in fundamentals_monthly
    is the root ticker from fundamentals_pit (e.g. 'PETR'), not the suffixed form.
    """
    from b3_pipeline.cvm_main import materialize_fundamentals_monthly

    _setup_petr_company(mem_conn)
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2022-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-01-31",
        period_end="2022-12-31",
        net_income=1000.0,
        equity=5000.0,
        shares_outstanding=10_000_000.0,
    )

    materialize_fundamentals_monthly(mem_conn)

    cursor = mem_conn.cursor()
    # Without prices, ADTV map is empty → ticker falls back to root "PETR"
    cursor.execute(
        "SELECT net_income FROM fundamentals_monthly WHERE ticker=? ORDER BY month_end LIMIT 1",
        ("PETR",),
    )
    row = cursor.fetchone()
    assert row is not None, "Expected a row for PETR in fundamentals_monthly"
    assert row[0] == pytest.approx(1000.0, rel=1e-3), f"Expected net_income 1000.0, got {row[0]}"


def test_materialize_monthly_forward_fills_financials(mem_conn):
    """net_income at later month-ends is forward-filled from the January filing."""
    from b3_pipeline.cvm_main import materialize_fundamentals_monthly
    import datetime as dt

    _setup_petr_company(mem_conn)
    filing_date = "2023-01-15"
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2022-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date=filing_date,
        period_end="2022-12-31",
        net_income=1000.0,
        shares_outstanding=10_000_000.0,
    )

    materialize_fundamentals_monthly(mem_conn)

    cursor = mem_conn.cursor()
    # Without prices, ADTV map is empty → ticker falls back to root "PETR"
    cursor.execute(
        "SELECT net_income FROM fundamentals_monthly WHERE ticker=? ORDER BY month_end DESC LIMIT 1",
        ("PETR",),
    )
    row = cursor.fetchone()
    assert row is not None, "Expected a row for PETR"
    assert row[0] == pytest.approx(1000.0, rel=1e-3), f"Expected net_income 1000.0, got {row[0]}"


def test_materialize_monthly_raw_financials_stored_without_price(mem_conn):
    """Raw financial metrics are stored even when no price is available for a ticker.
    Prices are no longer needed since ratios are computed dynamically at query time."""
    from b3_pipeline.cvm_main import materialize_fundamentals_monthly

    conn = mem_conn
    conn.execute(
        "INSERT OR REPLACE INTO cvm_companies (cnpj, ticker, company_name) VALUES (?,?,?)",
        ("11111111000191", "VALE", "Vale"),
    )
    conn.commit()
    _insert_pit_row(
        conn,
        filing_id="VALE_DFP_2022-12-31_1",
        cnpj="11111111000191",
        ticker="VALE",
        filing_date="2023-01-31",
        period_end="2022-12-31",
        net_income=2000.0,
        shares_outstanding=5_000_000.0,
    )
    # No prices inserted — prices are no longer needed for materialize_fundamentals_monthly

    materialize_fundamentals_monthly(conn)

    cursor = conn.cursor()
    # Without prices, ADTV map is empty → ticker falls back to root "VALE"
    cursor.execute(
        "SELECT net_income FROM fundamentals_monthly WHERE ticker=? LIMIT 1",
        ("VALE",),
    )
    row = cursor.fetchone()
    # Raw financial metrics should be stored regardless of price availability
    assert row is not None, "Expected a row for VALE even without prices"
    assert row[0] == pytest.approx(2000.0), f"Expected net_income=2000.0, got {row[0]}"


def test_materialize_monthly_forward_fills_equity(mem_conn):
    """equity is forward-filled from the original filing into all subsequent month-ends."""
    from b3_pipeline.cvm_main import materialize_fundamentals_monthly

    _setup_petr_company(mem_conn)
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2022-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-01-15",
        period_end="2022-12-31",
        equity=5000.0,
        net_income=1000.0,
    )

    materialize_fundamentals_monthly(mem_conn)

    cursor = mem_conn.cursor()
    # Without prices, ADTV map is empty → ticker falls back to root "PETR"
    cursor.execute(
        "SELECT equity FROM fundamentals_monthly WHERE ticker=? ORDER BY month_end DESC LIMIT 1",
        ("PETR",),
    )
    row = cursor.fetchone()
    assert row is not None, "Expected a row for PETR"
    assert row[0] == pytest.approx(5000.0, rel=1e-3), f"Expected equity=5000.0 (forward-filled), got {row[0]}"


def test_materialize_monthly_truncates_before_rebuild(mem_conn):
    """Old rows should be removed before rebuilding."""
    from b3_pipeline.cvm_main import materialize_fundamentals_monthly

    # Insert a stale row manually
    mem_conn.execute(
        "INSERT OR REPLACE INTO fundamentals_monthly (month_end, ticker, net_income) VALUES (?,?,?)",
        ("2020-01-31", "STALE", 999.0),
    )
    mem_conn.commit()

    _setup_petr_company(mem_conn)
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2022-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-01-31",
        period_end="2022-12-31",
        net_income=1000.0,
        shares_outstanding=10_000_000.0,
    )

    materialize_fundamentals_monthly(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fundamentals_monthly WHERE ticker='STALE'")
    assert cursor.fetchone()[0] == 0, "Stale rows should be removed after rebuild"


def test_materialize_monthly_returns_row_count(mem_conn):
    """Return value should equal SELECT COUNT(*) FROM fundamentals_monthly."""
    from b3_pipeline.cvm_main import materialize_fundamentals_monthly

    _setup_petr_company(mem_conn)
    mem_conn.execute(
        "INSERT OR REPLACE INTO cvm_companies (cnpj, ticker, company_name) VALUES (?,?,?)",
        ("11111111000191", "VALE", "Vale"),
    )
    mem_conn.commit()

    # PETR filing
    _insert_pit_row(
        mem_conn,
        filing_id="PETR_DFP_2022-12-31_1",
        cnpj="33000167000101",
        ticker="PETR",
        filing_date="2023-01-31",
        period_end="2022-12-31",
        net_income=1000.0,
        shares_outstanding=10_000_000.0,
    )
    # VALE filing
    _insert_pit_row(
        mem_conn,
        filing_id="VALE_DFP_2022-12-31_1",
        cnpj="11111111000191",
        ticker="VALE",
        filing_date="2023-01-31",
        period_end="2022-12-31",
        net_income=2000.0,
        shares_outstanding=5_000_000.0,
    )
    row_count = materialize_fundamentals_monthly(mem_conn)

    cursor = mem_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM fundamentals_monthly")
    actual_count = cursor.fetchone()[0]
    assert row_count == actual_count, (
        f"Return value {row_count} should match SELECT COUNT(*) {actual_count}"
    )


# ── Task 05: load_fundamentals_monthly and dynamic ratio helper tests ──────────

@pytest.fixture
def tmp_db(tmp_path):
    """File-based SQLite DB for load_fundamentals_monthly tests."""
    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    storage.init_db(conn)
    yield db_path, conn
    conn.close()


def test_load_fundamentals_monthly_returns_wide_df(tmp_db):
    """load_fundamentals_monthly returns a wide DatetimeIndex DataFrame."""
    from backtests.core.data import load_fundamentals_monthly

    db_path, conn = tmp_db
    rows = [
        ("2023-01-31", "PETR", 1000.0),
        ("2023-02-28", "PETR", 1200.0),
        ("2023-01-31", "VALE", 800.0),
    ]
    for month_end, ticker, ni in rows:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals_monthly (month_end, ticker, net_income) VALUES (?,?,?)",
            (month_end, ticker, ni),
        )
    conn.commit()

    wide = load_fundamentals_monthly(db_path, "net_income", "2023-01-01", "2023-03-31")

    assert isinstance(wide.index, pd.DatetimeIndex), "Index should be DatetimeIndex"
    assert "PETR" in wide.columns
    assert "VALE" in wide.columns
    assert wide.loc[pd.Timestamp("2023-01-31"), "PETR"] == pytest.approx(1000.0)
    assert wide.loc[pd.Timestamp("2023-02-28"), "PETR"] == pytest.approx(1200.0)
    assert wide.loc[pd.Timestamp("2023-01-31"), "VALE"] == pytest.approx(800.0)


def test_load_fundamentals_monthly_filters_date_range(tmp_db):
    """Rows outside start/end should not appear in result."""
    from backtests.core.data import load_fundamentals_monthly

    db_path, conn = tmp_db
    rows = [
        ("2022-12-31", "PETR", 900.0),
        ("2023-01-31", "PETR", 1000.0),
        ("2023-12-31", "PETR", 1500.0),
    ]
    for month_end, ticker, ni in rows:
        conn.execute(
            "INSERT OR REPLACE INTO fundamentals_monthly (month_end, ticker, net_income) VALUES (?,?,?)",
            (month_end, ticker, ni),
        )
    conn.commit()

    wide = load_fundamentals_monthly(db_path, "net_income", "2023-01-01", "2023-11-30")

    assert pd.Timestamp("2022-12-31") not in wide.index
    assert pd.Timestamp("2023-12-31") not in wide.index
    assert pd.Timestamp("2023-01-31") in wide.index


def test_load_fundamentals_monthly_unknown_metric_raises(tmp_db):
    """Unknown metric should raise ValueError."""
    from backtests.core.data import load_fundamentals_monthly

    db_path, conn = tmp_db
    with pytest.raises(ValueError, match="Unknown metric"):
        load_fundamentals_monthly(db_path, "foo", "2023-01-01", "2023-12-31")


def test_load_all_fundamentals_monthly_returns_dict(tmp_db):
    """load_all_fundamentals_monthly returns a dict with shares_outstanding, revenue, net_income keys."""
    from backtests.core.data import load_all_fundamentals_monthly

    db_path, conn = tmp_db
    conn.execute(
        "INSERT OR REPLACE INTO fundamentals_monthly (month_end, ticker, shares_outstanding, revenue) VALUES (?,?,?,?)",
        ("2023-01-31", "PETR", 1_000_000_000.0, 5000.0),
    )
    conn.commit()

    result = load_all_fundamentals_monthly(db_path, "2023-01-01", "2023-12-31")

    assert isinstance(result, dict)
    assert "shares_outstanding" in result
    assert "revenue" in result
    assert "net_income" in result


def test_compute_pe_ratio_dynamic_basic():
    """compute_pe_ratio_dynamic returns correct value for basic case."""
    from backtests.core.data import compute_pe_ratio_dynamic

    idx = [pd.Timestamp("2023-01-31")]
    cols = ["PETR"]
    # shares_outstanding as stored in DB (1000x raw units; divide by 1000 in helper)
    shares = pd.DataFrame([[10_000_000_000.0]], index=idx, columns=cols)
    net_income = pd.DataFrame([[1000.0]], index=idx, columns=cols)  # thousands BRL
    prices = pd.DataFrame([[30.0]], index=idx, columns=cols)

    result = compute_pe_ratio_dynamic(shares, net_income, prices)

    # (30 * 10_000_000_000 / 1000) / (1000 * 1000) = (30 * 10_000_000) / 1_000_000 = 300.0
    assert result.iloc[0, 0] == pytest.approx(300.0)


def test_compute_pe_ratio_dynamic_null_for_negative_income():
    """compute_pe_ratio_dynamic returns NaN for negative net_income."""
    from backtests.core.data import compute_pe_ratio_dynamic

    idx = [pd.Timestamp("2023-01-31")]
    cols = ["PETR"]
    shares = pd.DataFrame([[10_000_000_000.0]], index=idx, columns=cols)
    net_income = pd.DataFrame([[-500.0]], index=idx, columns=cols)
    prices = pd.DataFrame([[30.0]], index=idx, columns=cols)

    result = compute_pe_ratio_dynamic(shares, net_income, prices)

    assert pd.isna(result.iloc[0, 0]), f"Expected NaN for negative income, got {result.iloc[0, 0]}"


def test_compute_pb_ratio_dynamic_basic():
    """compute_pb_ratio_dynamic returns correct P/B ratio."""
    from backtests.core.data import compute_pb_ratio_dynamic

    idx = [pd.Timestamp("2023-01-31")]
    cols = ["PETR"]
    # shares_outstanding as stored in DB (1000x raw units; divide by 1000 in helper)
    shares = pd.DataFrame([[10_000_000_000.0]], index=idx, columns=cols)
    equity = pd.DataFrame([[5000.0]], index=idx, columns=cols)  # thousands BRL
    prices = pd.DataFrame([[30.0]], index=idx, columns=cols)

    result = compute_pb_ratio_dynamic(shares, equity, prices)

    # (30 * 10_000_000_000 / 1000) / (5000 * 1000) = 300_000_000 / 5_000_000 = 60.0
    assert result.iloc[0, 0] == pytest.approx(60.0)


def test_compute_ev_ebitda_dynamic_basic():
    """compute_ev_ebitda_dynamic returns correct EV/EBITDA."""
    from backtests.core.data import compute_ev_ebitda_dynamic

    idx = [pd.Timestamp("2023-01-31")]
    cols = ["PETR"]
    # shares_outstanding as stored in DB (1000x raw units; divide by 1000 in helper)
    shares = pd.DataFrame([[10_000_000_000.0]], index=idx, columns=cols)
    ebitda = pd.DataFrame([[2000.0]], index=idx, columns=cols)   # thousands BRL
    net_debt = pd.DataFrame([[1000.0]], index=idx, columns=cols)  # thousands BRL
    prices = pd.DataFrame([[30.0]], index=idx, columns=cols)

    result = compute_ev_ebitda_dynamic(shares, ebitda, net_debt, prices)

    # ev = 30 * (10_000_000_000 / 1000) + 1000 * 1000 = 300_000_000 + 1_000_000 = 301_000_000
    # ev_ebitda = 301_000_000 / (2000 * 1000) = 150.5
    assert result.iloc[0, 0] == pytest.approx(150.5)

"""
Integration test: Rust vs Python pivot+ffill for load_b3_data.

Verifies that cotahist_rs.pivot_and_ffill (via _pivot_and_ffill_rs) produces output
matching pd.pivot() + ffill on the same long DataFrame.
Skipped if cotahist_rs is not compiled.
"""
import sqlite3
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import cotahist_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="cotahist_rs not compiled")

from b3_pipeline import storage
from backtests.core.data import load_b3_data, _pivot_and_ffill_rs


# ── Fixture helpers ────────────────────────────────────────────────────────────

def _setup_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.sqlite")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    storage.init_db(conn)
    conn.commit()
    conn.close()
    return db_path


def _insert_price(conn, ticker, date_str, close=10.0, adj_close=None, volume=100_000):
    if adj_close is None:
        adj_close = close
    conn.execute(
        """INSERT INTO prices (ticker, isin_code, date, open, high, low, close,
                               adj_close, split_adj_high, split_adj_low, split_adj_close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ticker, "UNKNOWN", date_str, close, close, close, close,
         adj_close, close, close, close, volume)
    )
    conn.commit()


def _make_long_df(rows):
    """
    rows: list of (ticker, date_str, close, adj_close, fin_volume)
    Returns a long DataFrame as load_b3_data constructs it internally.
    """
    records = []
    for ticker, date_str, close, adj_close, fin_volume in rows:
        records.append({
            "date": pd.Timestamp(date_str),
            "ticker": ticker,
            "close": float(close),
            "adj_close": float(adj_close),
            "fin_volume": float(fin_volume),
        })
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestPivotAndFfillRsVsPython:

    def _python_pivot(self, df):
        adj_close = df.pivot(index="date", columns="ticker", values="adj_close").ffill()
        close_px  = df.pivot(index="date", columns="ticker", values="close").ffill()
        fin_vol   = df.pivot(index="date", columns="ticker", values="fin_volume")
        return adj_close, close_px, fin_vol

    def test_single_ticker_single_date_matches(self):
        df = _make_long_df([("PETR3", "2020-01-02", 100.0, 95.0, 1000.0)])
        py_adj, py_close, py_vol = self._python_pivot(df)
        rs_result = _pivot_and_ffill_rs(df)
        assert rs_result is not None
        rs_adj, rs_close, rs_vol = rs_result

        assert "PETR3" in rs_adj.columns
        assert abs(rs_adj.loc["2020-01-02", "PETR3"] - 95.0) < 1e-9
        assert abs(rs_close.loc["2020-01-02", "PETR3"] - 100.0) < 1e-9
        assert abs(rs_vol.loc["2020-01-02", "PETR3"] - 1000.0) < 1e-9

    def test_forward_fill_close_not_vol(self):
        df = _make_long_df([
            ("PETR3", "2020-01-02", 100.0, 95.0, 1000.0),
            ("VALE3", "2020-01-02", 200.0, 190.0, 2000.0),
            ("VALE3", "2020-01-03", 201.0, 191.0, 2100.0),
            # PETR3 has no row on 2020-01-03
        ])
        py_adj, py_close, py_vol = self._python_pivot(df)
        rs_adj, rs_close, rs_vol = _pivot_and_ffill_rs(df)

        # adj_close and close_px are ffilled
        assert abs(rs_adj.loc["2020-01-03", "PETR3"] - 95.0) < 1e-9
        assert abs(rs_close.loc["2020-01-03", "PETR3"] - 100.0) < 1e-9
        # fin_vol is NOT ffilled
        assert pd.isna(rs_vol.loc["2020-01-03", "PETR3"]), "fin_vol must not be ffilled"

    def test_large_multi_ticker_matches_python(self):
        """50 tickers x 100 dates with injected NaN gaps — full equality check."""
        n_tickers = 50
        n_dates = 100
        rows = []
        base = pd.Timestamp("2018-01-02")
        trading_days = pd.bdate_range(base, periods=n_dates).strftime("%Y-%m-%d").tolist()

        for t_idx in range(n_tickers):
            ticker = f"TEST{t_idx:02d}3"
            for d_idx, d in enumerate(trading_days):
                if t_idx % 5 == 0 and d_idx % 7 == 0:
                    continue  # introduce gaps to test ffill
                rows.append((ticker, d, 100.0 + t_idx, 95.0 + t_idx, 1000.0 + t_idx * 10))

        df = _make_long_df(rows)
        py_adj, py_close, py_vol = self._python_pivot(df)
        rs_adj, rs_close, rs_vol = _pivot_and_ffill_rs(df)

        assert rs_adj is not None

        # Sort columns for comparison
        common_cols = sorted(set(py_adj.columns) & set(rs_adj.columns))
        common_idx = py_adj.index.intersection(rs_adj.index)

        for col in common_cols:
            py_vals = py_adj.loc[common_idx, col].values
            rs_vals = rs_adj.loc[common_idx, col].values
            py_nan = np.isnan(py_vals)
            rs_nan = np.isnan(rs_vals)
            assert (py_nan == rs_nan).all(), f"NaN pattern mismatch for {col}"
            non_nan = ~py_nan
            if non_nan.any():
                np.testing.assert_allclose(
                    py_vals[non_nan], rs_vals[non_nan], atol=1e-9,
                    err_msg=f"Value mismatch for column {col}"
                )

    def test_end_to_end_via_load_b3_data(self, tmp_path):
        """Full load_b3_data call uses Rust path and returns correct wide DataFrames."""
        db_path = _setup_db(tmp_path)
        conn = sqlite3.connect(db_path)
        for ticker in ["PETR3", "VALE3"]:
            for date_str in ["2020-01-02", "2020-01-03"]:
                _insert_price(conn, ticker, date_str,
                               close=100.0, adj_close=95.0, volume=1_000_000)
        conn.close()

        adj_close, close_px, fin_vol = load_b3_data(db_path, "2020-01-01", "2020-01-31")

        assert "PETR3" in adj_close.columns
        assert "VALE3" in adj_close.columns
        assert len(adj_close.index) == 2
        # fin_vol should be volume / 100.0 = 10_000.0
        assert abs(fin_vol.loc["2020-01-02", "PETR3"] - 10_000.0) < 1e-6

    def test_date_index_is_datetime64(self):
        df = _make_long_df([
            ("PETR3", "2020-01-02", 100.0, 95.0, 1000.0),
            ("PETR3", "2020-01-03", 101.0, 96.0, 1100.0),
        ])
        rs_adj, rs_close, rs_vol = _pivot_and_ffill_rs(df)
        assert rs_adj.index.dtype == "datetime64[ns]", (
            f"Expected datetime64[ns] index, got {rs_adj.index.dtype}"
        )

    def test_column_name_matches_ticker_strings(self):
        df = _make_long_df([
            ("BOVA11", "2020-01-02", 100.0, 100.0, 500.0),
            ("ITUB4", "2020-01-02", 30.0, 28.0, 300.0),
        ])
        rs_adj, _, _ = _pivot_and_ffill_rs(df)
        assert "BOVA11" in rs_adj.columns
        assert "ITUB4" in rs_adj.columns

"""
Integration test: Rust vs Python implementation of compute_split_adjustment_factors.

Verifies that cotahist_rs.compute_split_adjustment (via the _compute_split_adjustment_rs
wrapper) produces output matching the pure Python implementation on the same fixture data.
Skipped if cotahist_rs is not compiled.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import date

try:
    import cotahist_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="cotahist_rs not compiled")

from b3_pipeline.adjustments import (
    compute_split_adjustment_factors, _compute_split_adjustment_rs,
    _MAX_CUMULATIVE_FACTOR,
)


def _make_prices(isin, dates, closes, open_=None, high=None, low=None):
    rows = []
    for i, (d, c) in enumerate(zip(dates, closes)):
        o = open_[i] if open_ else c
        h = high[i] if high else c * 1.01
        l = low[i] if low else c * 0.99
        rows.append({
            "isin_code": isin, "ticker": isin[:4] + "3", "date": d,
            "open": o, "high": h, "low": l, "close": c,
            "volume": 1_000_000.0, "quotation_factor": 1,
        })
    return pd.DataFrame(rows)


def _make_splits(isin, ex_dates, factors):
    return pd.DataFrame([
        {"isin_code": isin, "ex_date": ex_date, "split_factor": f}
        for ex_date, f in zip(ex_dates, factors)
    ])


class TestComputeSplitAdjustmentRsVsPython:

    def test_single_split_matches_python(self):
        dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
        closes = [100.0, 100.0, 50.0]
        prices = _make_prices("BRTEST000001", dates, closes)
        splits = _make_splits("BRTEST000001", [date(2020, 1, 6)], [0.5])

        py_result = compute_split_adjustment_factors(prices, splits)
        rs_result = _compute_split_adjustment_rs(prices, splits)

        assert rs_result is not None
        pd.testing.assert_series_equal(
            py_result["split_adj_close"].reset_index(drop=True),
            rs_result["split_adj_close"].reset_index(drop=True),
            check_names=False,
            atol=1e-9,
        )

    def test_multiple_isins_matches_python(self):
        isins = [f"BRTEST{i:06d}" for i in range(50)]
        all_prices = []
        all_splits = []
        dates = [date(2020, 1, 2 + i) for i in range(20)]

        for idx, isin in enumerate(isins):
            closes = [100.0] * 10 + [50.0] * 10
            all_prices.append(_make_prices(isin, dates, closes))
            if idx % 2 == 0:
                all_splits.append(_make_splits(isin, [dates[10]], [0.5]))

        prices = pd.concat(all_prices, ignore_index=True)
        splits = pd.concat(all_splits, ignore_index=True) if all_splits else pd.DataFrame(
            columns=["isin_code", "ex_date", "split_factor"]
        )

        py_result = compute_split_adjustment_factors(prices, splits)
        rs_result = _compute_split_adjustment_rs(prices, splits)

        assert rs_result is not None

        py_sorted = py_result.sort_values(["isin_code", "date"]).reset_index(drop=True)
        rs_sorted = rs_result.sort_values(["isin_code", "date"]).reset_index(drop=True)

        for col in ["split_adj_open", "split_adj_high", "split_adj_low", "split_adj_close"]:
            diff = (py_sorted[col] - rs_sorted[col]).abs()
            assert (diff < 1e-9).all(), f"Column {col} mismatch: max_diff={diff.max()}"

    def test_empty_splits_matches_python(self):
        dates = [date(2020, 1, 2), date(2020, 1, 3)]
        closes = [100.0, 100.0]
        prices = _make_prices("BRTEST000001", dates, closes)
        splits = pd.DataFrame(columns=["isin_code", "ex_date", "split_factor"])

        py_result = compute_split_adjustment_factors(prices, splits)
        rs_result = _compute_split_adjustment_rs(prices, splits)

        # With empty splits, Rust fast-path is not invoked (returns None); that is fine.
        # If it returns a result, it must match Python.
        if rs_result is not None:
            pd.testing.assert_series_equal(
                py_result["split_adj_close"].reset_index(drop=True),
                rs_result["split_adj_close"].reset_index(drop=True),
                atol=1e-9,
            )

    def test_two_splits_suffix_product_matches_python(self):
        dates = [date(2019, 12, 31), date(2020, 7, 1), date(2021, 6, 1)]
        closes = [400.0, 200.0, 100.0]
        prices = _make_prices("BRTEST000002", dates, closes)
        splits = _make_splits(
            "BRTEST000002",
            [date(2020, 6, 1), date(2021, 1, 1)],
            [0.5, 0.5],
        )

        py_result = compute_split_adjustment_factors(prices, splits)
        rs_result = _compute_split_adjustment_rs(prices, splits)

        assert rs_result is not None
        py_vals = py_result.sort_values("date")["split_adj_close"].values
        rs_vals = rs_result.sort_values("date")["split_adj_close"].values
        np.testing.assert_allclose(py_vals, rs_vals, atol=1e-9)

    def test_output_preserves_all_input_columns(self):
        """Rust result must contain all original columns from the input prices."""
        dates = [date(2020, 1, 2)]
        prices = _make_prices("BRTEST000001", dates, [100.0])
        splits = _make_splits("BRTEST000001", [date(2020, 1, 3)], [0.5])

        py_result = compute_split_adjustment_factors(prices, splits)
        rs_result = _compute_split_adjustment_rs(prices, splits)

        assert rs_result is not None
        for col in prices.columns:
            assert col in rs_result.columns, f"Column '{col}' missing from Rust result"

    def test_date_column_not_modified(self):
        """The 'date' column in the Rust result must not be corrupted."""
        dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
        prices = _make_prices("BRTEST000001", dates, [100.0, 100.0, 50.0])
        splits = _make_splits("BRTEST000001", [date(2020, 1, 6)], [0.5])

        py_result = compute_split_adjustment_factors(prices, splits)
        rs_result = _compute_split_adjustment_rs(prices, splits)

        assert rs_result is not None
        # Dates should be parseable and match
        py_dates = pd.to_datetime(py_result["date"]).sort_values().reset_index(drop=True)
        rs_dates = pd.to_datetime(rs_result["date"]).sort_values().reset_index(drop=True)
        pd.testing.assert_series_equal(py_dates, rs_dates, check_names=False)

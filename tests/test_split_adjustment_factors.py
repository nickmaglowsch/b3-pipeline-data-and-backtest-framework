"""
Unit tests for compute_split_adjustment_factors in b3_pipeline/adjustments.py.

Safety-net tests for the Rust rewrite in task-02. These tests must pass
before and after the Rust implementation is introduced.
"""
import datetime
import pandas as pd
import pytest

from b3_pipeline.adjustments import (
    compute_split_adjustment_factors,
    _MAX_CUMULATIVE_FACTOR,
)


def _make_prices_df(rows):
    """
    rows: list of dicts, each with keys:
      isin_code, date (str "YYYY-MM-DD" or datetime.date), open, high, low, close
    Returns a DataFrame with those columns plus: ticker (same as isin_code[:4]+"3"),
    volume (1_000_000), quotation_factor (1).
    """
    records = []
    for r in rows:
        records.append({
            "isin_code": r["isin_code"],
            "ticker": r["isin_code"][:4] + "3",
            "date": r["date"],
            "open": float(r["open"]),
            "high": float(r["high"]),
            "low": float(r["low"]),
            "close": float(r["close"]),
            "volume": 1_000_000,
            "quotation_factor": 1,
        })
    return pd.DataFrame(records)


def _make_splits_df(rows):
    """
    rows: list of dicts with keys: isin_code, ex_date, split_factor
    Returns a DataFrame with exactly those columns.
    """
    return pd.DataFrame(rows, columns=["isin_code", "ex_date", "split_factor"])


class TestComputeSplitAdjustmentFactors:

    def test_no_splits_returns_identity(self):
        """With no splits, split_adj_close == close exactly."""
        prices = _make_prices_df([
            {"isin_code": "BRTEST000001", "date": datetime.date(2020, 1, 2),
             "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0},
            {"isin_code": "BRTEST000001", "date": datetime.date(2020, 1, 3),
             "open": 102.0, "high": 106.0, "low": 100.0, "close": 104.0},
            {"isin_code": "BRTEST000001", "date": datetime.date(2020, 1, 6),
             "open": 104.0, "high": 108.0, "low": 102.0, "close": 106.0},
        ])
        splits = _make_splits_df([])

        result = compute_split_adjustment_factors(prices, splits)

        assert "split_adj_close" in result.columns
        assert "split_adj_open" in result.columns
        assert "split_adj_high" in result.columns
        assert "split_adj_low" in result.columns

        for _, row in result.iterrows():
            assert abs(row["split_adj_close"] - row["close"]) < 1e-9

    def test_single_split_adjusts_prices_before_ex_date(self):
        """A 2:1 split (split_factor=0.5) adjusts prices before ex_date by 0.5."""
        prices = _make_prices_df([
            {"isin_code": "BRTEST000001", "date": datetime.date(2020, 1, 2),
             "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"isin_code": "BRTEST000001", "date": datetime.date(2020, 1, 3),
             "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"isin_code": "BRTEST000001", "date": datetime.date(2020, 1, 6),
             "open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0},
        ])
        splits = _make_splits_df([{
            "isin_code": "BRTEST000001",
            "ex_date": datetime.date(2020, 1, 6),
            "split_factor": 0.5,
        }])

        result = compute_split_adjustment_factors(prices, splits)
        result = result.sort_values("date").reset_index(drop=True)

        # np.searchsorted(split_dates, price_date, side='left'):
        # for 2020-01-02: index=0 -> suffix[0]=0.5 -> adjusted
        # for 2020-01-03: index=0 -> suffix[0]=0.5 -> adjusted
        # for 2020-01-06 (== ex_date): index=0 -> suffix[0]=0.5 -> adjusted
        # All three dates land on suffix[0] because searchsorted 'left' includes the ex_date
        assert abs(result.loc[0, "split_adj_close"] - 50.0) < 1e-9  # 100 * 0.5
        assert abs(result.loc[1, "split_adj_close"] - 50.0) < 1e-9  # 100 * 0.5
        assert abs(result.loc[2, "split_adj_close"] - 25.0) < 1e-9  # 50 * 0.5

    def test_two_splits_suffix_product(self):
        """Two 2:1 splits produce suffix product 0.25 before both, 0.5 between them."""
        prices = _make_prices_df([
            {"isin_code": "BRTEST000002", "date": datetime.date(2019, 12, 31),
             "open": 400.0, "high": 410.0, "low": 390.0, "close": 400.0},
            {"isin_code": "BRTEST000002", "date": datetime.date(2020, 7, 1),
             "open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
            {"isin_code": "BRTEST000002", "date": datetime.date(2021, 6, 1),
             "open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0},
        ])
        splits = _make_splits_df([
            {"isin_code": "BRTEST000002", "ex_date": datetime.date(2020, 6, 1), "split_factor": 0.5},
            {"isin_code": "BRTEST000002", "ex_date": datetime.date(2021, 1, 1), "split_factor": 0.5},
        ])

        result = compute_split_adjustment_factors(prices, splits)
        result = result.sort_values("date").reset_index(drop=True)

        # 2019-12-31: before both splits -> factor = 0.5 * 0.5 = 0.25
        assert abs(result.loc[0, "split_adj_close"] - 100.0) < 1e-9  # 400 * 0.25

        # 2020-07-01: between the two splits -> factor = 0.5 (only second split remains)
        assert abs(result.loc[1, "split_adj_close"] - 100.0) < 1e-9  # 200 * 0.5

        # 2021-06-01: after both splits -> factor = 1.0
        assert abs(result.loc[2, "split_adj_close"] - 100.0) < 1e-9  # 100 * 1.0

    def test_split_applies_to_all_four_ohlc_columns(self):
        """All four OHLC columns are adjusted by the split factor."""
        prices = _make_prices_df([
            {"isin_code": "BRTEST000003", "date": datetime.date(2020, 1, 2),
             "open": 200.0, "high": 210.0, "low": 190.0, "close": 205.0},
        ])
        # split after this date
        splits = _make_splits_df([{
            "isin_code": "BRTEST000003",
            "ex_date": datetime.date(2020, 1, 3),
            "split_factor": 0.5,
        }])

        result = compute_split_adjustment_factors(prices, splits)

        assert abs(result.iloc[0]["split_adj_open"] - 100.0) < 1e-9
        assert abs(result.iloc[0]["split_adj_high"] - 105.0) < 1e-9
        assert abs(result.iloc[0]["split_adj_low"] - 95.0) < 1e-9
        assert abs(result.iloc[0]["split_adj_close"] - 102.5) < 1e-9

    def test_isin_not_in_splits_is_unchanged(self):
        """ISINs without splits get identity factors."""
        prices = _make_prices_df([
            {"isin_code": "ISIN_A000001", "date": datetime.date(2020, 1, 2),
             "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"isin_code": "ISIN_B000002", "date": datetime.date(2020, 1, 2),
             "open": 200.0, "high": 205.0, "low": 195.0, "close": 200.0},
        ])
        # Split only for ISIN_A
        splits = _make_splits_df([{
            "isin_code": "ISIN_A000001",
            "ex_date": datetime.date(2020, 1, 3),
            "split_factor": 0.5,
        }])

        result = compute_split_adjustment_factors(prices, splits)
        isin_b = result[result["isin_code"] == "ISIN_B000002"].iloc[0]
        assert abs(isin_b["split_adj_close"] - isin_b["close"]) < 1e-9

    def test_cumulative_factor_clamping(self):
        """Suffix product exceeding MAX_CUMULATIVE_FACTOR is clamped."""
        # Build a chain of large reverse splits whose product exceeds 100,000
        # e.g., 100 splits each with split_factor = 100.0 -> product = 100^100 >> 100_000
        n_splits = 10
        split_factor = 1000.0  # large enough that even a few exceed limit
        prices = _make_prices_df([
            {"isin_code": "BRTEST_CLAMP", "date": datetime.date(2000, 1, 2),
             "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0},
        ])
        splits_rows = []
        for i in range(n_splits):
            splits_rows.append({
                "isin_code": "BRTEST_CLAMP",
                "ex_date": datetime.date(2000 + i + 1, 1, 1),
                "split_factor": split_factor,
            })
        splits = _make_splits_df(splits_rows)

        result = compute_split_adjustment_factors(prices, splits)
        adj_close = result.iloc[0]["split_adj_close"]
        # Factor should be clamped to MAX_CUMULATIVE_FACTOR
        assert adj_close <= _MAX_CUMULATIVE_FACTOR + 1e-3

    def test_date_column_accepts_string_dates(self):
        """String date columns are handled without raising."""
        prices = _make_prices_df([
            {"isin_code": "BRTEST000001", "date": "2020-01-02",
             "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
            {"isin_code": "BRTEST000001", "date": "2020-01-03",
             "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
        ])
        splits = _make_splits_df([{
            "isin_code": "BRTEST000001",
            "ex_date": "2020-01-03",
            "split_factor": 0.5,
        }])

        # Should not raise
        result = compute_split_adjustment_factors(prices, splits)
        assert "split_adj_close" in result.columns

        # Prices before ex_date should be adjusted
        result = result.sort_values("date").reset_index(drop=True)
        assert abs(result.loc[0, "split_adj_close"] - 50.0) < 1e-9

    def test_original_columns_preserved(self):
        """All original columns are preserved; _date_ts temporary column is absent."""
        prices = _make_prices_df([
            {"isin_code": "BRTEST000001", "date": datetime.date(2020, 1, 2),
             "open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0},
        ])
        splits = _make_splits_df([{
            "isin_code": "BRTEST000001",
            "ex_date": datetime.date(2020, 1, 3),
            "split_factor": 0.5,
        }])

        result = compute_split_adjustment_factors(prices, splits)

        for col in prices.columns:
            assert col in result.columns, f"Column '{col}' missing from result"

        assert "_date_ts" not in result.columns

    def test_empty_splits_returns_identity_for_all_isins(self):
        """With 5 ISINs and empty splits, all adjusted columns equal originals."""
        rows = []
        for i in range(5):
            isin = f"BRTEST{i:06d}"
            for j in range(10):
                rows.append({
                    "isin_code": isin,
                    "date": datetime.date(2020, 1, 2 + j),
                    "open": 100.0 + i,
                    "high": 105.0 + i,
                    "low": 95.0 + i,
                    "close": 102.0 + i,
                })
        prices = _make_prices_df(rows)
        splits = _make_splits_df([])

        result = compute_split_adjustment_factors(prices, splits)

        for _, row in result.iterrows():
            assert abs(row["split_adj_close"] - row["close"]) < 1e-9
            assert abs(row["split_adj_open"] - row["open"]) < 1e-9

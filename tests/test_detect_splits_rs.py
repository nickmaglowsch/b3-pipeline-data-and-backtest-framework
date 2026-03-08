"""
Integration test: Rust vs Python implementation of detect_splits_from_prices.

Verifies that cotahist_rs.detect_splits (via the _detect_splits_rs wrapper) produces
output identical to the pure Python detect_splits_from_prices on the same fixture data.
Skipped if cotahist_rs is not compiled.
"""
import pytest
import pandas as pd
from datetime import date, timedelta

try:
    import cotahist_rs
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

pytestmark = pytest.mark.skipif(not HAS_RUST, reason="cotahist_rs not compiled")


def _make_multi_isin_prices(n_isins=20, n_days=50, inject_split_at=25):
    """
    Build a fixture with n_isins ISINs, each with n_days of price data.
    For even-numbered ISINs, inject a 2:1 forward split at day inject_split_at
    (price drops 50%).
    """
    rows = []
    base = date(2018, 1, 2)
    trading_days = []
    d = base
    while len(trading_days) < n_days:
        if d.weekday() < 5:
            trading_days.append(d)
        d += timedelta(days=1)

    for isin_idx in range(n_isins):
        isin = f"BRTESTX{isin_idx:05d}"
        for day_idx, day in enumerate(trading_days):
            if isin_idx % 2 == 0 and day_idx >= inject_split_at:
                close = 50.0
            else:
                close = 100.0
            rows.append({
                "isin_code": isin,
                "ticker": f"TEST{isin_idx:02d}3",
                "date": day,
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000.0,
                "quotation_factor": 1,
            })
    return pd.DataFrame(rows)


def _call_python_only(prices, existing, detect_nonstandard: bool = False):
    """
    Call the Python implementation directly, bypassing the Rust fast-path.
    """
    from b3_pipeline import config
    from b3_pipeline.adjustments import _normalize_date

    _common_ratios = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100]
    _tolerance = 0.08
    detected = []

    if prices.empty:
        return pd.DataFrame(
            columns=["isin_code", "ex_date", "action_type", "factor", "source"]
        )

    prices = prices.copy()
    prices["date"] = prices["date"].apply(_normalize_date)

    existing_keys: set = set()
    if not existing.empty:
        for _, row in existing.iterrows():
            ex = _normalize_date(row["ex_date"])
            if ex is not None:
                existing_keys.add((row["isin_code"], ex))

    for isin_code, group in prices.groupby("isin_code"):
        group = group.sort_values("date").reset_index(drop=True)
        if len(group) < 2:
            continue
        closes = group["close"].values
        dates = group["date"].values
        factors = group["quotation_factor"].values if "quotation_factor" in group.columns else None
        for i in range(1, len(closes)):
            prev_close = closes[i - 1]
            curr_close = closes[i]
            if prev_close <= 0 or curr_close <= 0:
                continue
            ratio = curr_close / prev_close
            if config.SPLIT_DETECTION_THRESHOLD_LOW <= ratio <= config.SPLIT_DETECTION_THRESHOLD_HIGH:
                continue
            jump_date = dates[i]
            already_recorded = False
            for lookback_i in range(max(0, i - 5), i + 1):
                if (isin_code, dates[lookback_i]) in existing_keys:
                    already_recorded = True
                    break
            if already_recorded:
                continue
            if factors is not None:
                prev_factor = factors[i - 1]
                curr_factor = factors[i]
                if prev_factor != curr_factor and prev_factor > 0 and curr_factor > 0:
                    factor_ratio = curr_factor / prev_factor
                    if abs(ratio - (1.0 / factor_ratio)) / max(abs(1.0 / factor_ratio), 0.001) < _tolerance:
                        continue
                    if abs(ratio - 1.0) < _tolerance:
                        continue
            matched = False
            for n in _common_ratios:
                target_forward = 1.0 / n
                if abs(ratio - target_forward) / target_forward < _tolerance:
                    detected.append({"isin_code": isin_code, "ex_date": jump_date,
                                     "action_type": config.EVENT_TYPE_STOCK_SPLIT,
                                     "factor": float(n), "source": "DETECTED"})
                    matched = True
                    break
                target_reverse = float(n)
                if abs(ratio - target_reverse) / target_reverse < _tolerance:
                    detected.append({"isin_code": isin_code, "ex_date": jump_date,
                                     "action_type": config.EVENT_TYPE_REVERSE_SPLIT,
                                     "factor": 1.0 / float(n), "source": "DETECTED"})
                    matched = True
                    break

            if not matched and detect_nonstandard:
                if ratio < 1.0:
                    factor = 1.0 / ratio
                    action_type = config.EVENT_TYPE_STOCK_SPLIT
                else:
                    factor = 1.0 / ratio
                    action_type = config.EVENT_TYPE_REVERSE_SPLIT
                detected.append({"isin_code": isin_code, "ex_date": jump_date,
                                 "action_type": action_type,
                                 "factor": factor, "source": "DETECTED_NONSTANDARD"})

    if detected:
        df = pd.DataFrame(detected)
        return df.drop_duplicates(subset=["isin_code", "ex_date", "action_type"])
    return pd.DataFrame(columns=["isin_code", "ex_date", "action_type", "factor", "source"])


class TestDetectSplitsRsVsPython:

    def test_same_detections_no_existing_actions(self):
        """Rust and Python find the same splits on a multi-ISIN fixture."""
        from b3_pipeline.adjustments import _detect_splits_rs

        prices = _make_multi_isin_prices(n_isins=20, n_days=50, inject_split_at=25)
        existing = pd.DataFrame(
            columns=["isin_code", "ex_date", "action_type", "factor", "source"]
        )

        py_result = _call_python_only(prices, existing)
        rs_result = _detect_splits_rs(prices, existing, detect_nonstandard=False)

        assert rs_result is not None, "Rust implementation returned None (not compiled?)"

        # Sort both results for deterministic comparison
        py_sorted = py_result.sort_values(["isin_code", "ex_date"]).reset_index(drop=True)
        rs_sorted = rs_result.sort_values(["isin_code", "ex_date"]).reset_index(drop=True)

        assert len(py_sorted) == len(rs_sorted), (
            f"Row count mismatch: Python={len(py_sorted)}, Rust={len(rs_sorted)}"
        )
        pd.testing.assert_frame_equal(
            py_sorted[["isin_code", "action_type", "source"]].reset_index(drop=True),
            rs_sorted[["isin_code", "action_type", "source"]].reset_index(drop=True),
            check_like=False,
        )
        # factor should match within 1%
        diff = (py_sorted["factor"] - rs_sorted["factor"]).abs()
        assert (diff < 0.01).all(), f"Factor mismatch: {diff.max()}"

    def test_same_suppression_with_existing_actions(self):
        """Rust suppresses the same splits as Python when existing actions cover the date."""
        from b3_pipeline.adjustments import _detect_splits_rs, detect_splits_from_prices

        prices = _make_multi_isin_prices(n_isins=4, n_days=50, inject_split_at=25)
        # Add an existing action for isin_idx=0 at the split date
        split_date = prices[prices["isin_code"] == "BRTESTX00000"]["date"].sort_values().iloc[25]
        existing = pd.DataFrame([{
            "isin_code": "BRTESTX00000",
            "ex_date": split_date,
            "action_type": "STOCK_SPLIT",
            "factor": 2.0,
            "source": "B3",
        }])

        py_result = _call_python_only(prices, existing)
        rs_result = _detect_splits_rs(prices, existing, detect_nonstandard=False)

        assert rs_result is not None
        py_isins = set(py_result["isin_code"].tolist())
        rs_isins = set(rs_result["isin_code"].tolist())
        assert py_isins == rs_isins, (
            f"Suppression mismatch: Python ISINs={py_isins}, Rust ISINs={rs_isins}"
        )

    def test_nonstandard_flag_same_results(self):
        """Rust and Python produce identical results with detect_nonstandard=True."""
        from b3_pipeline.adjustments import _detect_splits_rs, detect_splits_from_prices

        isin = "BRTESTXNONSTD"
        days = [date(2020, 1, 2 + i) for i in range(10)]
        closes = [10.0] * 5 + [23.7] * 5  # 2.37x jump — non-standard
        rows = [{"isin_code": isin, "ticker": "TSTS3", "date": d,
                 "open": c, "high": c, "low": c, "close": c,
                 "volume": 1_000_000.0, "quotation_factor": 1}
                for d, c in zip(days, closes)]
        prices = pd.DataFrame(rows)
        existing = pd.DataFrame(
            columns=["isin_code", "ex_date", "action_type", "factor", "source"]
        )

        # Compare Python and Rust with the SAME detect_nonstandard=True flag
        py_result = _call_python_only(prices, existing, detect_nonstandard=True)
        rs_result = _detect_splits_rs(prices, existing, detect_nonstandard=True)

        assert rs_result is not None
        assert len(rs_result) == len(py_result)
        if not rs_result.empty:
            assert rs_result.iloc[0]["source"] == "DETECTED_NONSTANDARD"
            assert rs_result.iloc[0]["isin_code"] == py_result.iloc[0]["isin_code"]

"""
Tests for Phase 2: Pipeline-level price-jump split detection.

These tests verify:
1. A 2:1 forward split is detected (price drops ~50% overnight)
2. A 2:1 reverse split is detected (price rises ~2x overnight)
3. The detector does NOT fire when a stock_action already exists for that date
4. The detector does NOT fire when a quotation_factor transition explains the jump
5. The resulting stock_actions have source='DETECTED'
6. Non-standard large jumps are not auto-corrected but logged/returned separately
"""
import pandas as pd
import pytest
from datetime import date, timedelta

from b3_pipeline.adjustments import detect_splits_from_prices, filter_fatcot_redundant_splits


def _make_prices_df(isin: str, dates_and_closes: list) -> pd.DataFrame:
    """Helper to build a minimal prices DataFrame."""
    rows = []
    for d, close in dates_and_closes:
        rows.append({
            "isin_code": isin,
            "ticker": isin[:4] + "3",
            "date": d,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
            "quotation_factor": 1,
        })
    return pd.DataFrame(rows)


def _make_stock_actions(isin: str, ex_date: date, action_type: str = "STOCK_SPLIT", factor: float = 2.0) -> pd.DataFrame:
    return pd.DataFrame([{
        "isin_code": isin,
        "ex_date": ex_date,
        "action_type": action_type,
        "factor": factor,
        "source": "B3",
    }])


class TestSplitDetection:
    """Test the pipeline-level split detector."""

    def _make_date_sequence(self, n: int, start: date = date(2020, 1, 2)) -> list:
        """Generate n consecutive trading dates (Mon-Fri)."""
        dates = []
        d = start
        while len(dates) < n:
            if d.weekday() < 5:  # Mon=0 ... Fri=4
                dates.append(d)
            d += timedelta(days=1)
        return dates

    def test_detects_2to1_forward_split(self):
        """
        GIVEN a price series where the close drops ~50% on one day (2:1 split)
        AND no stock_action exists for that date
        WHEN detect_splits_from_prices is called
        THEN it should return a stock_action with action_type=STOCK_SPLIT, factor=2.0
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)
        # Prices before split: 100.0; after split: 50.0
        closes = [100.0] * 5 + [50.0] * 5
        prices = _make_prices_df(isin, list(zip(dates, closes)))
        existing = pd.DataFrame(columns=["isin_code", "ex_date", "action_type", "factor", "source"])

        result = detect_splits_from_prices(prices, existing)

        assert not result.empty, "Should detect at least one split"
        assert result.iloc[0]["isin_code"] == isin
        assert result.iloc[0]["source"] == "DETECTED"
        # factor should be ~2.0 (the split ratio for a 2:1 forward split)
        assert abs(result.iloc[0]["factor"] - 2.0) < 0.2, (
            f"Expected factor ~2.0, got {result.iloc[0]['factor']}"
        )
        assert result.iloc[0]["action_type"] == "STOCK_SPLIT"

    def test_detects_2to1_reverse_split(self):
        """
        GIVEN a price series where the close rises ~2x on one day (2:1 reverse split)
        AND no stock_action exists for that date
        WHEN detect_splits_from_prices is called
        THEN it should return a stock_action with action_type=REVERSE_SPLIT
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)
        closes = [50.0] * 5 + [100.0] * 5
        prices = _make_prices_df(isin, list(zip(dates, closes)))
        existing = pd.DataFrame(columns=["isin_code", "ex_date", "action_type", "factor", "source"])

        result = detect_splits_from_prices(prices, existing)

        assert not result.empty, "Should detect the reverse split"
        assert result.iloc[0]["source"] == "DETECTED"
        assert result.iloc[0]["action_type"] == "REVERSE_SPLIT"

    def test_no_detection_when_action_already_exists(self):
        """
        GIVEN a price series with a 2:1 split pattern
        AND a stock_action already exists on the jump date
        WHEN detect_splits_from_prices is called
        THEN it should NOT return any new detections for that date
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)
        closes = [100.0] * 5 + [50.0] * 5
        prices = _make_prices_df(isin, list(zip(dates, closes)))

        # The jump happens on dates[5]
        existing = _make_stock_actions(isin, dates[5], "STOCK_SPLIT", 2.0)

        result = detect_splits_from_prices(prices, existing)

        # Should not add a duplicate detection
        assert result.empty, (
            "Should not detect a split when a stock_action already exists for that date"
        )

    def test_no_detection_for_fatcot_transition(self):
        """
        GIVEN a price series where the ~1000x drop is explained by a quotation_factor change
        (fatcot changes from 1000 to 1 on that date, meaning prices were already divided)
        WHEN detect_splits_from_prices is called
        THEN it should NOT detect a split for that transition
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)
        # Simulate: before transition, fatcot=1000 and prices are already normalized (e.g., 28.0)
        # after transition, fatcot=1 and prices are still 28.0 (continuous after normalization)
        # A real fatcot transition after normalization produces NO jump -- prices are continuous
        closes = [28.0] * 10  # No jump at all after normalization
        rows = []
        for i, (d, c) in enumerate(zip(dates, closes)):
            rows.append({
                "isin_code": isin,
                "ticker": "TEST3",
                "date": d,
                "open": c, "high": c, "low": c, "close": c,
                "volume": 1_000_000.0,
                "quotation_factor": 1000 if i < 5 else 1,
            })
        prices = pd.DataFrame(rows)
        existing = pd.DataFrame(columns=["isin_code", "ex_date", "action_type", "factor", "source"])

        result = detect_splits_from_prices(prices, existing)

        assert result.empty, (
            "Should not detect a split for a fatcot transition where prices are already normalized"
        )

    def test_normal_price_movement_not_flagged(self):
        """
        GIVEN a price series with no splits (normal up/down moves within 30%)
        WHEN detect_splits_from_prices is called
        THEN it should return empty DataFrame
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(20)
        # Normal oscillation between 95 and 105
        closes = [100.0, 102.0, 98.0, 103.0, 101.0, 99.0, 104.0, 97.0, 105.0, 95.0,
                  100.0, 102.0, 98.0, 103.0, 101.0, 99.0, 104.0, 97.0, 105.0, 95.0]
        prices = _make_prices_df(isin, list(zip(dates, closes)))
        existing = pd.DataFrame(columns=["isin_code", "ex_date", "action_type", "factor", "source"])

        result = detect_splits_from_prices(prices, existing)

        assert result.empty, "Normal price movements should not be flagged as splits"

    def test_detected_source_label(self):
        """
        GIVEN a detected split
        WHEN stored in stock_actions
        THEN source must be 'DETECTED' (not 'B3') for auditability
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)
        closes = [100.0] * 5 + [50.0] * 5
        prices = _make_prices_df(isin, list(zip(dates, closes)))
        existing = pd.DataFrame(columns=["isin_code", "ex_date", "action_type", "factor", "source"])

        result = detect_splits_from_prices(prices, existing)

        assert not result.empty
        assert all(result["source"] == "DETECTED"), (
            "All detected splits must have source='DETECTED'"
        )

    def test_fatcot_redundant_split_is_filtered(self):
        """
        GIVEN a B3 API split (STOCK_SPLIT, factor=1000) on a date
        AND the prices show a quotation_factor transition from 1000 to 1 on the same date
        WHEN filter_fatcot_redundant_splits is called
        THEN the B3 API split should be removed (it's redundant with fatcot normalization)
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)

        # Prices are already normalized by parser -- continuous at 28.0
        rows = []
        for i, d in enumerate(dates):
            rows.append({
                "isin_code": isin,
                "ticker": "TEST3",
                "date": d,
                "open": 28.0, "high": 28.5, "low": 27.5, "close": 28.0,
                "volume": 1_000_000.0,
                "quotation_factor": 1000 if i < 5 else 1,
            })
        prices = pd.DataFrame(rows)

        # B3 API reported a 1000:1 split on the transition date
        stock_actions = pd.DataFrame([{
            "isin_code": isin,
            "ex_date": dates[5],
            "action_type": "STOCK_SPLIT",
            "factor": 1000.0,
            "source": "B3",
        }])

        filtered = filter_fatcot_redundant_splits(stock_actions, prices)

        assert filtered.empty, (
            "B3 API split matching a fatcot transition should be filtered out "
            "to prevent double-adjustment"
        )

    def test_nonstandard_ratio_detected_when_enabled(self):
        """
        GIVEN a price series with a 2.37x jump (non-standard ratio)
        AND detect_nonstandard=True
        WHEN detect_splits_from_prices is called
        THEN it should return a stock_action with source='DETECTED_NONSTANDARD'
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)
        # Price rises from 10.0 to 23.7 (ratio=2.37x -> reverse split)
        closes = [10.0] * 5 + [23.7] * 5
        prices = _make_prices_df(isin, list(zip(dates, closes)))
        existing = pd.DataFrame(columns=["isin_code", "ex_date", "action_type", "factor", "source"])

        result = detect_splits_from_prices(prices, existing, detect_nonstandard=True)

        assert not result.empty, "Should detect non-standard ratio when enabled"
        row = result.iloc[0]
        assert row["source"] == "DETECTED_NONSTANDARD"
        assert row["action_type"] == "REVERSE_SPLIT"
        # factor should be 1/2.37 ≈ 0.422
        assert abs(row["factor"] - (1.0 / 2.37)) < 0.01, (
            f"Expected factor ~{1.0/2.37:.4f}, got {row['factor']}"
        )

    def test_nonstandard_ratio_not_detected_by_default(self):
        """
        GIVEN a price series with a 2.37x jump (non-standard ratio)
        AND detect_nonstandard=False (default)
        WHEN detect_splits_from_prices is called
        THEN it should NOT return any detection for that jump
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)
        closes = [10.0] * 5 + [23.7] * 5
        prices = _make_prices_df(isin, list(zip(dates, closes)))
        existing = pd.DataFrame(columns=["isin_code", "ex_date", "action_type", "factor", "source"])

        result = detect_splits_from_prices(prices, existing, detect_nonstandard=False)

        assert result.empty, (
            "Non-standard ratios should NOT be detected when detect_nonstandard=False"
        )

    def test_non_fatcot_split_preserved(self):
        """
        GIVEN a B3 API split where there is NO fatcot transition
        WHEN filter_fatcot_redundant_splits is called
        THEN the split should be preserved (not filtered)
        """
        isin = "BRTEST1TEST2"
        dates = self._make_date_sequence(10)

        # All fatcot=1, normal prices with a real split
        rows = []
        for i, d in enumerate(dates):
            close = 100.0 if i < 5 else 50.0
            rows.append({
                "isin_code": isin,
                "ticker": "TEST3",
                "date": d,
                "open": close, "high": close, "low": close, "close": close,
                "volume": 1_000_000.0,
                "quotation_factor": 1,
            })
        prices = pd.DataFrame(rows)

        stock_actions = pd.DataFrame([{
            "isin_code": isin,
            "ex_date": dates[5],
            "action_type": "STOCK_SPLIT",
            "factor": 2.0,
            "source": "B3",
        }])

        filtered = filter_fatcot_redundant_splits(stock_actions, prices)

        assert len(filtered) == 1, (
            "Real splits (no fatcot transition) should NOT be filtered"
        )

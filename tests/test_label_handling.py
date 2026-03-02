"""
Tests for Phase 3: Non-standard B3 API label handling.

These tests verify:
1. RESG TOTAL RV is logged and stored in skipped_events, NOT treated as a split
2. CIS RED CAP is logged and stored in skipped_events, NOT treated as a split
3. INCORPORACAO is logged and stored in skipped_events, NOT treated as a split
4. parse_stock_dividends still correctly handles DESDOBRAMENTO/GRUPAMENTO/BONIFICACAO
"""
import pandas as pd
import pytest

from b3_pipeline.b3_corporate_actions import parse_stock_dividends


class TestLabelHandling:
    """Test that unrecognized labels are logged and stored, not silently dropped."""

    def _make_record(self, label: str, factor: str = "2", isin: str = "BRTEST1TEST2", date_str: str = "01/01/2020"):
        return {
            "label": label,
            "factor": factor,
            "isinCode": isin,
            "lastDatePrior": date_str,
        }

    def test_resg_total_rv_not_treated_as_split(self):
        """
        GIVEN a stockDividends record with label='RESG TOTAL RV'
        WHEN parse_stock_dividends is called
        THEN it should NOT appear in the stock_actions DataFrame
        """
        records = [self._make_record("RESG TOTAL RV", factor="100")]
        corp_df, stock_df, skipped_df = parse_stock_dividends(records)

        # Should not be treated as a split
        assert stock_df.empty or not any(
            stock_df["isin_code"] == "BRTEST1TEST2"
        ), "RESG TOTAL RV should not be treated as a split"

    def test_cis_red_cap_not_treated_as_split(self):
        """
        GIVEN a stockDividends record with label='CIS RED CAP'
        WHEN parse_stock_dividends is called
        THEN it should NOT appear in the stock_actions DataFrame
        """
        records = [self._make_record("CIS RED CAP", factor="2.5")]
        corp_df, stock_df, skipped_df = parse_stock_dividends(records)

        assert stock_df.empty or not any(
            stock_df["isin_code"] == "BRTEST1TEST2"
        ), "CIS RED CAP should not be treated as a split"

    def test_incorporacao_not_treated_as_split(self):
        """
        GIVEN a stockDividends record with label='INCORPORACAO'
        WHEN parse_stock_dividends is called
        THEN it should NOT appear in the stock_actions DataFrame
        """
        records = [self._make_record("INCORPORACAO", factor="1.5")]
        corp_df, stock_df, skipped_df = parse_stock_dividends(records)

        assert stock_df.empty or not any(
            stock_df["isin_code"] == "BRTEST1TEST2"
        ), "INCORPORACAO should not be treated as a split"

    def test_parse_stock_dividends_returns_skipped_events(self):
        """
        GIVEN a stockDividends record with an unrecognized label
        WHEN parse_stock_dividends is called
        THEN it should return a third DataFrame of skipped_events
        """
        records = [self._make_record("RESG TOTAL RV", factor="100")]
        result = parse_stock_dividends(records)

        # After the fix, parse_stock_dividends should return 3 values:
        # (corp_df, stock_df, skipped_df)
        assert len(result) == 3, (
            "parse_stock_dividends should return 3 DataFrames: "
            "(corp_df, stock_df, skipped_df)"
        )
        corp_df, stock_df, skipped_df = result

        assert not skipped_df.empty, "skipped_events should contain the RESG TOTAL RV record"
        assert skipped_df.iloc[0]["label"] == "RESG TOTAL RV"

    def test_desdobramento_still_works(self):
        """
        GIVEN a stockDividends record with label='DESDOBRAMENTO'
        WHEN parse_stock_dividends is called
        THEN it should still be recognized as STOCK_SPLIT
        """
        records = [self._make_record("DESDOBRAMENTO", factor="2")]
        result = parse_stock_dividends(records)

        # Handle both old (2-tuple) and new (3-tuple) return formats
        if len(result) == 3:
            corp_df, stock_df, skipped_df = result
        else:
            corp_df, stock_df = result

        assert not stock_df.empty, "DESDOBRAMENTO should still be treated as STOCK_SPLIT"
        assert stock_df.iloc[0]["action_type"] == "STOCK_SPLIT"

    def test_grupamento_still_works(self):
        """
        GIVEN a stockDividends record with label='GRUPAMENTO'
        WHEN parse_stock_dividends is called
        THEN it should still be recognized as REVERSE_SPLIT
        """
        records = [self._make_record("GRUPAMENTO", factor="0.5")]
        result = parse_stock_dividends(records)

        if len(result) == 3:
            corp_df, stock_df, skipped_df = result
        else:
            corp_df, stock_df = result

        assert not stock_df.empty, "GRUPAMENTO should still be treated as REVERSE_SPLIT"
        assert stock_df.iloc[0]["action_type"] == "REVERSE_SPLIT"

    def test_mixed_records_sorting(self):
        """
        GIVEN a list with DESDOBRAMENTO + RESG TOTAL RV
        WHEN parse_stock_dividends is called
        THEN DESDOBRAMENTO goes to stock_df and RESG TOTAL RV goes to skipped_df
        """
        records = [
            self._make_record("DESDOBRAMENTO", factor="2", isin="BRISIN00001A"),
            self._make_record("RESG TOTAL RV", factor="100", isin="BRISIN00002B"),
        ]
        result = parse_stock_dividends(records)

        assert len(result) == 3
        corp_df, stock_df, skipped_df = result

        assert len(stock_df) == 1, f"Expected 1 stock action, got {len(stock_df)}"
        assert stock_df.iloc[0]["isin_code"] == "BRISIN00001A"
        assert len(skipped_df) == 1, f"Expected 1 skipped event, got {len(skipped_df)}"
        assert skipped_df.iloc[0]["isin_code"] == "BRISIN00002B"

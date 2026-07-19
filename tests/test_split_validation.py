"""Tests for validate_split_actions_against_prices (B3 factor vs observed prices)."""
import pandas as pd
import pytest

from b3_pipeline.adjustments import validate_split_actions_against_prices


def _prices(isin, closes, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "isin_code": isin,
        "date": dates.strftime("%Y-%m-%d"),
        "close": closes,
    })


def _action(isin, ex_date, action_type, factor, source="B3"):
    return pd.DataFrame([{
        "isin_code": isin, "ex_date": ex_date,
        "action_type": action_type, "factor": factor, "source": source,
    }])


def test_matching_split_is_kept():
    # 2:1 split on day 5: price 100 -> 50, factor 2 matches
    px = _prices("BRTESTACNOR1", [100, 100, 100, 100, 100, 50, 50, 50])
    ex = px.iloc[5]["date"]
    actions = _action("BRTESTACNOR1", ex, "STOCK_SPLIT", 2.0)
    out = validate_split_actions_against_prices(actions, px)
    assert len(out) == 1
    assert out.iloc[0]["factor"] == 2.0
    assert out.iloc[0]["source"] == "B3"


def test_phantom_reverse_split_is_dropped():
    # B3 claims 1-for-100 reverse split but price never moved (TIMS3 case)
    px = _prices("BRTIMSACNOR5", [22.0, 22.1, 21.9, 22.0, 22.6, 22.4, 22.5, 22.3])
    ex = px.iloc[4]["date"]
    actions = _action("BRTIMSACNOR5", ex, "REVERSE_SPLIT", 0.01)
    out = validate_split_actions_against_prices(actions, px)
    assert out.empty


def test_mislabeled_factor_is_corrected():
    # Price halves (a real 2:1 split) but B3 says 1-for-40 reverse (VIVT3 case)
    px = _prices("BRVIVTACNOR0", [51, 51, 50, 51, 51, 25.5, 25.7, 26.4])
    ex = px.iloc[5]["date"]
    actions = _action("BRVIVTACNOR0", ex, "REVERSE_SPLIT", 0.025)
    out = validate_split_actions_against_prices(actions, px)
    assert len(out) == 1
    assert out.iloc[0]["action_type"] == "STOCK_SPLIT"
    assert out.iloc[0]["factor"] == pytest.approx(2.0)
    assert out.iloc[0]["source"] == "B3_CORRECTED"


def test_non_b3_and_dividend_rows_untouched():
    px = _prices("BRTESTACNOR1", [100] * 8)
    ex = px.iloc[4]["date"]
    detected = _action("BRTESTACNOR1", ex, "STOCK_SPLIT", 2.0, source="DETECTED")
    out = validate_split_actions_against_prices(detected, px)
    assert len(out) == 1 and out.iloc[0]["source"] == "DETECTED"

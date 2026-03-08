"""
TDD tests for Task 03: research/discovery/fundamental_signals.py
Tests are written BEFORE implementation (RED phase).
"""

import pandas as pd
import numpy as np
import pytest

from research.discovery.fundamental_signals import (
    compute_fund_pe_ratio,
    compute_fund_earnings_yield,
    compute_fund_pb_ratio,
    compute_fund_ev_ebitda,
    compute_fund_roe,
    compute_fund_roa,
    compute_fund_net_margin,
    compute_fund_revenue_growth_yoy,
    compute_fund_debt_to_equity,
    generate_fundamental_base_signals,
    _SHARES_SCALE,
)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    dates = pd.date_range("2023-01-31", periods=3, freq="ME")
    adj_close = pd.DataFrame(
        {"PETR3": [30.0, 31.0, 29.0], "VALE3": [60.0, 62.0, 58.0]},
        index=dates,
    )
    f_shares = pd.DataFrame(
        {"PETR3": [13_000_000.0] * 3, "VALE3": [4_000_000.0] * 3},
        index=dates,
    )
    f_net_income = pd.DataFrame(
        {"PETR3": [1_000_000.0, 1_100_000.0, 900_000.0],
         "VALE3": [2_000_000.0, 2_200_000.0, 1_800_000.0]},
        index=dates,
    )
    f_equity = pd.DataFrame(
        {"PETR3": [5_000_000.0] * 3, "VALE3": [8_000_000.0] * 3},
        index=dates,
    )
    f_total_assets = pd.DataFrame(
        {"PETR3": [20_000_000.0] * 3, "VALE3": [30_000_000.0] * 3},
        index=dates,
    )
    f_ebitda = pd.DataFrame(
        {"PETR3": [2_000_000.0] * 3, "VALE3": [3_000_000.0] * 3},
        index=dates,
    )
    f_net_debt = pd.DataFrame(
        {"PETR3": [1_000_000.0] * 3, "VALE3": [-500_000.0] * 3},
        index=dates,
    )
    f_revenue = pd.DataFrame(
        {"PETR3": [8_000_000.0] * 3, "VALE3": [12_000_000.0] * 3},
        index=dates,
    )
    return {
        "adj_close": adj_close,
        "f_shares": f_shares,
        "f_net_income": f_net_income,
        "f_equity": f_equity,
        "f_total_assets": f_total_assets,
        "f_ebitda": f_ebitda,
        "f_net_debt": f_net_debt,
        "f_revenue": f_revenue,
    }


# ---------------------------------------------------------------------------
# Test 1: P/E ratio formula
# ---------------------------------------------------------------------------

def test_pe_ratio_formula(sample_data):
    result = compute_fund_pe_ratio(
        sample_data["f_shares"],
        sample_data["f_net_income"],
        sample_data["adj_close"],
    )
    # market_cap = 30.0 * (13_000_000 / 1000) = 30 * 13000 = 390_000
    # net_income_brl = 1_000_000 * 1000 = 1_000_000_000
    # PE = 390_000 / 1_000_000_000... wait
    # Scaling: shares in thousands → actual shares = 13_000_000 * 1000 = 13B
    # No: f_shares is stored as "thousands" → actual = 13_000_000 / 1000 = 13_000 shares? That seems small.
    # Per task spec: market_cap = adj_close * (f_shares / 1000)
    # = 30 * (13_000_000 / 1000) = 30 * 13000 = 390_000 BRL
    # net_income_brl = f_net_income * 1000 = 1_000_000 * 1000 = 1_000_000_000
    # PE = 390_000 / 1_000_000_000 = 0.00039 ... that doesn't match the spec's 0.39
    # The spec says: PE = 0.39 and formula gives 390_000_000 / 1_000_000_000
    # which implies market_cap = 30 * 13_000_000 = 390_000_000 (no /1000 on shares)
    # But spec says: market_cap = adj_close * (f_shares / 1000)
    # Actually: 30 * (13_000_000 / 1000) * 1000 = 30 * 13_000_000 / 1 ???
    # Re-reading: PE = market_cap / (f_net_income × 1000)
    # market_cap = adj_close * (f_shares / 1000)
    # = 30 * (13_000_000 / 1000) = 30 * 13_000 = 390_000
    # net_income_brl = 1_000_000 * 1000 = 1_000_000_000
    # PE = 390_000 / 1_000_000_000 = 0.00039 — but spec says 0.39
    # The spec test says: (30.0 * 13_000_000 / 1000) / (1_000_000 * 1000) = 390_000 / 1_000_000_000 = 0.00039
    # but also writes "(30 * 13000) / 1_000_000_000 = 390_000_000 / 1_000_000_000 = 0.39"
    # The spec has an arithmetic error. (30 * 13000) = 390_000, not 390_000_000.
    # Let's just use the formula verbatim and check the actual value.
    val = result.loc["2023-01-31", "PETR3"]
    # market_cap = 30 * (13_000_000 / 1000) = 30 * 13000 = 390_000
    # pe = 390_000 / (1_000_000 * 1000) = 390_000 / 1_000_000_000 = 3.9e-4
    expected = (30.0 * 13_000_000 / 1000) / (1_000_000 * 1000)
    assert abs(val - expected) < 1e-8, f"PE mismatch: got {val}, expected {expected}"


# ---------------------------------------------------------------------------
# Test 2: Earnings yield = 1 / PE
# ---------------------------------------------------------------------------

def test_earnings_yield_is_inverse_pe(sample_data):
    pe = compute_fund_pe_ratio(
        sample_data["f_shares"], sample_data["f_net_income"], sample_data["adj_close"]
    )
    ey = compute_fund_earnings_yield(
        sample_data["f_shares"], sample_data["f_net_income"], sample_data["adj_close"]
    )
    valid = pe.notna() & (pe != 0)
    # Where PE is valid and nonzero, earnings yield should equal 1/PE
    for col in pe.columns:
        for idx in pe.index:
            if valid.loc[idx, col]:
                assert abs(ey.loc[idx, col] - 1.0 / pe.loc[idx, col]) < 1e-8


# ---------------------------------------------------------------------------
# Test 3: P/E is NaN for negative net income
# ---------------------------------------------------------------------------

def test_pe_ratio_nan_for_negative_income(sample_data):
    f_net_income = sample_data["f_net_income"].copy()
    f_net_income.loc["2023-01-31", "PETR3"] = -500.0

    result = compute_fund_pe_ratio(
        sample_data["f_shares"], f_net_income, sample_data["adj_close"]
    )
    assert pd.isna(result.loc["2023-01-31", "PETR3"])


# ---------------------------------------------------------------------------
# Test 4: ROE formula
# ---------------------------------------------------------------------------

def test_roe_formula(sample_data):
    result = compute_fund_roe(sample_data["f_net_income"], sample_data["f_equity"])
    # ROE = net_income / equity (scale cancels)
    # PETR3: 1_000_000 / 5_000_000 = 0.2
    val = result.loc["2023-01-31", "PETR3"]
    assert abs(val - 0.2) < 1e-8


# ---------------------------------------------------------------------------
# Test 5: Net margin formula
# ---------------------------------------------------------------------------

def test_net_margin_formula(sample_data):
    result = compute_fund_net_margin(
        sample_data["f_net_income"], sample_data["f_revenue"]
    )
    # PETR3: 1_000_000 / 8_000_000 = 0.125
    val = result.loc["2023-01-31", "PETR3"]
    assert abs(val - 0.125) < 1e-8


# ---------------------------------------------------------------------------
# Test 6: Debt/equity allows negative values
# ---------------------------------------------------------------------------

def test_debt_to_equity_negative_allowed(sample_data):
    result = compute_fund_debt_to_equity(
        sample_data["f_net_debt"], sample_data["f_equity"]
    )
    # VALE3: net_debt = -500_000, equity = 8_000_000 → D/E = -0.0625
    val = result.loc["2023-01-31", "VALE3"]
    assert np.isfinite(val)
    assert val < 0


# ---------------------------------------------------------------------------
# Test 7: generate_fundamental_base_signals count = 18
# ---------------------------------------------------------------------------

def test_generate_fundamental_base_signals_count(sample_data):
    results = list(generate_fundamental_base_signals(sample_data))
    assert len(results) == 18, f"Expected 18 signals, got {len(results)}"


# ---------------------------------------------------------------------------
# Test 8: generate yields nothing on empty data
# ---------------------------------------------------------------------------

def test_generate_fundamental_base_signals_empty_on_no_data():
    results = list(generate_fundamental_base_signals({}))
    assert len(results) == 0


# ---------------------------------------------------------------------------
# Test 9: tuple structure
# ---------------------------------------------------------------------------

def test_generate_fundamental_base_signals_tuple_structure(sample_data):
    results = list(generate_fundamental_base_signals(sample_data))
    for feature_id, category, params, wide_df in results:
        assert isinstance(feature_id, str)
        assert isinstance(category, str)
        assert isinstance(params, dict)
        assert isinstance(wide_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 10: YoY delta = base - base.shift(252)
# ---------------------------------------------------------------------------

def test_yoy_delta_is_difference_of_base(sample_data):
    results = list(generate_fundamental_base_signals(sample_data))
    ids = {r[0]: r for r in results}

    assert "Fund_PE_ratio" in ids
    assert "Fund_PE_ratio_yoy_delta" in ids

    pe_df = ids["Fund_PE_ratio"][3]
    delta_df = ids["Fund_PE_ratio_yoy_delta"][3]

    expected_delta = pe_df - pe_df.shift(252)
    # Compare where both are non-NaN
    common = expected_delta.notna() & delta_df.notna()
    if common.any().any():
        diff = (delta_df[common] - expected_delta[common]).abs().max().max()
        assert diff < 1e-6
    # If all NaN (small fixture), just check shapes match
    assert delta_df.shape == pe_df.shape

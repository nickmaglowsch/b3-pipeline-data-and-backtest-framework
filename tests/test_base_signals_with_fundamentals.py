"""
TDD tests for Task 04: generate_all_base_signals() includes fundamental signals.
Tests are written BEFORE implementation (RED phase).
"""

import pandas as pd
import numpy as np
import pytest

from research.discovery.base_signals import generate_all_base_signals


def _make_df(n_dates=30, n_tickers=2, value=1.0):
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    tickers = ["PETR3", "VALE3"][:n_tickers]
    return pd.DataFrame(
        {t: [value] * n_dates for t in tickers},
        index=dates,
    )


def _make_series(n_dates=30, value=1.0):
    dates = pd.date_range("2023-01-02", periods=n_dates, freq="B")
    return pd.Series([value] * n_dates, index=dates)


def _make_full_data(include_fundamentals=True):
    data = {
        "adj_close": _make_df(value=30.0),
        "split_adj_high": _make_df(value=31.0),
        "split_adj_low": _make_df(value=29.0),
        "split_adj_close": _make_df(value=30.0),
        "fin_vol": _make_df(value=1_000_000.0),
        "ibov": _make_series(value=100.0),
        "cdi_daily": _make_series(value=0.0001),
    }
    if include_fundamentals:
        dates = pd.date_range("2023-01-02", periods=30, freq="B")
        f_shares = pd.DataFrame({"PETR3": [13_000_000.0] * 30, "VALE3": [4_000_000.0] * 30}, index=dates)
        f_net_income = pd.DataFrame({"PETR3": [1_000_000.0] * 30, "VALE3": [2_000_000.0] * 30}, index=dates)
        f_equity = pd.DataFrame({"PETR3": [5_000_000.0] * 30, "VALE3": [8_000_000.0] * 30}, index=dates)
        f_total_assets = pd.DataFrame({"PETR3": [20_000_000.0] * 30, "VALE3": [30_000_000.0] * 30}, index=dates)
        f_ebitda = pd.DataFrame({"PETR3": [2_000_000.0] * 30, "VALE3": [3_000_000.0] * 30}, index=dates)
        f_net_debt = pd.DataFrame({"PETR3": [1_000_000.0] * 30, "VALE3": [-500_000.0] * 30}, index=dates)
        f_revenue = pd.DataFrame({"PETR3": [8_000_000.0] * 30, "VALE3": [12_000_000.0] * 30}, index=dates)
        data.update({
            "f_shares": f_shares,
            "f_net_income": f_net_income,
            "f_equity": f_equity,
            "f_total_assets": f_total_assets,
            "f_ebitda": f_ebitda,
            "f_net_debt": f_net_debt,
            "f_revenue": f_revenue,
        })
    return data


# ---------------------------------------------------------------------------
# Test 1: Fund_PE_ratio and Fund_ROE present when fundamentals provided
# ---------------------------------------------------------------------------

def test_generate_all_includes_fundamental_signals():
    data = _make_full_data(include_fundamentals=True)
    feature_ids = [fid for fid, *_ in generate_all_base_signals(data)]
    assert "Fund_PE_ratio" in feature_ids
    assert "Fund_ROE" in feature_ids


# ---------------------------------------------------------------------------
# Test 2: exactly 18 fundamental signals yielded
# ---------------------------------------------------------------------------

def test_generate_all_yields_18_fundamental_signals():
    data = _make_full_data(include_fundamentals=True)
    feature_ids = [fid for fid, *_ in generate_all_base_signals(data)]
    fund_signals = [fid for fid in feature_ids if fid.startswith("Fund_")]
    assert len(fund_signals) == 18, f"Expected 18 Fund_ signals, got {len(fund_signals)}: {fund_signals}"


# ---------------------------------------------------------------------------
# Test 3: no Fund_ signals when fundamentals absent
# ---------------------------------------------------------------------------

def test_generate_all_no_fundamental_signals_when_absent():
    data = _make_full_data(include_fundamentals=False)
    feature_ids = [fid for fid, *_ in generate_all_base_signals(data)]
    fund_signals = [fid for fid in feature_ids if fid.startswith("Fund_")]
    assert len(fund_signals) == 0


# ---------------------------------------------------------------------------
# Test 4: no Fund_ signals when f_net_income is empty DataFrame
# ---------------------------------------------------------------------------

def test_generate_all_no_fundamental_signals_when_empty_df():
    data = _make_full_data(include_fundamentals=False)
    data["f_net_income"] = pd.DataFrame()
    feature_ids = [fid for fid, *_ in generate_all_base_signals(data)]
    fund_signals = [fid for fid in feature_ids if fid.startswith("Fund_")]
    assert len(fund_signals) == 0


# ---------------------------------------------------------------------------
# Test 5: Fund_PE_ratio tuple structure
# ---------------------------------------------------------------------------

def test_generate_all_tuple_structure_for_fundamental():
    data = _make_full_data(include_fundamentals=True)
    results = {fid: (fid, cat, params, df)
               for fid, cat, params, df in generate_all_base_signals(data)}

    assert "Fund_PE_ratio" in results
    fid, category, params, wide_df = results["Fund_PE_ratio"]
    assert fid == "Fund_PE_ratio"
    assert category == "valuation"
    assert isinstance(params, dict)
    assert isinstance(wide_df, pd.DataFrame)

"""
TDD tests for Task 01: _load_fundamentals() and load_all_data() fundamentals integration.
Tests are written BEFORE implementation — they should FAIL until research/data_loader.py is updated.
"""

import pickle
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_monthly_df(tickers, dates, value=1.0):
    return pd.DataFrame(
        {t: [value] * len(dates) for t in tickers},
        index=pd.to_datetime(dates),
    )


def _make_synthetic_fundamentals(dates=None):
    if dates is None:
        dates = ["2023-01-31", "2023-02-28"]
    tickers = ["PETR3", "VALE3"]
    val = 1_000_000.0
    return {
        "revenue": _make_monthly_df(tickers, dates, val),
        "net_income": _make_monthly_df(tickers, dates, val),
        "ebitda": _make_monthly_df(tickers, dates, val),
        "total_assets": _make_monthly_df(tickers, dates, val),
        "equity": _make_monthly_df(tickers, dates, val),
        "net_debt": _make_monthly_df(tickers, dates, val),
        "shares_outstanding": _make_monthly_df(tickers, dates, val),
    }


def _make_daily_index():
    return pd.date_range("2023-01-01", "2023-03-31", freq="B")


def _make_synthetic_hlc():
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="B")
    tickers = ["PETR3", "VALE3"]
    adj = pd.DataFrame({t: [30.0] * len(dates) for t in tickers}, index=dates)
    return adj, adj.copy(), adj.copy(), adj.copy(), adj.copy(), adj.copy()


# ---------------------------------------------------------------------------
# Test 1: _load_fundamentals returns 7 keys
# ---------------------------------------------------------------------------

def test_load_fundamentals_returns_seven_keys(tmp_path, monkeypatch):
    from research.data_loader import _load_fundamentals
    import research.data_loader as dl
    monkeypatch.setattr(dl.config, "OUTPUT_DIR", tmp_path)

    daily_index = _make_daily_index()
    synth = _make_synthetic_fundamentals()

    with patch("research.data_loader.load_all_fundamentals", return_value=synth), \
         patch("research.data_loader._is_cache_fresh", return_value=False):
        result = _load_fundamentals(daily_index)

    expected_keys = {
        "f_revenue", "f_net_income", "f_ebitda",
        "f_total_assets", "f_equity", "f_net_debt", "f_shares",
    }
    assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 2: forward-fill from monthly to daily
# ---------------------------------------------------------------------------

def test_load_fundamentals_daily_ffill(tmp_path, monkeypatch):
    from research.data_loader import _load_fundamentals
    import research.data_loader as dl
    monkeypatch.setattr(dl.config, "OUTPUT_DIR", tmp_path)

    # Monthly snapshot at Jan 31
    monthly_idx = pd.to_datetime(["2023-01-31"])
    synth = {
        "revenue": pd.DataFrame({"PETR3": [100.0]}, index=monthly_idx),
        "net_income": pd.DataFrame({"PETR3": [100.0]}, index=monthly_idx),
        "ebitda": pd.DataFrame({"PETR3": [100.0]}, index=monthly_idx),
        "total_assets": pd.DataFrame({"PETR3": [100.0]}, index=monthly_idx),
        "equity": pd.DataFrame({"PETR3": [100.0]}, index=monthly_idx),
        "net_debt": pd.DataFrame({"PETR3": [100.0]}, index=monthly_idx),
        "shares_outstanding": pd.DataFrame({"PETR3": [100.0]}, index=monthly_idx),
    }

    daily_index = pd.date_range("2023-01-01", "2023-02-28", freq="B")

    with patch("research.data_loader.load_all_fundamentals", return_value=synth), \
         patch("research.data_loader._is_cache_fresh", return_value=False):
        result = _load_fundamentals(daily_index)

    # Mid-February should carry the Jan 31 value via ffill
    val = result["f_revenue"].loc["2023-02-15", "PETR3"]
    assert abs(val - 100.0) < 1e-4, f"Expected 100.0 but got {val}"


# ---------------------------------------------------------------------------
# Test 3: empty DataFrames on missing table
# ---------------------------------------------------------------------------

def test_load_fundamentals_empty_on_missing_table(tmp_path, monkeypatch):
    from research.data_loader import _load_fundamentals
    import research.data_loader as dl
    monkeypatch.setattr(dl.config, "OUTPUT_DIR", tmp_path)

    daily_index = _make_daily_index()

    with patch(
        "research.data_loader.load_all_fundamentals",
        side_effect=Exception("no such table: fundamentals_pit"),
    ), patch("research.data_loader._is_cache_fresh", return_value=False):
        result = _load_fundamentals(daily_index)

    expected_keys = {
        "f_revenue", "f_net_income", "f_ebitda",
        "f_total_assets", "f_equity", "f_net_debt", "f_shares",
    }
    assert set(result.keys()) == expected_keys
    for k, v in result.items():
        assert isinstance(v, pd.DataFrame)
        assert v.empty, f"Expected empty DataFrame for {k} but got shape {v.shape}"


# ---------------------------------------------------------------------------
# Test 4: partial dict — missing net_debt is still in result as empty df
# ---------------------------------------------------------------------------

def test_load_fundamentals_partial_failure(tmp_path, monkeypatch):
    from research.data_loader import _load_fundamentals
    import research.data_loader as dl
    monkeypatch.setattr(dl.config, "OUTPUT_DIR", tmp_path)

    daily_index = _make_daily_index()
    synth = _make_synthetic_fundamentals()
    del synth["net_debt"]  # simulate partial load

    with patch("research.data_loader.load_all_fundamentals", return_value=synth), \
         patch("research.data_loader._is_cache_fresh", return_value=False):
        result = _load_fundamentals(daily_index)

    assert "f_net_debt" in result
    assert isinstance(result["f_net_debt"], pd.DataFrame)
    assert result["f_net_debt"].empty


# ---------------------------------------------------------------------------
# Test 5: integration — load_all_data returns f_* keys
# ---------------------------------------------------------------------------

def test_load_all_data_includes_fundamentals_keys(tmp_path, monkeypatch):
    from research.data_loader import load_all_data
    import research.data_loader as dl
    monkeypatch.setattr(dl.config, "OUTPUT_DIR", tmp_path)

    synth = _make_synthetic_fundamentals()
    adj, h, l, c, cpx, fvol = _make_synthetic_hlc()

    with patch("research.data_loader.load_b3_hlc_data", return_value=(adj, h, l, c, cpx, fvol)), \
         patch("research.data_loader.load_all_fundamentals", return_value=synth), \
         patch("research.data_loader._load_ibov", return_value=pd.Series(dtype=float)), \
         patch("research.data_loader._load_cdi", return_value=pd.Series(dtype=float)), \
         patch("research.data_loader._is_cache_fresh", return_value=False):
        result = load_all_data()

    assert "f_revenue" in result
    assert "f_net_income" in result
    assert "f_shares" in result

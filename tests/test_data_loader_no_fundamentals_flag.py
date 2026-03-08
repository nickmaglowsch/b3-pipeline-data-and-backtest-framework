"""
TDD tests for Task 05: --no-fundamentals CLI flag and use_fundamentals parameter.
Tests are written BEFORE implementation (RED phase).
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch


def _make_synthetic_hlc():
    dates = pd.date_range("2023-01-02", periods=30, freq="B")
    tickers = ["PETR3", "VALE3"]
    adj = pd.DataFrame({t: [30.0] * 30 for t in tickers}, index=dates)
    return adj, adj.copy(), adj.copy(), adj.copy(), adj.copy(), adj.copy()


def _make_synthetic_fundamentals():
    dates = ["2023-01-31", "2023-02-28"]
    tickers = ["PETR3", "VALE3"]
    val = 1_000_000.0
    return {
        m: pd.DataFrame({t: [val] * 2 for t in tickers}, index=pd.to_datetime(dates))
        for m in ["revenue", "net_income", "ebitda", "total_assets",
                  "equity", "net_debt", "shares_outstanding"]
    }


# ---------------------------------------------------------------------------
# Test 1: load_all_data(use_fundamentals=False) returns empty f_* keys
# ---------------------------------------------------------------------------

def test_load_all_data_no_fundamentals_returns_empty_f_keys(tmp_path, monkeypatch):
    from research.data_loader import load_all_data
    import research.data_loader as dl
    monkeypatch.setattr(dl.config, "OUTPUT_DIR", tmp_path)

    adj, h, l, c, cpx, fvol = _make_synthetic_hlc()

    with patch("research.data_loader.load_b3_hlc_data", return_value=(adj, h, l, c, cpx, fvol)), \
         patch("research.data_loader._load_ibov", return_value=pd.Series(dtype=float)), \
         patch("research.data_loader._load_cdi", return_value=pd.Series(dtype=float)), \
         patch("research.data_loader._is_cache_fresh", return_value=False):
        result = load_all_data(use_fundamentals=False)

    expected_f_keys = [
        "f_revenue", "f_net_income", "f_ebitda",
        "f_total_assets", "f_equity", "f_net_debt", "f_shares",
    ]
    for k in expected_f_keys:
        assert k in result, f"Missing key {k}"
        assert isinstance(result[k], pd.DataFrame)
        assert result[k].empty, f"Expected empty DataFrame for {k}"


# ---------------------------------------------------------------------------
# Test 2: use_fundamentals=False does NOT call load_all_fundamentals
# ---------------------------------------------------------------------------

def test_load_all_data_no_fundamentals_does_not_call_load_all_fundamentals(tmp_path, monkeypatch):
    from research.data_loader import load_all_data
    import research.data_loader as dl
    monkeypatch.setattr(dl.config, "OUTPUT_DIR", tmp_path)

    adj, h, l, c, cpx, fvol = _make_synthetic_hlc()

    with patch("research.data_loader.load_b3_hlc_data", return_value=(adj, h, l, c, cpx, fvol)), \
         patch("research.data_loader._load_ibov", return_value=pd.Series(dtype=float)), \
         patch("research.data_loader._load_cdi", return_value=pd.Series(dtype=float)), \
         patch("research.data_loader.load_all_fundamentals") as mock_fund, \
         patch("research.data_loader._is_cache_fresh", return_value=False):
        load_all_data(use_fundamentals=False)

    mock_fund.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: default (use_fundamentals=True) DOES call load_all_fundamentals
# ---------------------------------------------------------------------------

def test_load_all_data_default_calls_load_all_fundamentals(tmp_path, monkeypatch):
    from research.data_loader import load_all_data
    import research.data_loader as dl
    monkeypatch.setattr(dl.config, "OUTPUT_DIR", tmp_path)

    adj, h, l, c, cpx, fvol = _make_synthetic_hlc()
    synth = _make_synthetic_fundamentals()

    with patch("research.data_loader.load_b3_hlc_data", return_value=(adj, h, l, c, cpx, fvol)), \
         patch("research.data_loader._load_ibov", return_value=pd.Series(dtype=float)), \
         patch("research.data_loader._load_cdi", return_value=pd.Series(dtype=float)), \
         patch("research.data_loader.load_all_fundamentals", return_value=synth) as mock_fund, \
         patch("research.data_loader._is_cache_fresh", return_value=False):
        load_all_data()

    mock_fund.assert_called_once()


# ---------------------------------------------------------------------------
# Test 4: FUNDAMENTAL_SIGNALS_ENABLED is True in config
# ---------------------------------------------------------------------------

def test_config_fundamental_signals_enabled_is_true():
    from research.discovery.config import FUNDAMENTAL_SIGNALS_ENABLED
    assert FUNDAMENTAL_SIGNALS_ENABLED is True

"""
TDD tests for Task 02: compute_data_hash() fundamentals fingerprint.
Tests are written BEFORE implementation (RED phase).
"""

import pandas as pd
import numpy as np
import pytest

from research.discovery.store import compute_data_hash


def _make_adj_close(n_tickers=3, n_dates=100):
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    return pd.DataFrame(np.random.rand(n_dates, n_tickers), index=dates, columns=tickers)


def _make_f_net_income(n_values=100):
    dates = pd.date_range("2020-01-01", periods=n_values, freq="B")
    return pd.DataFrame({"T0": np.ones(n_values) * 1000.0}, index=dates)


# ---------------------------------------------------------------------------
# Test 1: hash changes when fundamentals change
# ---------------------------------------------------------------------------

def test_hash_changes_when_fundamentals_change():
    adj = _make_adj_close()

    # 100 non-NaN values
    ni1 = _make_f_net_income(100)
    # 200 non-NaN values
    ni2 = _make_f_net_income(200)

    data1 = {"adj_close": adj, "f_net_income": ni1}
    data2 = {"adj_close": adj, "f_net_income": ni2}

    h1 = compute_data_hash(data1)
    h2 = compute_data_hash(data2)
    assert h1 != h2, "Hashes should differ when f_net_income has different non-NaN counts"


# ---------------------------------------------------------------------------
# Test 2: hash is stable when fundamentals unchanged
# ---------------------------------------------------------------------------

def test_hash_stable_when_fundamentals_unchanged():
    adj = _make_adj_close()
    ni = _make_f_net_income(100)
    data = {"adj_close": adj, "f_net_income": ni}

    h1 = compute_data_hash(data)
    h2 = compute_data_hash(data)
    assert h1 == h2


# ---------------------------------------------------------------------------
# Test 3: hash works without fundamentals key
# ---------------------------------------------------------------------------

def test_hash_works_without_fundamentals_key():
    adj = _make_adj_close()
    data = {"adj_close": adj}  # no f_net_income key

    result = compute_data_hash(data)
    assert isinstance(result, str)
    assert len(result) == 16


# ---------------------------------------------------------------------------
# Test 4: hash works with empty fundamentals DataFrame
# ---------------------------------------------------------------------------

def test_hash_works_with_empty_fundamentals_df():
    adj = _make_adj_close()

    data_no_key = {"adj_close": adj}
    data_empty = {"adj_close": adj, "f_net_income": pd.DataFrame()}

    h_no_key = compute_data_hash(data_no_key)
    h_empty = compute_data_hash(data_empty)

    # Both should treat fund_count as 0 → same hash
    assert h_no_key == h_empty
    assert isinstance(h_empty, str)
    assert len(h_empty) == 16


# ---------------------------------------------------------------------------
# Test 5: hash changes when adj_close shape changes
# ---------------------------------------------------------------------------

def test_hash_changes_when_adj_close_shape_changes():
    adj1 = _make_adj_close(n_tickers=3)
    adj2 = _make_adj_close(n_tickers=5)

    data1 = {"adj_close": adj1}
    data2 = {"adj_close": adj2}

    assert compute_data_hash(data1) != compute_data_hash(data2)

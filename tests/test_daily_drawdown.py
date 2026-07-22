"""Daily NAV reconstruction must catch intra-rebalance drawdowns the
rebalance-cadence equity curve is blind to (backtests/core/metrics.py)."""
from __future__ import annotations

import pandas as pd

from backtests.core.metrics import reconstruct_daily_values, max_dd, value_to_ret


def test_daily_reconstruction_catches_intra_period_drawdown():
    # Asset crashes -40% mid-window then fully recovers by the rebalance date.
    days = pd.date_range("2020-01-01", periods=9, freq="D")
    px = pd.Series([100, 90, 80, 60, 60, 70, 85, 95, 100], index=days, dtype=float)
    daily_ret = px.pct_change().to_frame("A")

    # Rebalance only at the two endpoints -> cadence curve sees start==end.
    reb = [days[0], days[-1]]
    tw = pd.DataFrame({"A": [1.0, 1.0]}, index=reb)
    cadence_ret = pd.Series([0.0, px.iloc[-1] / px.iloc[0] - 1.0], index=reb)
    assert abs(max_dd(cadence_ret)) < 1e-9          # rebalance-cadence: hides it

    daily_vals = reconstruct_daily_values(tw, daily_ret, initial_capital=100.0)
    assert max_dd(value_to_ret(daily_vals)) < -0.35  # daily path: -40% trough visible


def test_uncovered_sleeve_bails_to_empty():
    # A blend sleeve (an ETF) with no daily series must NOT be silently zeroed —
    # reconstruction bails so the caller falls back to cadence DD.
    days = pd.date_range("2020-01-01", periods=4, freq="D")
    daily_ret = pd.DataFrame({"AAAA": [None, 0.0, 0.0, 0.0]}, index=days)
    tw = pd.DataFrame({"AAAA": [0.5], "IVVB11": [0.5]}, index=[days[0]])  # IVVB11 uncovered
    assert reconstruct_daily_values(tw, daily_ret, 100.0).empty


def test_two_asset_drift_matches_buy_and_hold():
    # Two equal-weighted assets, one held flat, one +10% total over the segment.
    days = pd.date_range("2020-01-01", periods=3, freq="D")
    daily_ret = pd.DataFrame(
        {"A": [None, 0.0, 0.0], "B": [None, 0.05, 0.047619]}, index=days
    )
    tw = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[days[0]])
    nav = reconstruct_daily_values(tw, daily_ret, initial_capital=100.0)
    # 0.5*100 flat + 0.5*100*1.10 = 105 at the end.
    assert nav.iloc[-1] == __import__("pytest").approx(105.0, abs=0.05)

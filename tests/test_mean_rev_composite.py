"""End-to-end behaviour tests for MeanRevComposite.

Ported from the old ``backtests/validate_mean_rev_composite.py`` runner script.
That script needed the 600MB SQLite DB; here the same assertions run against a
synthetic ``shared_data`` built from random daily prices through the real
``compute_mean_rev_features`` — no DB, no network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests.core.mean_rev_helpers import compute_mean_rev_features
from backtests.core.simulation import run_simulation
from backtests.core.strategy_registry import get_registry

FREQ = "ME"


def _synthetic_shared(n_days: int = 1500, n_tickers: int = 30, seed: int = 11) -> dict:
    """A shared_data dict with every frame MeanRevComposite reads, mirroring the
    derivations in shared_data.build_shared_data (daily panel -> monthly)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2014-01-01", periods=n_days)
    cols = [f"T{i:02d}" for i in range(n_tickers)]

    daily_ret = rng.normal(0.0004, 0.015, (n_days, n_tickers))
    adj_close = pd.DataFrame(
        100.0 * np.exp(np.cumsum(daily_ret, axis=0)), index=days, columns=cols)
    high, low = adj_close * 1.012, adj_close * 0.988
    # CDI wanders so the "CDI tightening" regime gate flips both ways
    cdi_daily = pd.Series(
        np.abs(rng.normal(0.0004, 0.0002, n_days)).cumsum() % 0.0008, index=days)
    ibov_px = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n_days))), index=days)
    ibov_daily_ret = ibov_px.pct_change()

    px = adj_close.resample(FREQ).last()
    ret = px.pct_change()
    ibov_vol_m = ibov_daily_ret.rolling(20).std().resample(FREQ).last()

    return {
        "adj_close": adj_close,
        "px": px,
        "ret": ret,
        "raw_close": adj_close.resample(FREQ).last(),
        "adtv": pd.DataFrame(
            rng.uniform(2e6, 9e6, (len(ret), n_tickers)), index=ret.index, columns=cols),
        "has_glitch": ((ret > 1.0) | (ret < -0.45)).rolling(12).max(),
        "cdi_monthly": (1 + cdi_daily).resample(FREQ).prod() - 1,
        "cdi_daily": cdi_daily,
        "ibov_ret": ibov_px.resample(FREQ).last().pct_change(),
        "ibov_px": ibov_px,
        "ibov_vol_pctrank": ibov_vol_m.expanding(min_periods=12).apply(
            lambda x: (x.iloc[-1] >= x).mean(), raw=False),
        "atr_20d_daily": adj_close.pct_change().abs().ewm(span=14, min_periods=14).mean(),
        **compute_mean_rev_features(
            adj_close, high, low, adj_close, cdi_daily, ibov_daily_ret, ibov_px),
    }


@pytest.fixture(scope="module")
def signals():
    strategy = get_registry().get("MeanRevComposite")
    shared = _synthetic_shared()
    ret, tw = strategy.generate_signals(shared, strategy.get_default_parameters())
    return shared, ret, tw


def test_both_mean_reversion_strategies_are_registered():
    names = get_registry().names()
    assert "MeanRevComposite" in names
    assert "SimpleMeanReversion" in names          # backward compat


def test_signals_allocate_between_equities_and_cdi(signals):
    _, ret, tw = signals
    assert ret.shape[0] == tw.shape[0]
    assert "CDI_ASSET" in tw.columns
    assert tw.abs().sum().sum() > 0, "target weights are all zero"
    # the regime filter must produce both risk-off (CDI) and risk-on (equity) months
    assert (tw["CDI_ASSET"] > 0).sum() > 0, "no risk-off months — regime filter dead"
    equity = tw.drop(columns=["CDI_ASSET"]).abs().sum(axis=1)
    assert (equity > 0).sum() > 0, "no risk-on months — alpha scoring dead"


def test_simulation_keeps_equity_positive(signals):
    _, ret, tw = signals
    result = run_simulation(
        ret.fillna(0.0), tw,
        initial_capital=100_000, tax_rate=0.15, slippage=0.001,
        name="MeanRevComposite", monthly_sales_exemption=20_000,
    )
    assert result["pretax_values"].iloc[-1] > 0
    assert result["aftertax_values"].iloc[-1] > 0


def test_long_short_mode_produces_negative_weights():
    strategy = get_registry().get("MeanRevComposite")
    params = strategy.get_default_parameters()
    params["enable_short"] = "Yes"
    params["short_gross"] = 0.20
    _, tw = strategy.generate_signals(_synthetic_shared(), params)
    assert (tw < 0).any(axis=1).sum() > 0, "no shorts in long-short mode"

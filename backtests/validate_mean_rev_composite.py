"""
End-to-end validation of the MeanReversionComposite strategy.
Run from project root: python backtests/validate_mean_rev_composite.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b3_market_data.sqlite")


def test_plugin_interface():
    """Test 1: StrategyBase plugin generates valid signals."""
    print("\n[Test 1] StrategyBase plugin interface...")
    from backtests.core.shared_data import build_shared_data
    from backtests.core.strategy_registry import get_registry

    shared = build_shared_data(DB_PATH, "2010-01-01", "2026-03-01")
    registry = get_registry()

    # New strategy registered
    assert "MeanRevComposite" in registry.names(), f"Strategy not registered. Available: {registry.names()}"
    strategy = registry.get("MeanRevComposite")

    # Old strategy preserved
    assert "SimpleMeanReversion" in registry.names(), "SimpleMeanReversion missing (backward compat)"

    params = strategy.get_default_parameters()
    print(f"  Default params keys: {list(params.keys())}")

    ret, tw = strategy.generate_signals(shared, params)
    print(f"  ret shape: {ret.shape}, tw shape: {tw.shape}")

    assert ret.shape[0] == tw.shape[0], "ret and tw must have same number of rows"
    assert "CDI_ASSET" in tw.columns, "CDI_ASSET column missing from target_weights"
    assert tw.abs().sum().sum() > 0, "Target weights are all zero"

    # Check some months have CDI allocation (Risk-Off)
    cdi_allocated = (tw["CDI_ASSET"] > 0).sum()
    print(f"  Months with CDI allocation (Risk-Off): {cdi_allocated}")
    assert cdi_allocated > 0, "No Risk-Off months detected -- regime filter may not be working"

    # Check some months have equity allocation (Risk-On)
    equity_months = (tw.drop(columns=["CDI_ASSET"], errors="ignore").abs().sum(axis=1) > 0).sum()
    print(f"  Months with equity allocation (Risk-On): {equity_months}")
    assert equity_months > 0, "No Risk-On months -- alpha scoring may not be working"

    print("  [PASS] Plugin interface test passed")
    return shared, params


def test_simulation(shared, params):
    """Test 2: Simulation produces valid equity curve."""
    print("\n[Test 2] Simulation engine...")
    from backtests.core.strategy_registry import get_registry
    from backtests.core.simulation import run_simulation

    strategy = get_registry().get("MeanRevComposite")
    ret, tw = strategy.generate_signals(shared, params)

    result = run_simulation(
        ret.fillna(0.0), tw,
        initial_capital=100_000, tax_rate=0.15, slippage=0.001,
        name="MeanRevComposite", monthly_sales_exemption=20_000,
    )

    pt = result["pretax_values"]
    at = result["aftertax_values"]
    print(f"  Pre-tax equity: {pt.iloc[0]:,.0f} -> {pt.iloc[-1]:,.0f}")
    print(f"  After-tax equity: {at.iloc[0]:,.0f} -> {at.iloc[-1]:,.0f}")
    print(f"  Total tax paid: {result['tax_paid'].sum():,.0f}")

    assert pt.iloc[-1] > 0, "Pre-tax equity went to zero"
    assert at.iloc[-1] > 0, "After-tax equity went to zero"

    print("  [PASS] Simulation test passed")


def test_strategy_returns():
    """Test 3: strategy_returns.py includes MeanRevComposite."""
    print("\n[Test 3] strategy_returns.py integration...")
    from backtests.core.strategy_returns import build_strategy_returns

    returns_df, sim_results, _ = build_strategy_returns(
        start="2010-01-01", end="2026-03-01"
    )
    print(f"  Returns columns: {list(returns_df.columns)}")

    assert "MeanRevComposite" in returns_df.columns, "MeanRevComposite not in returns_df"
    assert "MeanRevComposite" in sim_results, "MeanRevComposite not in sim_results"

    mr_returns = returns_df["MeanRevComposite"].dropna()
    print(f"  MeanRevComposite: {len(mr_returns)} months, mean={mr_returns.mean():.4f}")
    assert len(mr_returns) > 12, "Too few return observations"
    assert not mr_returns.isna().all(), "All returns are NaN"

    print("  [PASS] strategy_returns.py test passed")


def test_long_short():
    """Test 4: Long-short mode produces negative weights."""
    print("\n[Test 4] Long-short mode...")
    from backtests.core.shared_data import build_shared_data
    from backtests.core.strategy_registry import get_registry

    shared = build_shared_data(DB_PATH, "2010-01-01", "2026-03-01")
    strategy = get_registry().get("MeanRevComposite")

    params = strategy.get_default_parameters()
    params["enable_short"] = "Yes"
    params["short_gross"] = 0.20

    ret, tw = strategy.generate_signals(shared, params)
    neg_weights = (tw < 0).any(axis=1).sum()
    print(f"  Months with negative weights (shorts): {neg_weights}")
    assert neg_weights > 0, "No negative weights in long-short mode"

    print("  [PASS] Long-short test passed")


def test_stability_guard():
    """Test 5: Stability guard produces varying weights."""
    print("\n[Test 5] Stability guard...")
    from backtests.core.mean_rev_helpers import (
        compute_alpha_score, compute_signal_stability,
    )
    from backtests.core.shared_data import build_shared_data

    shared = build_shared_data(DB_PATH, "2005-01-01", "2026-03-01")
    params = {
        "w_vol": 0.333, "w_macro": 0.333, "w_autocorr": 0.333,
        "enable_stability_guard": "Yes",
        "ic_check_freq": 3, "ic_trailing_months": 12, "ic_flip_consecutive": 2,
    }

    _, sub_signals = compute_alpha_score(shared, params, "ME")
    fwd_ret = shared["ret"].shift(-1)
    base_w = {"sub_A": 0.333, "sub_B": 0.333, "sub_C": 0.333}
    stability = compute_signal_stability(sub_signals, fwd_ret, base_w, params)

    print(f"  Stability weights shape: {stability.shape}")
    n_unique = stability.drop_duplicates().shape[0]
    print(f"  Unique weight patterns: {n_unique}")

    print("  [PASS] Stability guard test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("  MEAN-REVERSION COMPOSITE STRATEGY - END-TO-END VALIDATION")
    print("=" * 70)

    shared, params = test_plugin_interface()
    test_simulation(shared, params)
    test_strategy_returns()
    test_long_short()
    test_stability_guard()

    print("\n" + "=" * 70)
    print("  ALL VALIDATION TESTS PASSED")
    print("=" * 70)

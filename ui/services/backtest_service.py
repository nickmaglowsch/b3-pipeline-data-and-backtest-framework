"""
Backtest Service
================
Executes a single backtest from the strategy registry.
Designed to be called inside a JobRunner thread.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"


@st.cache_data(ttl=3600)
def _get_shared_data_cached(db_path: str, start: str, end: str, freq: str) -> dict:
    """
    Cache the shared data dict in Streamlit's data cache.
    Uses cache_data (not cache_resource) to ensure callers get independent copies.
    Key: (db_path, start, end, freq).
    """
    from backtests.core.shared_data import build_shared_data
    return build_shared_data(db_path, start, end, freq)


def run_backtest(strategy_name: str, params: dict) -> dict:
    """
    Execute a single backtest.

    Args:
        strategy_name: Name of the strategy in the registry.
        params:        Dict of parameter values.

    Returns:
        Result dict with keys:
          pretax_values, aftertax_values, ibov_ret, cdi_ret,
          tax_paid, loss_carryforward, turnover, metrics,
          params, strategy_name.
    """
    sys.path.insert(0, str(PROJECT_ROOT))

    from backtests.core.strategy_registry import get_registry
    from backtests.core.simulation import run_simulation
    from backtests.core.metrics import build_metrics, value_to_ret

    registry = get_registry()
    strategy = registry.get(strategy_name)

    start = params.get("start_date", "2005-01-01")
    end = params.get("end_date", datetime.today().strftime("%Y-%m-%d"))
    freq = params.get("rebalance_freq", "ME")

    if end == "today":
        end = datetime.today().strftime("%Y-%m-%d")

    print(f"[backtest_service] Starting: {strategy_name}")
    print(f"[backtest_service] Params: start={start}, end={end}, freq={freq}")

    # Load shared data (cached in Streamlit resource cache)
    try:
        shared = _get_shared_data_cached(str(DB_PATH), start, end, freq)
    except Exception as e:
        print(f"[backtest_service] Cache miss or error, rebuilding: {e}")
        from backtests.core.shared_data import build_shared_data
        shared = build_shared_data(str(DB_PATH), start, end, freq)

    print(f"[backtest_service] Generating signals for {strategy_name}...")
    ret_matrix, target_weights = strategy.generate_signals(shared, params)

    print("[backtest_service] Running simulation...")
    result = run_simulation(
        returns_matrix=ret_matrix.fillna(0.0),
        target_weights=target_weights,
        initial_capital=float(params.get("initial_capital", 100_000)),
        tax_rate=float(params.get("tax_rate", 0.15)),
        slippage=float(params.get("slippage", 0.001)),
        monthly_sales_exemption=float(params.get("monthly_sales_exemption", 20_000)),
        name=strategy_name,
    )

    # Align with benchmarks
    ibov_ret = shared["ibov_ret"]
    cdi_ret = shared["cdi_monthly"]
    common = result["pretax_values"].index.intersection(ibov_ret.index)

    at_val = result["aftertax_values"].loc[common]
    pt_val = result["pretax_values"].loc[common]
    at_ret = value_to_ret(at_val)
    pt_ret = value_to_ret(pt_val)

    metrics = [
        build_metrics(pt_ret, f"{strategy_name} (Pre-Tax)"),
        build_metrics(at_ret, f"{strategy_name} (After-Tax)"),
        build_metrics(ibov_ret.loc[common], "IBOV"),
        build_metrics(cdi_ret.reindex(common).fillna(0), "CDI"),
    ]

    sharpe_val = metrics[1].get('Sharpe', 'N/A')
    sharpe_str = f"{sharpe_val:.2f}" if isinstance(sharpe_val, (int, float)) else str(sharpe_val)
    print(f"[backtest_service] Done. Sharpe (after-tax): {sharpe_str}")

    def _safe_loc(series: pd.Series, idx) -> pd.Series:
        """Safely index a Series; return empty if index not present."""
        try:
            common_idx = series.index.intersection(idx)
            return series.loc[common_idx]
        except Exception:
            return pd.Series(dtype=float)

    return {
        "pretax_values": pt_val,
        "aftertax_values": at_val,
        "ibov_ret": ibov_ret.loc[common],
        "cdi_ret": cdi_ret.reindex(common).fillna(0),
        "tax_paid": _safe_loc(result["tax_paid"], common),
        "loss_carryforward": _safe_loc(result["loss_carryforward"], common),
        "turnover": _safe_loc(result["turnover"], common),
        "metrics": metrics,
        "params": params,
        "strategy_name": strategy_name,
    }

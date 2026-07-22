"""
Backtest Service
================
Executes a single backtest from the strategy registry.
Designed to be called inside a JobRunner thread.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"


@st.cache_data(ttl=3600)
def _get_shared_data_cached(
    db_path: str,
    db_mtime: float,
    start: str,
    end: str,
    freq: str,
    include_fundamentals: bool = False,
) -> dict:
    """
    Cache the shared data dict in Streamlit's data cache.
    Uses cache_data (not cache_resource) to ensure callers get independent copies.
    Key: (db_path, db_mtime, start, end, freq, include_fundamentals) —
    db_mtime invalidates the cache after a pipeline rebuild touches the DB.
    """
    from backtests.core.shared_data import build_shared_data
    return build_shared_data(db_path, start, end, freq, include_fundamentals=include_fundamentals)


def _twr_curve(values: pd.Series, ret: pd.Series) -> pd.Series:
    """Growth-of-capital curve from time-weighted returns.

    Based on the first NON-ZERO NAV: a pure-DCA run (initial_capital=0) starts
    at 0, and scaling by that would flatten the whole curve to zeros.
    """
    nonzero = values[values != 0]
    base = float(nonzero.iloc[0]) if len(nonzero) else 1.0
    return base * (1 + ret).cumprod()


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
    from backtests.core.metrics import build_metrics, value_to_ret, strategy_daily_values
    from backtests.core.strategy_base import dedup_target_weights

    registry = get_registry()
    strategy = registry.get(strategy_name)

    start = params.get("start_date", "2005-01-01")
    end = params.get("end_date", datetime.today().strftime("%Y-%m-%d"))
    freq = params.get("rebalance_freq", "ME")

    if end == "today":
        end = datetime.today().strftime("%Y-%m-%d")

    # Detect whether strategy needs CVM fundamentals data
    needs_funds = getattr(strategy, "needs_fundamentals", False)

    print(f"[backtest_service] Starting: {strategy_name}")
    print(f"[backtest_service] Params: start={start}, end={end}, freq={freq}")
    print(f"[backtest_service] include_fundamentals={needs_funds}")

    # Load shared data (cached in Streamlit data cache, keyed on DB mtime)
    shared = _get_shared_data_cached(
        str(DB_PATH), os.path.getmtime(DB_PATH), start, end, freq, include_fundamentals=needs_funds
    )

    print(f"[backtest_service] Generating signals for {strategy_name}...")
    ret_matrix, target_weights = strategy.generate_signals(shared, params)
    # Global: one ticker per company (merge dual-class holdings onto most-liquid).
    target_weights = dedup_target_weights(target_weights, shared["adtv"])

    print("[backtest_service] Running simulation...")
    result = run_simulation(
        returns_matrix=ret_matrix.fillna(0.0),
        target_weights=target_weights,
        initial_capital=float(params.get("initial_capital", 100_000)),
        tax_rate=float(params.get("tax_rate", 0.15)),
        slippage=float(params.get("slippage", 0.001)),
        monthly_sales_exemption=float(params.get("monthly_sales_exemption", 20_000)),
        contribution=float(params.get("contribution", 0.0)),
        name=strategy_name,
    )

    # Align with benchmarks
    ibov_ret = shared["ibov_ret"]
    cdi_ret = shared["cdi_monthly"]
    common = result["pretax_values"].index.intersection(ibov_ret.index)

    at_val = result["aftertax_values"].loc[common]
    pt_val = result["pretax_values"].loc[common]
    # Buy-ins are stripped out here: NAV includes deposits, so a raw pct_change
    # would book every aporte as a gain.
    contribs = result["contributions"].loc[common]
    at_ret = value_to_ret(at_val, contribs)
    pt_ret = value_to_ret(pt_val, contribs)

    ppy = {"ME": 12, "QE": 4, "W-FRI": 52}.get(freq, 12)

    # Daily mark-to-market paths so Max Drawdown / Calmar see intra-rebalance
    # lows — the rebalance-cadence equity curve is blind to a mid-period crash
    # (a quarterly curve can report an -8% max DD through a -37% quarter).
    w0, w1 = common[0], common[-1]
    strat_daily = strategy_daily_values(
        shared, target_weights, float(params.get("initial_capital", 100_000))
    ).loc[w0:w1]
    ibov_daily = shared["ibov_px"].loc[w0:w1]                       # prices = NAV path
    cdi_daily_nav = (1 + shared["cdi_daily"]).cumprod().loc[w0:w1]

    metrics = [
        build_metrics(pt_ret, f"{strategy_name} (Pre-Tax)", ppy, daily_values=strat_daily),
        build_metrics(at_ret, f"{strategy_name} (After-Tax)", ppy, daily_values=strat_daily),
        build_metrics(ibov_ret.loc[common], "IBOV", ppy, daily_values=ibov_daily),
        build_metrics(cdi_ret.reindex(common).fillna(0), "CDI", ppy, daily_values=cdi_daily_nav),
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
        # Contribution-neutral (time-weighted) curves: identical to the NAV
        # curves when there are no buy-ins, honest drawdowns when there are
        # (deposits otherwise refill the peak and mute the DD).
        "pretax_twr": _twr_curve(pt_val, pt_ret),
        "aftertax_twr": _twr_curve(at_val, at_ret),
        "contributions": contribs,
        "invested": result["invested"].loc[common],
        "ibov_ret": ibov_ret.loc[common],
        "cdi_ret": cdi_ret.reindex(common).fillna(0),
        "tax_paid": _safe_loc(result["tax_paid"], common),
        "loss_carryforward": _safe_loc(result["loss_carryforward"], common),
        "turnover": _safe_loc(result["turnover"], common),
        "metrics": metrics,
        "params": params,
        "strategy_name": strategy_name,
    }

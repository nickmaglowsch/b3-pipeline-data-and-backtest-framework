"""
Strategy Return Extraction Module
==================================
Runs the 9 core B3 strategies and returns their monthly after-tax return series
as a single clean DataFrame, so portfolio construction can be done over strategy
return streams without re-deriving the signals.

The strategies are the *registered* ones (backtests/strategies/ plugins and
specs/*.yaml), looked up by name — this module owns no signal logic of its own.
It used to carry a private copy of each strategy plus a second copy of
build_shared_data(); those drifted from the registry versions the UI runs, so a
name here is now the single source of truth.

Usage:
    from backtests.core.strategy_returns import build_strategy_returns

    returns_df, sim_results, regime_signals = build_strategy_returns()
    # returns_df: DataFrame[date x strategy], values = after-tax monthly returns
    # sim_results: dict of {strategy_name: simulation_result_dict}
    # regime_signals: dict of {signal_name: pd.Series} for downstream regime use
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd

from backtests.core.metrics import value_to_ret
from backtests.core.shared_data import build_shared_data
from backtests.core.simulation import run_simulation
from backtests.core.strategy_registry import get_registry

_DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "b3_market_data.sqlite",
)

_FREQ = "ME"

# The 9 core streams, in display order. Names must match the registry.
STRATEGY_NAMES = [
    "CDI+MA200",
    "Res.MultiFactor",
    "RegimeSwitching",
    "COPOM Easing",
    "MultiFactor",
    "SmallcapMom",      # NOTE: ADTV >= R$100K, below the R$1M liquid floor
    "LowVol",
    "MomSharpe",
    "MeanRevComposite",
]

# SmallcapMom uses a lower ADTV floor -- flagged for downstream exclusion
SMALLCAP_MOM_NOTE = (
    "SmallcapMom uses ADTV >= R$100K (below the R$1M liquid floor). "
    "Its 'small cap' ceiling is the median ADTV of the names that already pass "
    "that floor -- NOT the median of the whole cross-section, which is what the "
    "private copy this module used to carry did. The registered spec therefore "
    "reaches materially further up the cap scale. "
    "Downstream portfolio optimization should test with and without this strategy."
)

_REGIME_KEYS = ("is_easing", "ibov_calm", "ibov_uptrend", "ibov_above")


def build_strategy_returns(
    db_path: str = None,
    start: str = "2005-01-01",
    end: str = None,
    capital: float = 100_000,
    tax_rate: float = 0.15,
    slippage: float = 0.001,
    monthly_sales_exemption: float = 20_000,
) -> "tuple[pd.DataFrame, dict, dict]":
    """
    Run the 9 core strategies and return their after-tax monthly return series.

    Parameters
    ----------
    db_path : str
        Path to b3_market_data.sqlite. Defaults to project root.
    start, end : str
        Backtest window (ISO 8601). `end` defaults to today.
    capital : float
        Initial capital in BRL.
    tax_rate : float
        Capital gains tax rate (default 0.15 = 15%).
    slippage : float
        Round-trip transaction cost fraction (default 0.001 = 0.1%).
    monthly_sales_exemption : float
        Monthly sales exemption (BRL) under Brazilian CGT rules — months whose
        total sales are at or below this are tax-exempt.

    Returns
    -------
    returns_df : pd.DataFrame
        Columns = [strategy_names..., "IBOV", "CDI"], index = monthly dates,
        values = after-tax simple monthly returns.
    sim_results : dict
        {strategy_name: simulation_result_dict} — full equity curves. A strategy
        that raises is omitted here and left all-NaN in returns_df.
    regime_signals : dict
        {"is_easing", "ibov_calm", "ibov_uptrend", "ibov_above"} -> bool Series.
    """
    if db_path is None:
        db_path = _DEFAULT_DB
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    print("  [strategy_returns] Loading market data...")
    shared = build_shared_data(db_path, start, end, _FREQ)
    registry = get_registry()

    sim_results: dict = {}
    monthly_returns: dict = {}

    for name in STRATEGY_NAMES:
        print(f"  [strategy_returns] Running {name}...")
        try:
            strategy = registry.get(name)
            params = {
                **strategy.get_default_parameters(),
                "start_date": start,
                "end_date": end,
                "rebalance_freq": _FREQ,
            }
            r, tw = strategy.generate_signals(shared, params)
            result = run_simulation(
                returns_matrix=r.fillna(0.0),
                target_weights=tw,
                initial_capital=capital,
                tax_rate=tax_rate,
                slippage=slippage,
                name=name,
                monthly_sales_exemption=monthly_sales_exemption,
            )
            sim_results[name] = result
            monthly_returns[name] = value_to_ret(result["aftertax_values"])
        except Exception as exc:
            print(f"    WARNING: {name} failed -- {exc}")
            import traceback
            traceback.print_exc()
            monthly_returns[name] = pd.Series(dtype=float)

    monthly_returns["IBOV"] = shared["ibov_ret"]
    monthly_returns["CDI"] = shared["cdi_monthly"]

    returns_df = pd.DataFrame(monthly_returns).dropna(how="all")
    regime_signals = {k: shared[k] for k in _REGIME_KEYS}

    print(
        f"  [strategy_returns] Done. Shape={returns_df.shape}, "
        f"Columns={list(returns_df.columns)}"
    )
    return returns_df, sim_results, regime_signals


if __name__ == "__main__":
    import time

    t0 = time.time()
    returns_df, sim_results, regime_signals = build_strategy_returns()
    print(f"\nShape: {returns_df.shape}  ({time.time() - t0:.1f}s)")
    print(f"Index: {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"Sim results: {list(sim_results.keys())}")
    print(f"\nSmallcapMom note:\n  {SMALLCAP_MOM_NOTE}")
    print(f"\n{returns_df.tail(3).to_string()}")

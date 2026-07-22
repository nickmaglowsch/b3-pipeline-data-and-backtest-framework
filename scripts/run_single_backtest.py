"""Headless single-strategy backtest — mirrors ui/services/backtest_service.run_backtest
(incl. the daily-resolution drawdown fix) without the Streamlit deps.

Usage: python scripts/run_single_backtest.py "QDL-Equity" [monthly_buy_in_brl] [initial_capital_brl]
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backtests.core.shared_data import build_shared_data
from backtests.core.strategy_registry import get_registry
from backtests.core.simulation import run_simulation
from backtests.core.metrics import (
    build_metrics, value_to_ret, strategy_daily_values, display_metrics_table,
)
from backtests.core.strategy_base import dedup_target_weights

DB = str(ROOT / "b3_market_data.sqlite")


def main(name: str, contribution: float = 0.0,
         initial_capital: float | None = None) -> None:
    strat = get_registry().get(name)
    params = strat.get_default_parameters()
    if initial_capital is not None:
        params["initial_capital"] = initial_capital

    start = params.get("start_date", "2005-01-01")
    end = params.get("end_date", "today")
    if end == "today":
        end = datetime.today().strftime("%Y-%m-%d")
    freq = params.get("rebalance_freq", "ME")
    needs_funds = getattr(strat, "needs_fundamentals", False)

    print(f"Strategy : {name}")
    print(f"Window   : {start} .. {end}  ({freq})")
    print(f"Key params: min_adtv={params.get('min_adtv')}, top_n={params.get('top_n')}, "
          f"ewma_n={params.get('ewma_n')}, min_div_years={params.get('min_div_years')}")
    print(f"Loading shared data (fundamentals={needs_funds}) ...")
    shared = build_shared_data(DB, start, end, freq, include_fundamentals=needs_funds)

    print("Generating signals + running simulation ...")
    ret_matrix, tw = strat.generate_signals(shared, params)
    tw = dedup_target_weights(tw, shared["adtv"])   # one ticker per company (global)
    result = run_simulation(
        returns_matrix=ret_matrix.fillna(0.0),
        target_weights=tw,
        initial_capital=float(params.get("initial_capital", 100_000)),
        tax_rate=float(params.get("tax_rate", 0.15)),
        slippage=float(params.get("slippage", 0.001)),
        monthly_sales_exemption=float(params.get("monthly_sales_exemption", 20_000)),
        contribution=contribution,
        name=name,
    )

    ibov_ret = shared["ibov_ret"]
    cdi_ret = shared["cdi_monthly"]
    common = result["pretax_values"].index.intersection(ibov_ret.index)
    w0, w1 = common[0], common[-1]

    contribs = result["contributions"].loc[common]      # deposits are not returns
    at_ret = value_to_ret(result["aftertax_values"].loc[common], contribs)
    pt_ret = value_to_ret(result["pretax_values"].loc[common], contribs)

    # Daily NAV paths -> intra-rebalance-aware Max Drawdown / Calmar.
    strat_daily = strategy_daily_values(
        shared, tw, float(params.get("initial_capital", 100_000))
    ).loc[w0:w1]
    ibov_daily = shared["ibov_px"].loc[w0:w1]
    cdi_daily_nav = (1 + shared["cdi_daily"]).cumprod().loc[w0:w1]

    ppy = {"ME": 12, "QE": 4, "W-FRI": 52}.get(freq, 12)
    metrics = [
        build_metrics(pt_ret, f"{name} (Pre-Tax)", ppy, daily_values=strat_daily),
        build_metrics(at_ret, f"{name} (After-Tax)", ppy, daily_values=strat_daily),
        build_metrics(ibov_ret.loc[common], "IBOV", ppy, daily_values=ibov_daily),
        build_metrics(cdi_ret.reindex(common).fillna(0), "CDI", ppy, daily_values=cdi_daily_nav),
    ]

    final_nav = result["aftertax_values"].loc[common].iloc[-1]
    print(f"\nCommon window: {w0.date()} .. {w1.date()}  ({len(common)} periods)")
    print(f"Final after-tax NAV: {final_nav:,.0f} "
          f"(from {params.get('initial_capital', 100_000):,.0f})")
    if contribution:
        invested = result["invested"].loc[common].iloc[-1]
        print(f"Monthly buy-in    : {contribution:,.0f}  "
              f"({contribs.sum():,.0f} deposited over the window)")
        print(f"Total invested    : {invested:,.0f}   "
              f"Profit: {final_nav - invested:,.0f}")
    display_metrics_table(metrics)


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "QDL-Equity",
        float(sys.argv[2]) if len(sys.argv) > 2 else 0.0,
        float(sys.argv[3]) if len(sys.argv) > 3 else None,
    )

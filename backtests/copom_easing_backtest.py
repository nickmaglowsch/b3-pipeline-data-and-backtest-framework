"""
B3 COPOM Easing Strategy Backtest
==================================
Simple binary switch: IBOV during easing cycles, CDI during tightening.
Easing = current month's CDI rate is lower than 3 months ago.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.simulation import run_simulation

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
REBALANCE_FREQ = "ME"
PERIODS_PER_YEAR = 12
TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def main():
    print("\n" + "=" * 70)
    print("  B3 COPOM EASING STRATEGY BACKTEST (15% CGT)")
    print("  Easing → IBOV  |  Tightening → CDI")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()
    cdi_monthly = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    # Easing = current month's CDI rate is lower than 3 months ago
    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    cdi_ret = cdi_monthly.copy()

    # Add IBOV and CDI to returns matrix
    ret["IBOV"] = ibov_ret
    ret["CDI_ASSET"] = cdi_monthly

    # Build target weights
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

    easing_months = 0
    tight_months = 0
    for i in range(5, len(ret)):
        if is_easing.iloc[i]:
            target_weights.iloc[i, target_weights.columns.get_loc("IBOV")] = 1.0
            easing_months += 1
        else:
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            tight_months += 1

    print(f"\n  Running simulation ({REBALANCE_FREQ})...")
    result = run_simulation(
        returns_matrix=ret.fillna(0.0),
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="COPOM Easing",
        monthly_sales_exemption=20_000,
    )

    common = result["pretax_values"].index.intersection(ibov_ret.index)
    pretax_val = result["pretax_values"].loc[common]
    aftertax_val = result["aftertax_values"].loc[common]
    ibov_ret = ibov_ret.loc[common]
    cdi_ret = cdi_ret.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = result["tax_paid"].sum()

    m_pretax = build_metrics(pretax_ret, "COPOM Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret, "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    total_months = easing_months + tight_months
    print(f"\n  Regime Summary:")
    print(f"    Easing (IBOV):      {easing_months} months ({easing_months/max(total_months,1)*100:.0f}%)")
    print(f"    Tightening (CDI):   {tight_months} months ({tight_months/max(total_months,1)*100:.0f}%)")

    plot_tax_backtest(
        title=(
            f"COPOM Easing Strategy (B3 Native)\n"
            f"Easing → 100% IBOV  |  Tightening → 100% CDI\n"
            f"15% CGT + {SLIPPAGE*100:.1f}% Slippage  ·  {START_DATE[:4]}–{END_DATE[:4]}"
        ),
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=total_tax,
        out_path="copom_easing_backtest.png",
        cdi_ret=cdi_ret,
    )


if __name__ == "__main__":
    main()

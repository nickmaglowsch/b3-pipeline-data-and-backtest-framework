"""
B3 Dual Momentum Strategy (Brazil vs US Dollar)
=======================================================================
Strategy: In emerging markets, the currency dictates the return. 
When Brazil enters a bear market, the BRL crashes against the USD. 
When Brazil enters a bull market, the BRL strengthens.

This strategy applies Gary Antonacci's "Dual Momentum" framework purely 
to the macro level. Every month, it calculates the 3-month momentum of 
the Ibovespa (IBOV) and the S&P 500 in BRL (IVVB11).
It allocates 100% of the portfolio to whichever asset has the higher 
relative momentum. If both are negative, it moves 100% to CDI (Risk-free).
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
LOOKBACK_MONTHS = 3  # 3-month momentum

TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"
IVVB11_TICKER = "IVVB11.SA" 

def main():
    print("\n" + "=" * 70)
    print("  B3 DUAL MOMENTUM BACKTEST (IBOV vs IVVB11 vs CDI)")
    print("=" * 70)

    # We don't actually need stock data, but the framework requires a returns matrix.
    # We will build a synthetic returns matrix containing only IBOV, IVVB11, and CDI
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    ivvb_px = download_benchmark(IVVB11_TICKER, START_DATE, END_DATE)
    
    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ivvb_ret = ivvb_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1
    
    # Align dates
    df = pd.DataFrame({"IBOV_ASSET": ibov_ret, "IVVB11_ASSET": ivvb_ret, "CDI_ASSET": cdi_ret}).dropna()
    
    # Calculate 3-Month Momentum
    mom_ibov = np.log1p(df["IBOV_ASSET"]).rolling(LOOKBACK_MONTHS).sum()
    mom_ivvb = np.log1p(df["IVVB11_ASSET"]).rolling(LOOKBACK_MONTHS).sum()
    
    target_weights = pd.DataFrame(0.0, index=df.index, columns=df.columns)
    
    start_idx = LOOKBACK_MONTHS
    
    print("\n🧠 Generating target weights matrix...")
    for i in range(start_idx, len(df)):
        # Use i-1 to prevent look-ahead bias
        i_mom = mom_ibov.iloc[i-1]
        v_mom = mom_ivvb.iloc[i-1]
        
        # Absolute Momentum: If both are negative, go to cash (CDI)
        if i_mom <= 0 and v_mom <= 0:
            target_weights.iloc[i]["CDI_ASSET"] = 1.0
        # Relative Momentum: Otherwise pick the highest
        elif i_mom > v_mom:
            target_weights.iloc[i]["IBOV_ASSET"] = 1.0
        else:
            target_weights.iloc[i]["IVVB11_ASSET"] = 1.0

    print(f"\n🚀 Running generic simulation engine ({REBALANCE_FREQ})...")
    result = run_simulation(
        returns_matrix=df,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="Dual Momentum",
        monthly_sales_exemption=20_000,
    )

    common = result["pretax_values"].index.intersection(df.index)
    m_pretax = build_metrics(value_to_ret(result["pretax_values"].loc[common]), "Dual Mom Pre-Tax")
    m_aftertax = build_metrics(value_to_ret(result["aftertax_values"].loc[common]), "After-Tax 15% CGT")
    m_ibov = build_metrics(df["IBOV_ASSET"].loc[common], "IBOV")
    m_cdi = build_metrics(df["CDI_ASSET"].loc[common], "CDI")

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    plot_tax_backtest(
        title=f"Dual Momentum: IBOV vs IVVB11 vs CDI\n{LOOKBACK_MONTHS}M Lookback  ·  15% CGT + {SLIPPAGE*100}% Slippage\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=result["pretax_values"].loc[common],
        aftertax_val=result["aftertax_values"].loc[common],
        ibov_ret=df["IBOV_ASSET"].loc[common],
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=result["tax_paid"].sum(),
        out_path="dual_momentum_backtest.png",
        cdi_ret=df["CDI_ASSET"].loc[common]
    )

if __name__ == "__main__":
    main()

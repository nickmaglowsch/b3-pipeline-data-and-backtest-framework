"""
B3 Multi-Factor + CDI + IVVB11 Strategy Backtest
=======================================================================
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
WEIGHT_EQUITY = 0.33
WEIGHT_CDI = 0.33
WEIGHT_IVVB11 = 0.34

BORROW_SPREAD = 0.02  # 2% annualized spread on margin loans

REBALANCE_FREQ = "ME"  
LOOKBACK_YEARS = 1

period_map = {"ME": 12, "QE": 4, "YE": 1, "W": 52, "W-FRI": 52, "W-MON": 52}
PERIODS_PER_YEAR = period_map.get(REBALANCE_FREQ, 12)
LOOKBACK_PERIODS = int(LOOKBACK_YEARS * PERIODS_PER_YEAR)

TOP_DECILE = 0.10
TAX_RATE = 0.15
SLIPPAGE = 0.001  # 0.1% transaction cost
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"
IVVB11_TICKER = "IVVB11.SA" 

def generate_signals(adj_close, close_px, fin_vol):
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()

    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    log_ret = np.log1p(ret)
    mom_signal = log_ret.shift(1).rolling(LOOKBACK_PERIODS).sum()
    mom_glitch = ((ret > 1.0) | (ret < -0.90)).shift(1).rolling(LOOKBACK_PERIODS).max()
    mom_signal[mom_glitch == 1] = np.nan

    vol_signal = -ret.shift(1).rolling(LOOKBACK_PERIODS).std()
    vol_glitch = ((ret > 1.0) | (ret < -0.90)).shift(1).rolling(LOOKBACK_PERIODS).max()
    vol_signal[vol_glitch == 1] = np.nan

    mom_rank = mom_signal.rank(axis=1, pct=True)
    vol_rank = vol_signal.rank(axis=1, pct=True)

    composite = (mom_rank * 0.5) + (vol_rank * 0.5)

    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    
    # We add external assets columns
    target_weights["CDI_ASSET"] = 0.0
    target_weights["IVVB11_ASSET"] = 0.0

    start_idx = LOOKBACK_PERIODS + 2
    
    prev_sel = set()
    
    for i in range(start_idx, len(ret)):
        # Important: the logic is what target weights should be FOR month i.
        # This decision can only use data up to i-1.
        sig_row = composite.iloc[i - 1]
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]
        
        valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)
        valid_eq = sig_row[valid_mask].dropna()
        
        if len(valid_eq) >= 5:
            n_sel = max(1, int(len(valid_eq) * TOP_DECILE))
            sel = set(valid_eq.nlargest(n_sel).index)
        else:
            sel = prev_sel
            
        if not sel:
            continue
            
        # Assign weights
        eq_weight_per_stock = WEIGHT_EQUITY / len(sel)
        for t in sel:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = eq_weight_per_stock
            
        target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = WEIGHT_CDI
        target_weights.iloc[i, target_weights.columns.get_loc("IVVB11_ASSET")] = WEIGHT_IVVB11
        
        prev_sel = sel

    return ret, target_weights

def main():
    print("\n" + "=" * 70)
    print("  B3 MULTI-FACTOR + CDI + IVVB11 (WITH GENERIC ENGINE)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ivvb_px = download_benchmark(IVVB11_TICKER, START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    
    ivvb_ret = ivvb_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    # If margin loan, add borrowing spread
    if WEIGHT_CDI < 0:
        cdi_ret += BORROW_SPREAD / PERIODS_PER_YEAR

    print("\n🧠 Generating target weights matrix...")
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol)
    
    # Merge external asset returns into the main returns matrix
    ret["CDI_ASSET"] = cdi_ret
    ret["IVVB11_ASSET"] = ivvb_ret
    
    # Fill NAs in returns to 0 to prevent issues during apply
    ret = ret.fillna(0.0)

    print(f"\n🚀 Running generic simulation engine ({REBALANCE_FREQ})...")
    result = run_simulation(
        returns_matrix=ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name=f"{int(WEIGHT_EQUITY * 100)}/{int(WEIGHT_CDI * 100)}/{int(WEIGHT_IVVB11 * 100)}",
        monthly_sales_exemption=20_000,
    )

    common = result["pretax_values"].index.intersection(ibov_ret.index)
    pretax_val = result["pretax_values"].loc[common]
    aftertax_val = result["aftertax_values"].loc[common]
    ibov_ret = ibov_ret.loc[common]
    
    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = result["tax_paid"].sum()

    m_pretax = build_metrics(pretax_ret, f"{int(WEIGHT_EQUITY*100)}/{int(WEIGHT_CDI*100)}/{int(WEIGHT_IVVB11*100)} Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret.loc[common], "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    plot_tax_backtest(
        title=f"{int(WEIGHT_EQUITY * 100)}% Multi-Factor / {int(WEIGHT_CDI * 100)}% CDI / {int(WEIGHT_IVVB11 * 100)}% IVVB11\n{REBALANCE_FREQ} Rebalance  ·  15% CGT + {SLIPPAGE*100}% Slippage\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=total_tax,
        out_path="multifactor_cdi_ivvb11_backtest.png",
        cdi_ret=cdi_ret.loc[common]
    )

if __name__ == "__main__":
    main()

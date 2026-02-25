"""
B3 Regime-Switching Multi-Factor Backtest
=======================================================================
Strategy: In emerging markets, macro drawdowns will destroy even the best 
stock picking models. This strategy uses a classic Trend-Following regime filter.

If the IBOV benchmark is > its 10-month Moving Average:
   -> Buy the Top 10% Multi-Factor stocks (Momentum + Low Vol).
If the IBOV benchmark is < its 10-month Moving Average:
   -> Sell all equities and sit safely in 100% CDI (Risk-free yield).
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
LOOKBACK_YEARS = 1

period_map = {"ME": 12, "QE": 4, "YE": 1, "W": 52, "W-FRI": 52, "W-MON": 52}
PERIODS_PER_YEAR = period_map.get(REBALANCE_FREQ, 12)
LOOKBACK_PERIODS = int(LOOKBACK_YEARS * PERIODS_PER_YEAR)
SKIP_PERIODS = 1 if REBALANCE_FREQ == "ME" else 0

MACRO_TREND_PERIODS = 10 if REBALANCE_FREQ == "ME" else int(10 * PERIODS_PER_YEAR / 12)

TOP_DECILE = 0.10
TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"

def generate_signals(adj_close, close_px, fin_vol, ibov_px):
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()
    
    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()
    
    # ── Multi-Factor Signal (Mom + Low Vol) ──
    log_ret = np.log1p(ret)
    mom_signal = log_ret.shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).sum()
    mom_glitch = ((ret > 1.0) | (ret < -0.90)).shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).max()
    mom_signal[mom_glitch == 1] = np.nan
    
    vol_signal = -ret.shift(1).rolling(LOOKBACK_PERIODS).std()
    vol_glitch = ((ret > 1.0) | (ret < -0.90)).shift(1).rolling(LOOKBACK_PERIODS).max()
    vol_signal[vol_glitch == 1] = np.nan

    mom_rank = mom_signal.rank(axis=1, pct=True)
    vol_rank = vol_signal.rank(axis=1, pct=True)
    composite = (mom_rank * 0.5) + (vol_rank * 0.5)
    
    # ── Macro Regime Filter ──
    # Is the Ibovespa index price strictly > its 10-period SMA?
    ibov_sampled = ibov_px.resample(REBALANCE_FREQ).last()
    # We shift by 1 to prevent look-ahead bias on the macro filter
    ibov_sma = ibov_sampled.shift(1).rolling(MACRO_TREND_PERIODS).mean()
    is_bull_market = ibov_sampled.shift(1) > ibov_sma
    
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    target_weights["CDI_ASSET"] = 0.0
    
    start_idx = max(LOOKBACK_PERIODS + SKIP_PERIODS, MACRO_TREND_PERIODS) + 1
    
    prev_sel = set()
    for i in range(start_idx, len(ret)):
        date = ret.index[i]
        
        # Check Macro Regime for this month
        bull_regime = is_bull_market.iloc[i]
        
        if not bull_regime:
            # Bear Market: 100% CDI
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            prev_sel = set()
            continue
            
        # Bull Market: 100% Equities
        sig_row = composite.iloc[i - 1]
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]
        
        valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)
        valid = sig_row[valid_mask].dropna()
        
        if len(valid) < 5:
            sel = prev_sel
        else:
            n_sel = max(1, int(len(valid) * TOP_DECILE))
            sel = set(valid.nlargest(n_sel).index)
            
        if not sel: 
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            continue
            
        weight_per_stock = 1.0 / len(sel)
        for t in sel:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight_per_stock
        prev_sel = sel

    return ret, target_weights

def main():
    print("\n" + "=" * 70)
    print("  B3 REGIME-SWITCHING MULTI-FACTOR BACKTEST (15% CGT)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    
    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n🧠 Generating target weights matrix...")
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol, ibov_px)
    
    ret["CDI_ASSET"] = cdi_ret
    ret = ret.fillna(0.0)

    print(f"\n🚀 Running generic simulation engine ({REBALANCE_FREQ})...")
    result = run_simulation(ret, target_weights, INITIAL_CAPITAL, TAX_RATE, SLIPPAGE, name="Regime-Switching")

    common = result["pretax_values"].index.intersection(ibov_ret.index)
    m_pretax = build_metrics(value_to_ret(result["pretax_values"].loc[common]), "Regime Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(value_to_ret(result["aftertax_values"].loc[common]), "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret.loc[common], "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret.loc[common], "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    plot_tax_backtest(
        title=f"Regime Switching: Bull = Multi-Factor, Bear = 100% CDI\n{MACRO_TREND_PERIODS}M Trend Filter  ·  15% CGT + {SLIPPAGE*100}% Slippage\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=result["pretax_values"].loc[common],
        aftertax_val=result["aftertax_values"].loc[common],
        ibov_ret=ibov_ret.loc[common],
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=result["tax_paid"].sum(),
        out_path="regime_switching_backtest.png",
        cdi_ret=cdi_ret.loc[common]
    )

if __name__ == "__main__":
    main()

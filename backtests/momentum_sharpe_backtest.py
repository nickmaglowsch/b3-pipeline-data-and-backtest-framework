"""
B3 Momentum Strategy (Top 20, Sharpe-Weighted)
=======================================================================
Strategy: Uses purely 12-Month Momentum to rank and pick the Top 20 stocks.
Once selected, capital is allocated based on their historical Sharpe Ratio 
(stocks with higher Sharpe ratios get proportionally more capital).
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

PORTFOLIO_SIZE = 50  # Always buy exactly 20 stocks
TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"

def generate_signals(adj_close, close_px, fin_vol):
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()
    
    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    # ── Signal Generation (Pure Momentum) ──
    log_ret = np.log1p(ret)
    mom_signal = log_ret.shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).sum()
    mom_glitch = ((ret > 1.0) | (ret < -0.90)).shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).max()
    mom_signal[mom_glitch == 1] = np.nan
    
    # ── Weighting Metric: Historical Sharpe Ratio ──
    vol = ret.shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).std() + 1e-6
    sharpe = mom_signal / vol
    
    # Floor Sharpe at 0.01 to prevent negative/zero weights
    sharpe = sharpe.clip(lower=0.01)
    
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    start_idx = LOOKBACK_PERIODS + SKIP_PERIODS + 1
    
    prev_sel_weights = {}
    
    for i in range(start_idx, len(ret)):
        mom_row = mom_signal.iloc[i - 1]
        sharpe_row = sharpe.iloc[i - 1]
        
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]
        
        valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)
        valid = mom_row[valid_mask].dropna()
        
        # We need at least the portfolio size available to trade
        if len(valid) < PORTFOLIO_SIZE:
            for t, w in prev_sel_weights.items():
                target_weights.iloc[i, target_weights.columns.get_loc(t)] = w
            continue
            
        # Select the absolute Top N based purely on Momentum
        winners = valid.nlargest(PORTFOLIO_SIZE).index
        
        # Extract their Sharpe ratios for weighting
        winner_sharpes = sharpe_row[winners]
        
        # If any sharpe is NaN, fill with cross-sectional median
        if winner_sharpes.isna().any():
            median_sharpe = winner_sharpes.median()
            winner_sharpes = winner_sharpes.fillna(median_sharpe)
            
        # Normalize weights so they sum to 1.0
        total_sharpe = winner_sharpes.sum()
        if total_sharpe > 0:
            weights = winner_sharpes / total_sharpe
        else:
            # Fallback to equal weight if something weird happened
            weights = pd.Series(1.0 / PORTFOLIO_SIZE, index=winners)
            
        prev_sel_weights = {}
        for t, w in weights.items():
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = w
            prev_sel_weights[t] = w

    return ret, target_weights

def main():
    print("\n" + "=" * 70)
    print("  B3 NATIVE MOMENTUM (TOP 20, SHARPE WEIGHTED)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    
    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n🧠 Generating target weights matrix...")
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol)
    
    ret = ret.fillna(0.0)

    print(f"\n🚀 Running generic simulation engine ({REBALANCE_FREQ})...")
    result = run_simulation(
        returns_matrix=ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="Mom (Sharpe Wgt)",
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

    m_pretax = build_metrics(pretax_ret, "Mom(Sharpe) Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret, "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    plot_tax_backtest(
        title=f"Pure Momentum: Top {PORTFOLIO_SIZE} Stocks (Sharpe Weighted)\nR$ {MIN_ADTV / 1_000_000:.0f}M+ ADTV  ·  15% CGT + {SLIPPAGE*100}% Slippage\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=total_tax,
        out_path="momentum_top20_sharpe_backtest.png",
        cdi_ret=cdi_ret
    )

if __name__ == "__main__":
    main()

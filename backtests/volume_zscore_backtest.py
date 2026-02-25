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

START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

def generate_signals(adj_close, close_px, fin_vol):
    ret = adj_close.resample("ME").last().pct_change()
    adtv = fin_vol.resample("ME").mean()
    raw_close = close_px.resample("ME").last()
    
    # ── Signal: Volume Z-Score ──
    # How anomalous was this month's volume compared to the last 12 months?
    vol_mean = adtv.shift(1).rolling(12).mean()
    vol_std = adtv.shift(1).rolling(12).std()
    
    z_score = (adtv.shift(1) - vol_mean) / vol_std
    signal = z_score
    
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    
    prev_sel = set()
    for i in range(14, len(ret)):
        sig_row = signal.iloc[i - 1]
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]
        
        valid_mask = (adtv_row >= 2_000_000) & (raw_close_row >= 2.0)
        valid = sig_row[valid_mask].dropna()
        
        if len(valid) >= 5:
            # Buy the top 20 stocks with the craziest volume spikes
            n_sel = 20
            sel = set(valid.nlargest(n_sel).index)
        else:
            sel = prev_sel
            
        if not sel: continue
        weight_per_stock = 1.0 / len(sel)
        for t in sel:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight_per_stock
        prev_sel = sel

    return ret, target_weights

def main():
    adj_close, close_px, fin_vol = load_b3_data("b3_market_data.sqlite", START_DATE, END_DATE)
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol)
    result = run_simulation(ret.fillna(0.0), target_weights, 100000, 0.15, 0.001, "Vol Spike")
    ann = (1 + result['aftertax_values'].pct_change().dropna()).prod() ** (12 / len(result['aftertax_values'])) - 1
    print(f"\nVolume Z-Score Strategy Return: {ann*100:.2f}%")

if __name__ == "__main__":
    main()

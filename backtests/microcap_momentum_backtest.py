import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import load_b3_data
from core.simulation import run_simulation

def generate_signals(adj_close, close_px, fin_vol):
    ret = adj_close.resample("ME").last().pct_change()
    adtv = fin_vol.resample("ME").mean()
    raw_close = close_px.resample("ME").last()
    
    log_ret = np.log1p(ret)
    mom_signal = log_ret.shift(1).rolling(6).sum()
    
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    
    prev_sel = set()
    for i in range(8, len(ret)):
        sig_row = mom_signal.iloc[i - 1]
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]
        
        # KEY: ADTV must be > 500k but LESS than 10M (Microcaps only)
        valid_mask = (adtv_row >= 500_000) & (adtv_row <= 5_000_000) & (raw_close_row >= 1.0)
        valid = sig_row[valid_mask].dropna()
        
        if len(valid) >= 5:
            n_sel = max(5, int(len(valid) * 0.10))
            sel = set(valid.nlargest(n_sel).index)
        else:
            sel = prev_sel
            
        if not sel: continue
        weight_per_stock = 1.0 / len(sel)
        for t in sel:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight_per_stock
        prev_sel = sel

    return ret, target_weights

adj_close, close_px, fin_vol = load_b3_data("b3_market_data.sqlite", "2012-01-01", "2026-02-24")
ret, target_weights = generate_signals(adj_close, close_px, fin_vol)
result = run_simulation(ret.fillna(0.0), target_weights, 100000, 0.15, 0.001)
ann = (1 + result['aftertax_values'].pct_change().dropna()).prod() ** (12 / len(result['aftertax_values'])) - 1
print(f"\nMicro-Cap Momentum Return: {ann*100:.2f}%")

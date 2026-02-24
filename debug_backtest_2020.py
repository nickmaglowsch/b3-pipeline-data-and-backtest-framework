import pandas as pd
import numpy as np
from backtests.momentum_tax_backtest import load_b3_data, momentum_signal, run_backtest, MIN_ADTV, TOP_DECILE, LOOKBACK_MONTHS, SKIP_MONTHS

START_DATE = "2000-01-01"
END_DATE = "2026-02-24"
DB_PATH = "b3_market_data.sqlite"

adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

monthly_px = adj_close.resample("ME").last()
monthly_ret = monthly_px.pct_change()
monthly_raw_close = close_px.resample("ME").last()
monthly_adtv = fin_vol.resample("ME").mean()
signal = momentum_signal(monthly_ret, LOOKBACK_MONTHS, SKIP_MONTHS)

start_idx = LOOKBACK_MONTHS + SKIP_MONTHS + 1

worst_months = []

for i in range(start_idx, len(monthly_ret)):
    date = monthly_ret.index[i]
    if str(date.date()) == "2005-10-31":
        sig_row = signal.iloc[i - 1]
        adtv_row = monthly_adtv.iloc[i - 1]
        raw_close_row = monthly_raw_close.iloc[i - 1]
        valid_universe_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)
        valid = sig_row[valid_universe_mask].dropna()
        n_select = max(1, int(len(valid) * TOP_DECILE))
        new_selected = list(valid.nlargest(n_select).index)
        ret_row = monthly_ret.iloc[i]
        portfolio_returns = [ret_row.get(t, 0.0) for t in new_selected]
        for t, r in zip(new_selected, portfolio_returns):
            print(f"Ticker {t}: return {r*100:.2f}%, raw close {raw_close_row.get(t)}, adtv {adtv_row.get(t):.0f}")


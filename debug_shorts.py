import sqlite3
import pandas as pd
from backtests.core.data import load_b3_data
from backtests.core.simulation import run_simulation
from backtests.bull_trap_short_backtest import generate_signals

adj_close, close_px, fin_vol = load_b3_data("b3_market_data.sqlite", "2014-01-01", "2014-04-01")
ret, tw = generate_signals(adj_close, close_px, fin_vol)
ret["CDI_ASSET"] = 0.002
ret = ret.fillna(0.0)

res = run_simulation(ret, tw, 100000, 0.15, 0.001)
print(res)


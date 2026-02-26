import pandas as pd
from backtests.core.simulation import run_simulation
from backtests.bull_trap_short_backtest import generate_signals
from backtests.core.data import load_b3_data
adj, close, vol = load_b3_data("b3_market_data.sqlite", "2014-01-01", "2014-04-01")

ret, target_weights = generate_signals(adj, close, vol)
print("Target weights head:")
print(target_weights.iloc[6][target_weights.iloc[6] != 0])

import pandas as pd
from backtests.core.data import load_b3_data
adj, close, vol = load_b3_data("b3_market_data.sqlite", "2014-01-01", "2014-03-01")
ret = adj.resample("W-FRI").last().pct_change()
print(ret.loc["2014-02-07"].max())
print(ret.loc["2014-02-07"].idxmax())
print(ret.loc["2014-02-07"]["VIVT3"])

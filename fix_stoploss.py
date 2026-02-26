import pandas as pd
from backtests.core.simulation import _apply_returns
pos = {"A": {"cost_basis": -2000, "current_value": -2000}}
# To implement a stop loss of 50%, we just cap the max_ret allowed against short positions
# Or the simulation engine could explicitly trigger a margin call stop loss.

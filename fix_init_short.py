import pandas as pd
from backtests.core.simulation import run_simulation
returns = pd.DataFrame({"PETR4": [0.0, 0.1, -0.1], "CDI": [0.0, 0.01, 0.01]})
weights = pd.DataFrame({"PETR4": [-1.0, -1.0, -1.0], "CDI": [2.0, 2.0, 2.0]}) # Hold 200k CDI, short 100k PETR
# Initial NAV = 100k
res = run_simulation(returns, weights)
print(res)

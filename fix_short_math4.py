import pandas as pd
from backtests.core.simulation import run_simulation

df = pd.DataFrame({"A": [0.0, 50.0], "CDI_ASSET": [0.0, 0.01]})
tw = pd.DataFrame({"A": [-1.0, -1.0], "CDI_ASSET": [2.0, 2.0]})

# Initial capital 100_000. 
# We put -100_000 in A. We put +200_000 in CDI.
# Next month, A goes up 5000% (x 51.0). Current value becomes -5,100,000.
# CDI goes to 202,000.
# NAV = 202,000 - 5,100,000 = -4,898,000.
# BANKRUPT.

import pandas as pd
from backtests.core.simulation import run_simulation

returns = pd.DataFrame({
    "TOYB3": [0.0, 50.0, -0.5, 0.1],
    "CDI_ASSET": [0.0, 0.002, 0.002, 0.002]
}, index=pd.date_range("2014-01-01", periods=4, freq="W-FRI"))

# We want 100k CDI, -100k short (total net is 0 target weights?)
# Wait, if we want 100k CDI and -20k short on a 100k portfolio, target weights should be:
# 1.0 CDI, -0.2 Short. (Sum = 0.8)
# Is my engine normalizing them?

import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.simulation import run_simulation

adj_close, close_px, fin_vol = load_b3_data("b3_market_data.sqlite", "2010-01-01", "2026-02-24")
cdi_daily = download_cdi_daily("2010-01-01", "2026-02-24")

px = adj_close.resample("ME").last()
ret = px.pct_change()
cdi_monthly = (1 + cdi_daily).resample("ME").prod() - 1

# Are we in an easing cycle? 
# If current month's CDI rate is lower than 3 months ago.
is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
target_weights["CDI_ASSET"] = 0.0

ibov_px = download_benchmark("^BVSP", "2010-01-01", "2026-02-24")
ibov_monthly = ibov_px.resample("ME").last().pct_change().dropna()
ret["IBOV"] = ibov_monthly
ret["CDI_ASSET"] = cdi_monthly

for i in range(5, len(ret)):
    if is_easing.iloc[i]:
        target_weights.iloc[i]["IBOV"] = 1.0
    else:
        target_weights.iloc[i]["CDI_ASSET"] = 1.0

result = run_simulation(ret.fillna(0.0), target_weights, 100000, 0.15, 0.001)
ann = (1 + result['aftertax_values'].pct_change().dropna()).prod() ** (12 / len(result['aftertax_values'])) - 1
print(f"\nCOPOM Easing Strategy Return: {ann*100:.2f}%")

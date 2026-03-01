import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.data import load_b3_data, download_benchmark
from core.simulation import run_simulation

ibov_px = download_benchmark("^BVSP", "2010-01-01", "2026-02-24")
ivvb_px = download_benchmark("IVVB11.SA", "2010-01-01", "2026-02-24")

# Resample to monthly
ibov_monthly = ibov_px.resample("ME").last()
ivvb_monthly = ivvb_px.resample("ME").last()

# Returns
ret = pd.DataFrame({
    "IBOV": ibov_monthly.pct_change(),
    "IVVB11": ivvb_monthly.pct_change()
}).dropna()

# Signal: 10-month SMA of IBOV
sma10 = ibov_monthly.shift(1).rolling(10).mean()
bull_market = ibov_monthly.shift(1) > sma10

target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
for i in range(11, len(ret)):
    if bull_market.iloc[i]:
        target_weights.iloc[i]["IBOV"] = 1.0
    else:
        target_weights.iloc[i]["IVVB11"] = 1.0

result = run_simulation(ret.fillna(0.0), target_weights, 100000, 0.15, 0.001, monthly_sales_exemption=20_000)
ann = (1 + result['aftertax_values'].pct_change().dropna()).prod() ** (12 / len(result['aftertax_values'])) - 1
print(f"\nGlobal Flight-to-Quality Return: {ann*100:.2f}%")

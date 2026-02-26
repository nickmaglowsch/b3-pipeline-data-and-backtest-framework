import pandas as pd
from backtests.core.simulation import _execute_rebalance

positions = {"CDI_ASSET": {"cost_basis": 100000, "current_value": 100000}}
target_weights = pd.Series({"CDI_ASSET": 1.0, "PETR4": -0.10, "VALE3": -0.10})

tax, cf, drag, turn = _execute_rebalance(positions, target_weights, 100000, 0.002, 0.15, 0.0)

print(f"Tax: {tax}, CF: {cf}, Drag: {drag}, Turn: {turn}")
print("Positions:")
for k, v in positions.items():
    print(k, v)

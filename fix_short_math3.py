import pandas as pd
from backtests.core.simulation import _execute_rebalance

positions = {"CDI_ASSET": {"cost_basis": 100000, "current_value": 100000}, "PETR4": {"cost_basis": -10000, "current_value": -8000}} # Short made 2k

# Now target is back to 0 (cover the short)
target_weights = pd.Series({"CDI_ASSET": 1.0, "PETR4": 0.0})

# NAV = 100k - 8k = 92k
tax, cf, drag, turn = _execute_rebalance(positions, target_weights, 92000, 0.002, 0.15, 0.0)

print(f"Tax: {tax}, CF: {cf}, Drag: {drag}, Turn: {turn}")
print("Positions:")
for k, v in positions.items():
    print(k, v)

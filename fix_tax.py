from backtests.core.simulation import _compute_tax

pos = {"A": {"cost_basis": -2000, "current_value": -3000}} # Short lost 1000
sales = {"A": 3000} # Sell to cover

tax, cf = _compute_tax(sales, pos, 0.0, 0.15)
print(f"Tax: {tax}, CF: {cf}")

pos2 = {"B": {"cost_basis": -2000, "current_value": -1000}} # Short made 1000
sales2 = {"B": 1000}

tax2, cf2 = _compute_tax(sales2, pos2, 0.0, 0.15)
print(f"Tax: {tax2}, CF: {cf2}")

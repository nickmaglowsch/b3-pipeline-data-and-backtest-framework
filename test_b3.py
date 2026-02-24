import logging
import sys
from b3_pipeline import b3_corporate_actions, storage

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

tickers = ["PETR3", "PETR4"]
trading_names = ["PETR"]

corp, stock = b3_corporate_actions.fetch_all_corporate_actions(trading_names, tickers)

print("Corporate Actions:")
print(corp.head(10) if not corp.empty else "Empty")

print("\nStock Actions:")
print(stock.head(10) if not stock.empty else "Empty")

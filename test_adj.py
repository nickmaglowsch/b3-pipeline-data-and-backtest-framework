import sqlite3
import pandas as pd
from b3_pipeline import adjustments, storage

conn = storage.get_connection()
prices = storage.get_prices_for_ticker(conn, "PETR4")
corp_actions = storage.get_all_corporate_actions(conn)
stock_actions = storage.get_all_stock_actions(conn)

# Filter for PETR4
corp_actions = corp_actions[corp_actions["ticker"] == "PETR4"]
stock_actions = stock_actions[stock_actions["ticker"] == "PETR4"]

adj_prices, splits = adjustments.compute_all_adjustments(prices, corp_actions, stock_actions)

# Show recent prices
print("Recent Prices (last 5 days):")
print(adj_prices[["date", "close", "split_adj_close", "adj_close"]].tail(5))

# Show historically old prices to verify split/dividend math
print("\nHistorical Prices (from 2005, pre-split):")
print(adj_prices[["date", "close", "split_adj_close", "adj_close"]].head(5))

conn.close()

import sqlite3
import sys
import pandas as pd
from b3_pipeline import adjustments, storage

conn = storage.get_connection()
print("Loading data...")
prices = storage.get_all_prices(conn)
corp_actions = storage.get_all_corporate_actions(conn)
stock_actions = storage.get_all_stock_actions(conn)

print(f"Loaded {len(prices)} prices, {len(corp_actions)} corp actions, {len(stock_actions)} stock actions")

adj_prices, splits = adjustments.compute_all_adjustments(prices, corp_actions, stock_actions)

print("Updating db...")
storage.update_adjusted_columns(conn, adj_prices)
print("Done")

# Verify
df = pd.read_sql_query("SELECT date, ticker, close, split_adj_close, adj_close FROM prices WHERE ticker = 'PETR4' ORDER BY date DESC LIMIT 5", conn)
print("\nRecent Prices from DB:")
print(df)

conn.close()

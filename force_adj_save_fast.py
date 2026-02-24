import sqlite3
import pandas as pd
from b3_pipeline import adjustments, storage
import time

conn = storage.get_connection()
print("Loading data...")
prices = storage.get_prices_for_ticker(conn, "PETR4")
corp_actions = storage.get_all_corporate_actions(conn)
stock_actions = storage.get_all_stock_actions(conn)

corp_actions = corp_actions[corp_actions["ticker"] == "PETR4"]
stock_actions = stock_actions[stock_actions["ticker"] == "PETR4"]

print("Computing adjustments for PETR4...")
adj_prices, splits = adjustments.compute_all_adjustments(prices, corp_actions, stock_actions)

print("Saving adjusted prices for PETR4...")
cursor = conn.cursor()
sql = """
    UPDATE prices 
    SET split_adj_close = ?, adj_close = ? 
    WHERE ticker = ? AND date = ?
"""
records = []
for _, row in adj_prices.iterrows():
    records.append((row["split_adj_close"], row["adj_close"], row["ticker"], row["date"]))
    
cursor.executemany(sql, records)
conn.commit()

df = pd.read_sql_query("SELECT date, ticker, close, split_adj_close, adj_close FROM prices WHERE ticker = 'PETR4' ORDER BY date DESC LIMIT 5", conn)
print("\nRecent Prices from DB for PETR4:")
print(df)
conn.close()

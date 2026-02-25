import sqlite3
import pandas as pd
from b3_pipeline import adjustments, storage

conn = storage.get_connection()
prices = storage.get_prices_for_ticker(conn, "PETR4")
corp = pd.read_sql_query("SELECT * FROM corporate_actions WHERE isin_code = 'BRPETRACNPR6'", conn)
print(f"PETR4 Corp Actions: {len(corp)}")
print(corp.head())

# Run dividend adjustments
div_adj = adjustments.compute_dividend_adjustment_factors(prices, corp)
print(div_adj[['date', 'split_adj_close', 'adj_close']].tail(5))

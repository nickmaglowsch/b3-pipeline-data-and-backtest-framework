import sqlite3
import pandas as pd
conn = sqlite3.connect("b3_market_data.sqlite")

df = pd.read_sql_query("SELECT date, ticker, close, split_adj_close, adj_close FROM prices WHERE ticker = 'ITAU4' AND date LIKE '2005-09%' ORDER BY date ASC", conn)
print("ITAU4 Prices September 2005:")
print(df)


import sqlite3
import pandas as pd
conn = sqlite3.connect("b3_market_data.sqlite")
splits = pd.read_sql_query("SELECT * FROM stock_actions WHERE ticker = 'VIVT3'", conn)
print(splits)

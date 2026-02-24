import sqlite3
import pandas as pd
conn = sqlite3.connect("b3_market_data.sqlite")

itau = pd.read_sql_query("SELECT ticker, MIN(date) as first_date, MAX(date) as last_date, COUNT(*) as days FROM prices WHERE ticker IN ('ITAU4', 'ITUB4') GROUP BY ticker", conn)
print(itau)


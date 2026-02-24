import sqlite3
import pandas as pd
conn = sqlite3.connect("b3_market_data.sqlite")

divs = pd.read_sql_query("SELECT ticker, COUNT(*) as count, MIN(event_date) as first_div, MAX(event_date) as last_div FROM corporate_actions WHERE ticker IN ('ITAU4', 'ITUB4') GROUP BY ticker", conn)
print(divs)


import sqlite3
import pandas as pd

conn = sqlite3.connect("b3_market_data.sqlite")
df = pd.read_sql_query(
    """
    SELECT date, ticker, close, split_adj_close, adj_close 
    FROM prices 
    WHERE ticker = 'PETR4' 
    ORDER BY date DESC 
    LIMIT 5
""",
    conn,
)
print(df)
conn.close()

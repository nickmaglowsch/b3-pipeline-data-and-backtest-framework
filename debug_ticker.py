import sqlite3
import pandas as pd
conn = sqlite3.connect("b3_market_data.sqlite")

df = pd.read_sql_query("SELECT date, ticker, close, split_adj_close, adj_close FROM prices WHERE ticker = 'VIVT3' ORDER BY date DESC", conn)
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")
monthly = df.resample("ME").last()
monthly["ret"] = monthly["adj_close"].pct_change()
print("VIVT3 Top Monthly Returns:")
print(monthly.nlargest(5, "ret"))

print("\nVIVT3 Actions:")
actions = pd.read_sql_query("SELECT * FROM corporate_actions WHERE ticker = 'VIVT3' ORDER BY event_date DESC LIMIT 5", conn)
print(actions)


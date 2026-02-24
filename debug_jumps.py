import sqlite3
import pandas as pd
import numpy as np

conn = sqlite3.connect("b3_market_data.sqlite")
df = pd.read_sql_query("SELECT date, ticker, close, adj_close, volume/100.0 as fin_volume FROM prices WHERE date >= '2020-01-01' AND ((LENGTH(ticker) = 5 AND SUBSTR(ticker, 5, 1) IN ('3', '4', '5', '6')) OR (LENGTH(ticker) = 6 AND SUBSTR(ticker, 5, 2) = '11'))", conn)
df["date"] = pd.to_datetime(df["date"])

adj_close = df.pivot(index="date", columns="ticker", values="adj_close").ffill()
monthly_px = adj_close.resample("ME").last()
monthly_ret = monthly_px.pct_change()

max_ret = monthly_ret.max()
print("Top massive monthly returns:")
for t, r in max_ret.nlargest(10).items():
    print(f"{t}: {r*100:.1f}%")

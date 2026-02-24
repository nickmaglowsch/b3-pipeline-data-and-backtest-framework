import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

conn = sqlite3.connect("b3_market_data.sqlite")

query = f"""
    SELECT date, ticker, close, adj_close, volume/100.0 as fin_volume
    FROM prices
    WHERE date >= '2000-01-01'
    AND (
        (LENGTH(ticker) = 5 AND SUBSTR(ticker, 5, 1) IN ('3', '4', '5', '6'))
        OR 
        (LENGTH(ticker) = 6 AND SUBSTR(ticker, 5, 2) = '11')
    )
"""
df = pd.read_sql_query(query, conn)
df["date"] = pd.to_datetime(df["date"])

adj_close = df.pivot(index="date", columns="ticker", values="adj_close").ffill()
monthly_px = adj_close.resample("ME").last()
monthly_ret = monthly_px.pct_change()

# Look at the cumulative return over time of a naive equal weight to see where the drop happens
avg_ret = monthly_ret.mean(axis=1)
cum_ret = (1 + avg_ret).cumprod()

print("Worst 10 months for average market return:")
print(avg_ret.nsmallest(10) * 100)


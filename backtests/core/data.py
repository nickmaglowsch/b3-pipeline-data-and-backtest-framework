"""
Core utilities for backtesting with B3 data.
"""

import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime


def load_b3_data(
    db_path: str, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from the local B3 SQLite database.
    Filters for valid standard lot tickers (ending in 3, 4, 5, 6, 11).
    Calculates daily financial volume correctly.

    Returns:
        tuple: (adj_close, close_px, fin_vol)
        All as wide DataFrames with shape (date, ticker)
    """
    print(f"⬇  Loading B3 data from {db_path} ({start} to {end})...")

    # Query: standard lot tickers ending in 3, 4, 5, 6, 11
    # BDRs (34, 35, etc.) and other weird assets are excluded using strict length and suffix checks.
    query = """
        SELECT date, ticker, close, adj_close, volume
        FROM prices
        WHERE date >= ? AND date <= ?
        AND (
            (LENGTH(ticker) = 5 AND SUBSTR(ticker, 5, 1) IN ('3', '4', '5', '6'))
            OR
            (LENGTH(ticker) = 6 AND SUBSTR(ticker, 5, 2) = '11')
        )
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[start, end])

    df["date"] = pd.to_datetime(df["date"])

    # Calculate daily financial volume in BRL.
    # In our DB, the `volume` column comes from COTAHIST VOLTOT (Financial Volume),
    # but it is parsed as an integer with 2 implied decimal places.
    # Therefore, true financial volume = volume / 100
    df["fin_volume"] = df["volume"] / 100.0

    # Pivot to wide format
    print("🔄  Pivoting data to wide format...")
    adj_close = df.pivot(index="date", columns="ticker", values="adj_close")
    close_px = df.pivot(index="date", columns="ticker", values="close")
    fin_vol = df.pivot(index="date", columns="ticker", values="fin_volume")

    # Forward fill prices to handle missing days, but leave volume as NaN/0
    # to accurately calculate averages
    adj_close = adj_close.ffill()
    close_px = close_px.ffill()

    print(f"✅  Loaded {adj_close.shape[1]} unique standard tickers.")
    return adj_close, close_px, fin_vol


import requests


def download_benchmark(ticker: str, start: str, end: str) -> pd.Series:
    """
    Download benchmark index data from Yahoo Finance.
    """
    print(f"⬇  Downloading {ticker} benchmark from Yahoo...")
    index_data = yf.download(
        ticker, start=start, end=end, auto_adjust=True, progress=False
    )["Close"]

    if isinstance(index_data, pd.DataFrame):
        index_data = index_data.squeeze()

    index_data.index = pd.to_datetime(index_data.index)

    # Remove timezone if present to align with local sqlite dates
    if index_data.index.tz is not None:
        index_data.index = index_data.index.tz_localize(None)

    return index_data


def load_b3_hlc_data(
    db_path: str, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load split-adjusted HLC + adj_close + raw close + financial volume for ATR computation.

    Uses split_adj_* (split-only adjustment) for ATR to avoid false True Range
    spikes from ex-dividend gaps. Returns adj_close separately for the returns
    matrix (simulation needs dividend-adjusted returns).

    Returns:
        tuple: (adj_close, split_adj_high, split_adj_low, split_adj_close, close_px, fin_vol)
        All as wide DataFrames with shape (date, ticker)
    """
    print(f"⬇  Loading B3 HLC data from {db_path} ({start} to {end})...")

    query = """
        SELECT date, ticker, split_adj_high, split_adj_low, split_adj_close,
               adj_close, close, volume
        FROM prices
        WHERE date >= ? AND date <= ?
        AND (
            (LENGTH(ticker) = 5 AND SUBSTR(ticker, 5, 1) IN ('3', '4', '5', '6'))
            OR
            (LENGTH(ticker) = 6 AND SUBSTR(ticker, 5, 2) = '11')
        )
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=[start, end])

    df["date"] = pd.to_datetime(df["date"])
    df["fin_volume"] = df["volume"] / 100.0

    print("🔄  Pivoting HLC data to wide format...")
    adj_close = df.pivot(index="date", columns="ticker", values="adj_close")
    split_high = df.pivot(index="date", columns="ticker", values="split_adj_high")
    split_low = df.pivot(index="date", columns="ticker", values="split_adj_low")
    split_close = df.pivot(index="date", columns="ticker", values="split_adj_close")
    close_px = df.pivot(index="date", columns="ticker", values="close")
    fin_vol = df.pivot(index="date", columns="ticker", values="fin_volume")

    # Forward fill prices to handle missing days, leave volume as-is
    adj_close = adj_close.ffill()
    split_high = split_high.ffill()
    split_low = split_low.ffill()
    split_close = split_close.ffill()
    close_px = close_px.ffill()

    print(f"✅  Loaded HLC for {adj_close.shape[1]} unique standard tickers.")
    return adj_close, split_high, split_low, split_close, close_px, fin_vol


from dateutil.relativedelta import relativedelta


def download_cdi_daily(start: str, end: str) -> pd.Series:
    """
    Download daily accumulated CDI directly from the Brazilian Central Bank (SGS API Series 12).
    Because BCB limits queries to 10 years for daily data, this safely batches requests.
    Returns a pandas Series of daily returns (e.g., 0.0001 for 0.01%).
    """
    print("⬇  Downloading Daily CDI from Brazilian Central Bank...")

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    series = []
    current_start = start_dt

    while current_start <= end_dt:
        # BCB API allows max 10 years for daily data. We'll query in 5-year chunks.
        current_end = min(current_start + relativedelta(years=5), end_dt)

        s_str = current_start.strftime("%d/%m/%Y")
        e_str = current_end.strftime("%d/%m/%Y")

        url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={s_str}&dataFinal={e_str}"

        try:
            response = requests.get(url)
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
                df["valor"] = pd.to_numeric(df["valor"]) / 100.0
                df.set_index("data", inplace=True)
                series.append(df["valor"])
        except Exception as e:
            print(f"Warning: Failed to fetch CDI chunk {s_str} to {e_str}: {e}")

        current_start = current_end + relativedelta(days=1)

    if not series:
        import warnings
        warnings.warn("CDI daily data download returned empty. Check network connectivity.")
        return pd.Series(dtype=float, name="CDI")

    result = pd.concat(series)
    # Ensure no duplicates at chunk boundaries
    result = result[~result.index.duplicated(keep="first")]
    result.name = "CDI"

    return result

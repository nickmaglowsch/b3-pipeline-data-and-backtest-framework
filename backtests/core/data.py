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


# ── Fundamentals (point-in-time) data loading ─────────────────────────────────

_FUNDAMENTALS_METRICS = [
    "revenue",
    "net_income",
    "ebitda",
    "total_assets",
    "equity",
    "net_debt",
    "shares_outstanding",
    "pe_ratio",
    "pb_ratio",
    "ev_ebitda",
]


def _get_rebalance_dates(db_path: str, start: str, end: str, freq: str) -> pd.DatetimeIndex:
    """Build the rebalance date grid by sampling the prices table."""
    with sqlite3.connect(db_path) as conn:
        dates_df = pd.read_sql_query(
            "SELECT DISTINCT date FROM prices WHERE date >= ? AND date <= ? ORDER BY date",
            conn,
            params=[start, end],
        )
    if dates_df.empty:
        return pd.date_range(start=start, end=end, freq=freq)
    daily_idx = pd.to_datetime(dates_df["date"])
    return pd.date_range(start=daily_idx.min(), end=daily_idx.max(), freq=freq)


def load_fundamentals_pit(
    db_path: str,
    metric: str,
    start: str,
    end: str,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Load point-in-time fundamentals for a single metric.

    For each (rebalance_date, ticker), returns the metric value from the most
    recently filed version of the most recently filed period whose filing_date
    <= rebalance_date.

    Args:
        db_path:  Path to the B3 SQLite database file.
        metric:   Column name in fundamentals_pit (e.g. 'pe_ratio', 'revenue').
        start:    Start date string.
        end:      End date string.
        freq:     Rebalance calendar frequency (default 'ME' = month-end).

    Returns:
        Wide DataFrame: DatetimeIndex (rebalance dates) × ticker columns.
        Values are forward-filled from the last filing_date on or before each date.
    """
    if metric not in _FUNDAMENTALS_METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Choose from: {_FUNDAMENTALS_METRICS}")

    sql = f"""
        SELECT
            f.ticker,
            f.filing_date,
            f.period_end,
            f.filing_version,
            f.{metric}
        FROM fundamentals_pit f
        WHERE f.ticker IS NOT NULL
          AND f.filing_date >= :start
          AND f.filing_date <= :end
          AND f.{metric} IS NOT NULL
        ORDER BY f.ticker, f.filing_date, f.filing_version
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params={"start": start, "end": end})

    if df.empty:
        rebal_dates = _get_rebalance_dates(db_path, start, end, freq)
        return pd.DataFrame(index=rebal_dates, dtype=float)

    # Sort so that within a given filing_date, the highest version wins.
    # Do NOT deduplicate across all time — each version must retain its own
    # filing_date so that forward-fill preserves strict point-in-time semantics:
    # a restatement only becomes visible after its own filing_date.
    df = df.sort_values(["ticker", "filing_date", "filing_version"])

    # Pivot: index = filing_date, columns = ticker.
    # aggfunc="last" picks the highest version when multiple versions share
    # the same filing_date (rare, but possible).
    wide = df.pivot_table(
        index="filing_date",
        columns="ticker",
        values=metric,
        aggfunc="last",
    )
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None

    # Reindex to the rebalance calendar and forward-fill
    rebal_dates = _get_rebalance_dates(db_path, start, end, freq)
    wide = wide.reindex(rebal_dates).ffill()

    return wide


def load_all_fundamentals(
    db_path: str,
    start: str,
    end: str,
    freq: str = "ME",
) -> dict:
    """
    Load all 10 fundamentals metrics and return them as a dict of wide DataFrames.

    Returns:
        dict with keys matching _FUNDAMENTALS_METRICS:
        {"revenue": df, "net_income": df, ..., "ev_ebitda": df}
        Each df has DatetimeIndex (rebalance dates) and ticker columns.
    """
    result = {}
    for metric in _FUNDAMENTALS_METRICS:
        try:
            result[metric] = load_fundamentals_pit(db_path, metric, start, end, freq=freq)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load fundamentals metric '{metric}': {e}")
            result[metric] = pd.DataFrame()
    return result

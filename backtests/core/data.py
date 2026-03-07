"""
Core utilities for backtesting with B3 data.
"""

import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime


def load_active_tickers(db_path: str, as_of_date: str) -> set:
    """
    Return the set of ticker roots that were active (not yet delisted)
    as of as_of_date, according to cvm_companies.

    A ticker root is active if:
      - Its delisting_date IS NULL (still listed), OR
      - Its delisting_date >= as_of_date

    Returns a set of 4-character ticker roots (e.g. 'PETR', 'VALE').
    Tickers not in cvm_companies are NOT returned here; the caller's
    conservative filter handles those separately.

    Graceful degradation: returns empty set if cvm_companies lacks
    delisting_date column (un-migrated DB) — no exception is raised.
    """
    sql = """
        SELECT DISTINCT ticker FROM cvm_companies
        WHERE ticker IS NOT NULL
          AND (delisting_date IS NULL OR delisting_date >= ?)
    """
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(sql, conn, params=[as_of_date])
        return set(df["ticker"].dropna().tolist())
    except Exception:
        return set()


def _get_all_known_roots(db_path: str) -> set:
    """Return all ticker roots present in cvm_companies (with or without delisting_date)."""
    sql = "SELECT DISTINCT ticker FROM cvm_companies WHERE ticker IS NOT NULL"
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql_query(sql, conn)
        return set(df["ticker"].dropna().tolist())
    except Exception:
        return set()


def _apply_delisted_filter(
    db_path: str, start: str, *dataframes: pd.DataFrame
) -> tuple:
    """Apply the survivorship bias filter to a set of wide DataFrames.

    Drops columns (tickers) where:
    - The ticker's 4-char root is in cvm_companies (i.e., it's a known company), AND
    - The root's delisting_date is before `start` (i.e., it delisted before the backtest start)

    Tickers whose roots are NOT in cvm_companies at all are kept (conservative).
    """
    active_roots = load_active_tickers(db_path, start)
    all_known_roots = _get_all_known_roots(db_path)

    result = []
    for df in dataframes:
        if df is None or df.empty:
            result.append(df)
            continue
        keep = [
            t for t in df.columns
            if t[:4] in active_roots or t[:4] not in all_known_roots
        ]
        result.append(df[keep])
    return tuple(result)


def load_b3_data(
    db_path: str, start: str, end: str, filter_delisted: bool = False
) -> tuple:
    """
    Load data from the local B3 SQLite database.
    Filters for valid standard lot tickers (ending in 3, 4, 5, 6, 11).
    Calculates daily financial volume correctly.

    Args:
        filter_delisted: If True, drop tickers that were delisted before `start`.
                         Default False — no change to existing behavior.

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

    if filter_delisted:
        adj_close, close_px, fin_vol = _apply_delisted_filter(
            db_path, start, adj_close, close_px, fin_vol
        )

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
    db_path: str, start: str, end: str, filter_delisted: bool = False
) -> tuple:
    """
    Load split-adjusted HLC + adj_close + raw close + financial volume for ATR computation.

    Uses split_adj_* (split-only adjustment) for ATR to avoid false True Range
    spikes from ex-dividend gaps. Returns adj_close separately for the returns
    matrix (simulation needs dividend-adjusted returns).

    Args:
        filter_delisted: If True, drop tickers that were delisted before `start`.
                         Default False — no change to existing behavior.

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

    if filter_delisted:
        adj_close, split_high, split_low, split_close, close_px, fin_vol = _apply_delisted_filter(
            db_path, start,
            adj_close, split_high, split_low, split_close, close_px, fin_vol
        )

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

# All metrics available in fundamentals_monthly (raw financials + ratio columns)
_FUNDAMENTALS_MONTHLY_METRICS = set(_FUNDAMENTALS_METRICS)


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
          AND f.filing_date <= :end
          AND f.{metric} IS NOT NULL
        ORDER BY f.ticker, f.filing_date, f.filing_version
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params={"end": end})

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

    # Reindex to the rebalance calendar and forward-fill.
    # To allow pre-period filings to seed the forward-fill, we first reindex
    # to the union of the filing dates and the rebalance calendar, then ffill,
    # then slice to only the rebalance calendar dates within [start, end].
    rebal_dates = _get_rebalance_dates(db_path, start, end, freq)
    extended_index = wide.index.union(rebal_dates).sort_values()
    wide = wide.reindex(extended_index).ffill()

    # Slice to only the rebalance calendar dates within the requested window
    wide = wide.reindex(rebal_dates)
    wide = wide.loc[wide.index >= pd.Timestamp(start)]

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


# ── fundamentals_monthly snapshot reader ──────────────────────────────────────

def load_fundamentals_monthly(
    db_path: str,
    metric: str,
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Load a single metric from the pre-materialized fundamentals_monthly snapshot.

    Args:
        db_path:  Path to the B3 SQLite database file.
        metric:   Column name in fundamentals_monthly (raw metrics or ratio columns).
        start:    Start date string (inclusive).
        end:      End date string (inclusive).

    Returns:
        Wide DataFrame: DatetimeIndex (month_end) × ticker columns.
        No forward-fill — the snapshot is already forward-filled.

    Raises:
        ValueError: If metric is not in the allowed set.
    """
    if metric not in _FUNDAMENTALS_MONTHLY_METRICS:
        raise ValueError(
            f"Unknown metric {metric!r}. Choose from: {sorted(_FUNDAMENTALS_MONTHLY_METRICS)}"
        )

    # metric is validated above — safe to interpolate in column name
    sql = f"""
        SELECT month_end, ticker, {metric}
        FROM fundamentals_monthly
        WHERE month_end >= ?
          AND month_end <= ?
          AND {metric} IS NOT NULL
        ORDER BY month_end, ticker
    """

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=[start, end])

    if df.empty:
        return pd.DataFrame(dtype=float)

    df["month_end"] = pd.to_datetime(df["month_end"])
    wide = df.pivot_table(index="month_end", columns="ticker", values=metric, aggfunc="last")
    wide.index = pd.to_datetime(wide.index)
    wide.columns.name = None
    return wide


def load_all_fundamentals_monthly(
    db_path: str,
    start: str,
    end: str,
) -> dict:
    """
    Load all metrics from fundamentals_monthly and return them as a dict of wide DataFrames.

    Returns:
        dict with keys matching _FUNDAMENTALS_MONTHLY_METRICS:
        {"revenue": df, ..., "pe_ratio": df, "pb_ratio": df, "ev_ebitda": df}
        Each df has DatetimeIndex (month_end dates) and ticker columns.
    """
    result = {}
    for metric in _FUNDAMENTALS_MONTHLY_METRICS:
        try:
            result[metric] = load_fundamentals_monthly(db_path, metric, start, end)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to load monthly metric '{metric}': {e}")
            result[metric] = pd.DataFrame()
    return result


# ── Dynamic ratio helpers (pure pandas, no DB access) ─────────────────────────

def compute_pe_ratio_dynamic(
    shares: pd.DataFrame,
    net_income: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute P/E ratio from wide DataFrames aligned on the same DatetimeIndex.

    Args:
        shares:     Wide DataFrame: rebalance_date × ticker, shares_outstanding (units).
        net_income: Wide DataFrame: rebalance_date × ticker, net_income in thousands BRL.
        prices:     Wide DataFrame: rebalance_date × ticker, close price (BRL).

    Returns:
        Wide DataFrame with same shape: pe_ratio = (prices × shares) / (net_income × 1000).
        Entries where net_income × 1000 <= 0 or shares <= 0 or prices <= 0 are NaN.
    """
    market_cap = prices * shares
    net_income_brl = net_income * 1_000.0
    result = market_cap / net_income_brl
    # Mask invalid denominators/numerators
    result = result.where(net_income_brl > 0)
    result = result.where(shares > 0)
    result = result.where(prices > 0)
    return result


def compute_pb_ratio_dynamic(
    shares: pd.DataFrame,
    equity: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute P/B ratio from wide DataFrames aligned on the same DatetimeIndex.

    Args:
        shares: Wide DataFrame: shares_outstanding (units).
        equity: Wide DataFrame: equity in thousands BRL.
        prices: Wide DataFrame: close price (BRL).

    Returns:
        Wide DataFrame: pb_ratio = (prices × shares) / (equity × 1000).
        Entries where equity × 1000 <= 0 or shares <= 0 or prices <= 0 are NaN.
    """
    market_cap = prices * shares
    equity_brl = equity * 1_000.0
    result = market_cap / equity_brl
    result = result.where(equity_brl > 0)
    result = result.where(shares > 0)
    result = result.where(prices > 0)
    return result


def compute_ev_ebitda_dynamic(
    shares: pd.DataFrame,
    ebitda: pd.DataFrame,
    net_debt: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute EV/EBITDA from wide DataFrames aligned on the same DatetimeIndex.

    Args:
        shares:   Wide DataFrame: shares_outstanding (units).
        ebitda:   Wide DataFrame: EBITDA in thousands BRL.
        net_debt: Wide DataFrame: net debt in thousands BRL.
        prices:   Wide DataFrame: close price (BRL).

    Returns:
        Wide DataFrame: ev_ebitda = (prices × shares + net_debt × 1000) / (ebitda × 1000).
        Entries where ebitda × 1000 <= 0 or shares <= 0 or prices <= 0 are NaN.
    """
    market_cap = prices * shares
    ebitda_brl = ebitda * 1_000.0
    net_debt_brl = net_debt * 1_000.0
    ev = market_cap + net_debt_brl
    result = ev / ebitda_brl
    result = result.where(ebitda_brl > 0)
    result = result.where(shares > 0)
    result = result.where(prices > 0)
    return result

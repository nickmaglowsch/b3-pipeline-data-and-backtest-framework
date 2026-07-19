"""
Core utilities for backtesting with B3 data.
"""

import glob
import hashlib
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime


# ── Disk+memory cache for external market series (benchmarks, CDI) ────────────
# Every backtest run (and every UI re-run) used to refetch IBOV/ETFs from Yahoo
# and CDI from BCB. A past [start, end] window is effectively immutable, so we
# cache by (kind, *args). Mirrors the pickle+24h-TTL pattern already used in
# research/data_loader.py, but keyed by args so it serves all callers.
_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "market_data"
_CACHE_TTL_SECONDS = 86400  # 24h: refreshes an open-ended `end` (=today) once a day
_MEM_CACHE: dict = {}


def _fetch_cached(kind: str, key_parts: tuple, fetch):
    """Return a cached series or call ``fetch()`` and cache its result.

    Hit order: in-process memory → on-disk pickle (if younger than the TTL) →
    network. Empty/failed fetches are never cached (so a transient outage can't
    poison the cache). Set ``B3_DISABLE_CACHE=1`` to force a fresh refetch.
    The canonical object never escapes — callers always get a ``.copy()`` — so a
    caller mutating its result in place can't corrupt the cache.
    """
    if os.environ.get("B3_DISABLE_CACHE"):
        return fetch()

    key = (kind, *(str(p) for p in key_parts))
    if key in _MEM_CACHE:
        return _MEM_CACHE[key].copy()

    digest = hashlib.md5("|".join(key).encode()).hexdigest()[:16]  # ponytail: filename key, not security
    path = _CACHE_DIR / f"{kind}_{digest}.pkl"
    if path.exists() and (time.time() - path.stat().st_mtime) < _CACHE_TTL_SECONDS:
        try:
            obj = pd.read_pickle(path)
            _MEM_CACHE[key] = obj
            return obj.copy()
        except Exception:
            pass  # corrupt/version-incompatible pickle → fall through and refetch

    obj = fetch()
    if obj is not None and not obj.empty:
        _MEM_CACHE[key] = obj
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            obj.to_pickle(path)
        except Exception:
            pass  # disk cache is best-effort; never fail a backtest over it
        return obj.copy()
    return obj


def load_sector_pit(fca_dir, db_path) -> pd.DataFrame:
    """
    Load point-in-time CVM sector classification joined to tickers.

    Reads every FCA geral CSV (survivorship-free — includes delisted companies)
    from `fca_dir` and maps CNPJ -> ticker via company_tickers_pit. Returns
    long-format [ticker, ref_date, sector]. Returns an empty frame if FCA data
    is unavailable.
    """
    from b3_pipeline.fca_parser import parse_fca_sectors

    zips = sorted(glob.glob(str(Path(fca_dir) / "fca_cia_aberta_*.zip")))
    if not zips:
        return pd.DataFrame(columns=["ticker", "ref_date", "sector"])

    frames = []
    for zp in zips:
        try:
            frames.append(parse_fca_sectors(zp))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["ticker", "ref_date", "sector"])
    sectors = pd.concat(frames, ignore_index=True)

    with sqlite3.connect(str(db_path)) as conn:
        cmap = pd.read_sql_query(
            "SELECT DISTINCT cnpj, ticker FROM company_tickers_pit", conn
        )
    out = sectors.merge(cmap, on="cnpj", how="inner")
    out["ref_date"] = pd.to_datetime(out["ref_date"], errors="coerce")
    return out[["ticker", "ref_date", "sector"]].dropna(subset=["ticker", "ref_date"])


def sector_membership(sector_pit: pd.DataFrame, keywords: list) -> pd.DataFrame:
    """
    Given long-format PIT sector records [ticker, ref_date, sector] and a list of
    sector keywords (case-insensitive substring match), return [ticker,
    effective_date] — one row per FCA filing where the sector matched. A ticker
    is eligible at rebalance date t if any of its effective_date <= t.
    """
    if sector_pit.empty:
        return pd.DataFrame(columns=["ticker", "effective_date"])
    kw = [k.lower() for k in keywords]
    mask = sector_pit["sector"].str.lower().apply(lambda s: any(k in s for k in kw))
    hits = sector_pit[mask]
    return hits[["ticker", "ref_date"]].rename(columns={"ref_date": "effective_date"})


def sector_membership_asof(
    sector_pit: pd.DataFrame, keywords: list, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    As-of sector eligibility: for each date in `dates`, a ticker is eligible iff
    its LATEST filing with ref_date <= date matches one of `keywords`
    (case-insensitive substring). Unlike sector_membership ("ever classified"),
    a later reclassification out of the sector revokes eligibility.

    Returns a boolean DataFrame indexed by `dates` with ticker columns
    (False before a ticker's first filing).
    """
    if sector_pit.empty or len(dates) == 0:
        return pd.DataFrame(index=dates)
    kw = [k.lower() for k in keywords]
    sp = sector_pit.copy()
    sp["match"] = (
        sp["sector"].str.lower().apply(lambda s: any(k in s for k in kw)).astype(float)
    )
    wide = (
        sp.sort_values("ref_date")
        .pivot_table(index="ref_date", columns="ticker", values="match", aggfunc="last")
    )
    wide = wide.reindex(wide.index.union(dates)).ffill().reindex(dates)
    return wide.notna() & (wide > 0)


def _cast_str_cols(batch):
    """Cast large_string/string_view columns to utf8 (StringArray) for Rust.

    pandas 3.x arrow-backed str dtype converts to large_string, but the
    cotahist_rs extension downcasts to arrow-rs StringArray (utf8).
    """
    import pyarrow as pa
    for idx in range(batch.num_columns):
        col = batch.column(idx)
        if pa.types.is_large_string(col.type) or pa.types.is_string_view(col.type):
            batch = batch.set_column(
                idx, batch.schema.field(idx).name, col.cast(pa.utf8())
            )
    return batch


def _pivot_and_ffill_rs(df: "pd.DataFrame"):
    """
    Attempt to call the Rust pivot+ffill implementation.
    Input: long-format DataFrame with columns [date, ticker, close, adj_close, fin_volume].
           The 'date' column must already be datetime64[ns] (as set by the caller).
    Returns: (adj_close, close_px, fin_vol) as wide DataFrames, or None if unavailable.
    """
    try:
        import cotahist_rs
        import pyarrow as pa
    except ImportError:
        return None

    # Rust expects date as "YYYY-MM-DD" string (it came from SQLite as string originally)
    df_for_rust = df[["date", "ticker", "close", "adj_close", "fin_volume"]].copy()
    df_for_rust["date"] = df_for_rust["date"].dt.strftime("%Y-%m-%d")

    long_batch = _cast_str_cols(pa.RecordBatch.from_pandas(df_for_rust, preserve_index=False))

    adj_close_batch, close_px_batch, fin_vol_batch = cotahist_rs.pivot_and_ffill(long_batch)

    def _to_wide_df(batch):
        wide = batch.to_pandas()
        # pandas 3 parses date strings as datetime64[us]; keep the [ns] contract
        wide["date"] = pd.to_datetime(wide["date"]).astype("datetime64[ns]")
        wide = wide.set_index("date")
        wide.columns.name = "ticker"
        return wide

    return _to_wide_df(adj_close_batch), _to_wide_df(close_px_batch), _to_wide_df(fin_vol_batch)


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

    # Daily financial volume in BRL.
    # The parser already divides COTAHIST VOLTOT's 2 implied decimals,
    # so the DB `volume` column is stored in reais (verified: PETR4 ~1e9/day).
    df["fin_volume"] = df["volume"].astype(float)

    # Attempt Rust pivot+ffill (5-15x faster)
    rust_result = _pivot_and_ffill_rs(df)
    if rust_result is not None:
        adj_close, close_px, fin_vol = rust_result
    else:
        # Python fallback
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
    Download benchmark index data from Yahoo Finance (cached by ticker+window).
    """
    def _fetch() -> pd.Series:
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

    return _fetch_cached("bench", (ticker, start, end), _fetch)


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
    # DB `volume` is already in reais (parser handles VOLTOT's implied decimals)
    df["fin_volume"] = df["volume"].astype(float)

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
    Returns a pandas Series of daily returns (e.g., 0.0001 for 0.01%). Cached by window.
    """
    return _fetch_cached("cdi", (start, end), lambda: _download_cdi_daily(start, end))


def _download_cdi_daily(start: str, end: str) -> pd.Series:
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

        # BCB throttles intermittently and returns non-JSON — a silently missing
        # chunk makes every CDI-holding backtest earn 0% cash for years, so retry.
        for attempt in range(3):
            try:
                response = requests.get(url, timeout=30)
                data = response.json()
                if data:
                    df = pd.DataFrame(data)
                    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
                    df["valor"] = pd.to_numeric(df["valor"]) / 100.0
                    df.set_index("data", inplace=True)
                    series.append(df["valor"])
                break
            except Exception as e:
                if attempt == 2:
                    print(f"Warning: Failed to fetch CDI chunk {s_str} to {e_str}: {e}")
                else:
                    import time
                    time.sleep(2 * (attempt + 1))

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
    "net_income_ttm",
    "ebitda",
    "total_assets",
    "equity",
    "net_debt",
    "shares_outstanding",
]
# Ratio columns (pe_ratio, pb_ratio, ev_ebitda) intentionally excluded — compute dynamically at query time

# All metrics available in fundamentals_monthly (raw financials only — no stored ratios)
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
    staleness_days: int = 400,
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
        staleness_days: Values whose source filing is older than this many days
                  become NaN instead of forward-filling forever (mirrors the
                  400-day cut in sp500_b3.pit_snapshot), so delisted companies
                  don't carry years-stale fundamentals.

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
    wide = wide.reindex(extended_index)

    # Track the filing date behind each forward-filled cell so we can blank
    # values once they exceed the staleness cutoff.
    obs_date = pd.DataFrame(
        np.where(wide.notna(), wide.index.values[:, None], np.datetime64("NaT")),
        index=wide.index,
        columns=wide.columns,
    )
    wide = wide.ffill()
    obs_date = obs_date.ffill()
    stale = (wide.index.values[:, None] - obs_date.values) > np.timedelta64(
        staleness_days, "D"
    )
    wide = wide.mask(stale)

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
    Load all fundamentals metrics and return them as a dict of wide DataFrames.

    Returns:
        dict with keys matching _FUNDAMENTALS_METRICS:
        {"revenue": df, "net_income": df, "ebitda": df, "total_assets": df,
         "equity": df, "net_debt": df, "shares_outstanding": df}
        Each df has DatetimeIndex (rebalance dates) and ticker columns.
        Ratio columns (pe_ratio, pb_ratio, ev_ebitda) are excluded — compute dynamically.
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
        {"revenue": df, "net_income": df, "ebitda": df, "total_assets": df,
         "equity": df, "net_debt": df, "shares_outstanding": df}
        Each df has DatetimeIndex (month_end dates) and ticker columns.
        Ratio columns excluded — computed dynamically in shared_data / strategies.
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
#
# History: a 2026-03-07 investigation observed some stored share counts 1000x+
# too large (CIEL 6e12, VSPT 2.1e14) and concluded CVM reports shares in
# thousands, adding a /1000 correction here. The 2026-07-17 FRE parser fix
# found the real root cause: the parser was keeping arbitrary Tipo_Capital rows
# ("Capital Autorizado" = issuance ceiling, or fat-fingered duplicate entries)
# instead of "Capital Integralizado". shares_outstanding is now stored in RAW
# UNITS, so no scaling is applied. Formulas assume: prices in BRL, financial
# metrics in thousands BRL, shares in raw units.

_SHARES_SCALE = 1.0  # shares_outstanding stored in raw units (see history above)


def compute_pe_ratio_dynamic(
    shares: pd.DataFrame,
    net_income: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute P/E ratio from wide DataFrames aligned on the same DatetimeIndex.

    Args:
        shares:     Wide DataFrame: rebalance_date × ticker, shares_outstanding as stored
                    in DB (divide by 1000 to get raw unit count — see _SHARES_SCALE note).
        net_income: Wide DataFrame: rebalance_date × ticker, net_income in thousands BRL.
        prices:     Wide DataFrame: rebalance_date × ticker, close price (BRL).

    Returns:
        Wide DataFrame with same shape: pe_ratio = (prices × shares / SCALE) / (net_income × 1000).
        Entries where net_income × 1000 <= 0 or shares <= 0 or prices <= 0 are NaN.
    """
    market_cap = prices * (shares / _SHARES_SCALE)
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
        shares: Wide DataFrame: shares_outstanding as stored in DB (divide by 1000
                to get raw unit count — see _SHARES_SCALE note).
        equity: Wide DataFrame: equity in thousands BRL.
        prices: Wide DataFrame: close price (BRL).

    Returns:
        Wide DataFrame: pb_ratio = (prices × shares / SCALE) / (equity × 1000).
        Entries where equity × 1000 <= 0 or shares <= 0 or prices <= 0 are NaN.
    """
    market_cap = prices * (shares / _SHARES_SCALE)
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
        shares:   Wide DataFrame: shares_outstanding as stored in DB (divide by 1000
                  to get raw unit count — see _SHARES_SCALE note).
        ebitda:   Wide DataFrame: EBITDA in thousands BRL.
        net_debt: Wide DataFrame: net debt in thousands BRL.
        prices:   Wide DataFrame: close price (BRL).

    Returns:
        Wide DataFrame: ev_ebitda = (prices × shares / SCALE + net_debt × 1000) / (ebitda × 1000).
        Entries where ebitda × 1000 <= 0 or shares <= 0 or prices <= 0 are NaN.
    """
    market_cap = prices * (shares / _SHARES_SCALE)
    ebitda_brl = ebitda * 1_000.0
    net_debt_brl = net_debt * 1_000.0
    ev = market_cap + net_debt_brl
    result = ev / ebitda_brl
    result = result.where(ebitda_brl > 0)
    result = result.where(shares > 0)
    result = result.where(prices > 0)
    return result

"""
Data loader for B3 feature importance study.
Loads B3 stock prices (wide-format), IBOV index, and daily CDI rates.
Reuses existing backtests/core/data.py functions.
"""

import sys
import os
import pickle
import time
from pathlib import Path

import warnings

import pandas as pd

# Add backtests directory to sys.path so we can import from backtests/core/
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtests"))

from core.data import load_b3_hlc_data, download_benchmark, download_cdi_daily, load_all_fundamentals

from research import config


_CACHE_MAX_AGE_SECONDS = 86400  # 24 hours


def _is_cache_fresh(cache_path: Path) -> bool:
    """Return True if cache file exists and is less than 24 hours old."""
    if not cache_path.exists():
        return False
    age = time.time() - cache_path.stat().st_mtime
    return age < _CACHE_MAX_AGE_SECONDS


def _load_ibov() -> pd.Series:
    """Load IBOV from cache or download from Yahoo Finance."""
    cache_path = config.OUTPUT_DIR / ".cache_ibov.pkl"
    if _is_cache_fresh(cache_path):
        print("  [cache] Loading IBOV from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    ibov = download_benchmark(config.IBOV_TICKER, config.START_DATE, config.END_DATE)
    with open(cache_path, "wb") as f:
        pickle.dump(ibov, f)
    return ibov


def _load_cdi() -> pd.Series:
    """Load CDI from cache or download from BCB API."""
    cache_path = config.OUTPUT_DIR / ".cache_cdi.pkl"
    if _is_cache_fresh(cache_path):
        print("  [cache] Loading CDI from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    cdi = download_cdi_daily(config.START_DATE, config.END_DATE)
    with open(cache_path, "wb") as f:
        pickle.dump(cdi, f)
    return cdi


def _load_fundamentals(adj_close_index: pd.DatetimeIndex) -> dict:
    """
    Load fundamentals from fundamentals_pit, forward-fill to daily frequency.

    Returns dict with keys:
        f_revenue, f_net_income, f_ebitda, f_total_assets, f_equity, f_net_debt, f_shares.
    All values are wide DataFrames aligned to adj_close_index (daily).
    Returns empty DataFrames for all keys if no fundamentals data exists.
    """
    cache_path = config.OUTPUT_DIR / ".cache_fundamentals.pkl"
    if _is_cache_fresh(cache_path):
        print("  [cache] Loading fundamentals from cache...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    _EMPTY = {
        "f_revenue": pd.DataFrame(),
        "f_net_income": pd.DataFrame(),
        "f_ebitda": pd.DataFrame(),
        "f_total_assets": pd.DataFrame(),
        "f_equity": pd.DataFrame(),
        "f_net_debt": pd.DataFrame(),
        "f_shares": pd.DataFrame(),
    }

    try:
        raw = load_all_fundamentals(str(config.DB_PATH), config.START_DATE, config.END_DATE, freq="ME")
    except Exception as e:
        print(f"  [fundamentals] No fundamentals data found — skipping. ({e})")
        return _EMPTY

    metric_map = {
        "revenue": "f_revenue",
        "net_income": "f_net_income",
        "ebitda": "f_ebitda",
        "total_assets": "f_total_assets",
        "equity": "f_equity",
        "net_debt": "f_net_debt",
        "shares_outstanding": "f_shares",
    }

    result = {}
    for metric, key in metric_map.items():
        df = raw.get(metric, pd.DataFrame())
        if df is None or df.empty:
            result[key] = pd.DataFrame()
        else:
            try:
                # Reindex to daily adj_close index, forward-fill from monthly snapshots
                extended = df.reindex(df.index.union(adj_close_index)).sort_index().ffill()
                result[key] = extended.reindex(adj_close_index)
            except Exception as e:
                warnings.warn(f"Failed to reindex fundamentals metric '{metric}': {e}")
                result[key] = pd.DataFrame()

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def load_all_data(use_fundamentals: bool = True) -> dict:
    """
    Load all data required for the feature importance study.

    Args:
        use_fundamentals: If True (default), load CVM fundamentals from fundamentals_pit
            and add f_* keys to the returned dict. If False, skip fundamentals loading
            and return empty DataFrames for all f_* keys. Pass False to get a
            technical-signals-only run equivalent to pre-fundamentals behavior.

    Returns dict with keys:
        "adj_close": pd.DataFrame (date x ticker) -- dividend+split adjusted close
        "split_adj_high": pd.DataFrame (date x ticker)
        "split_adj_low": pd.DataFrame (date x ticker)
        "split_adj_close": pd.DataFrame (date x ticker)
        "close_px": pd.DataFrame (date x ticker) -- raw unadjusted close
        "fin_vol": pd.DataFrame (date x ticker) -- financial volume in BRL
        "ibov": pd.Series (date index) -- IBOV close prices
        "cdi_daily": pd.Series (date index) -- daily CDI return fractions
        "f_revenue": pd.DataFrame (date x ticker) -- revenue in thousands BRL, daily ffill
        "f_net_income": pd.DataFrame (date x ticker) -- net income in thousands BRL, daily ffill
        "f_ebitda": pd.DataFrame (date x ticker) -- EBITDA in thousands BRL, daily ffill
        "f_total_assets": pd.DataFrame (date x ticker) -- total assets in thousands BRL, daily ffill
        "f_equity": pd.DataFrame (date x ticker) -- equity in thousands BRL, daily ffill
        "f_net_debt": pd.DataFrame (date x ticker) -- net debt in thousands BRL, daily ffill
        "f_shares": pd.DataFrame (date x ticker) -- shares outstanding (thousands), daily ffill
    All f_* values are empty DataFrames if fundamentals_pit table is absent/empty or
    use_fundamentals=False.
    """
    # Load B3 HLC data from SQLite
    adj_close, split_adj_high, split_adj_low, split_adj_close, close_px, fin_vol = (
        load_b3_hlc_data(str(config.DB_PATH), config.START_DATE, config.END_DATE)
    )

    # Load external data in parallel (I/O-bound downloads)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as pool:
        ibov_future = pool.submit(_load_ibov)
        cdi_future = pool.submit(_load_cdi)
        ibov = ibov_future.result()
        cdi_daily = cdi_future.result()

    # Load fundamentals (local SQLite + cache — not a download, run outside executor)
    if use_fundamentals:
        fundamentals = _load_fundamentals(adj_close.index)
    else:
        fundamentals = {
            "f_revenue": pd.DataFrame(),
            "f_net_income": pd.DataFrame(),
            "f_ebitda": pd.DataFrame(),
            "f_total_assets": pd.DataFrame(),
            "f_equity": pd.DataFrame(),
            "f_net_debt": pd.DataFrame(),
            "f_shares": pd.DataFrame(),
        }

    result = {
        "adj_close": adj_close,
        "split_adj_high": split_adj_high,
        "split_adj_low": split_adj_low,
        "split_adj_close": split_adj_close,
        "close_px": close_px,
        "fin_vol": fin_vol,
        "ibov": ibov,
        "cdi_daily": cdi_daily,
    }
    result.update(fundamentals)
    return result


def print_data_summary(data: dict) -> None:
    """Print shape and date range for each loaded dataset."""
    print("\n  Data Summary:")
    print(f"  {'Key':<20} {'Shape/Length':<25} {'Date Range'}")
    print(f"  {'-'*70}")
    for key, val in data.items():
        if isinstance(val, pd.DataFrame):
            shape_str = f"{val.shape[0]} rows x {val.shape[1]} cols"
            date_min = val.index.min()
            date_max = val.index.max()
            start_str = date_min.date() if pd.notna(date_min) else "N/A"
            end_str = date_max.date() if pd.notna(date_max) else "N/A"
            print(f"  {key:<20} {shape_str:<25} {start_str} to {end_str}")
        elif isinstance(val, pd.Series):
            shape_str = f"{len(val)} rows"
            if len(val) > 0:
                date_min = val.index.min()
                date_max = val.index.max()
                print(f"  {key:<20} {shape_str:<25} {date_min.date()} to {date_max.date()}")
            else:
                print(f"  {key:<20} {shape_str:<25} (empty)")
    print()

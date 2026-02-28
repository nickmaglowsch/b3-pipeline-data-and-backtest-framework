"""
Shared Data Builder
===================
Loads and precomputes all DataFrames needed by strategy plugins.
Mirrors the data loading done in strategy_returns.py lines 464-549,
but is exposed as a standalone function for use by the UI backtest service.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtests.core.data import load_b3_data, download_benchmark, download_cdi_daily


def build_shared_data(
    db_path: str,
    start: str,
    end: str,
    freq: str = "ME",
) -> dict:
    """
    Load and precompute all shared DataFrames needed by strategy plugins.

    Args:
        db_path:  Path to the B3 SQLite database file.
        start:    Start date string, e.g. '2005-01-01'.
        end:      End date string, e.g. '2025-12-31' or 'today'.
        freq:     Resampling frequency (default 'ME' = month-end).

    Returns:
        Dict containing all precomputed DataFrames and Series.
    """
    lookback = 12  # default lookback periods

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    adj_close, close_px, fin_vol = load_b3_data(db_path, start, end)
    cdi_daily = download_cdi_daily(start, end)
    ibov_px = download_benchmark("^BVSP", start, end)

    # ── 2. Resample ───────────────────────────────────────────────────────────
    ibov_ret = ibov_px.resample(freq).last().pct_change().dropna()
    cdi_ret = (1 + cdi_daily).resample(freq).prod() - 1
    cdi_monthly = cdi_ret.copy()

    px = adj_close.resample(freq).last()
    ret = px.pct_change()
    raw_close = close_px.resample(freq).last()
    adtv = fin_vol.resample(freq).mean()

    log_ret = np.log1p(ret)
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(lookback).max()

    # ── 3. Regime signals ─────────────────────────────────────────────────────
    # COPOM easing: CDI was lower 1 month ago vs 4 months ago
    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # MA200 and above-MA200 flag
    ma200_daily = adj_close.rolling(200, min_periods=200).mean()
    ma200_m = ma200_daily.resample(freq).last()
    above_ma200 = px > ma200_m
    dist_ma200 = px / ma200_m - 1

    # IBOV regime signals
    ibov_daily_ret = ibov_px.pct_change()
    ibov_vol_20d = ibov_daily_ret.rolling(20).std()
    ibov_vol_m = ibov_vol_20d.resample(freq).last()
    ibov_vol_pctrank = ibov_vol_m.expanding(min_periods=12).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False
    )
    ibov_calm = ibov_vol_pctrank <= 0.70

    ibov_ret_20d = ibov_px.pct_change(20)
    ibov_ret_m = ibov_ret_20d.resample(freq).last()
    ibov_uptrend = ibov_ret_m > 0

    ibov_m = ibov_px.resample(freq).last()
    ibov_ma10 = ibov_m.rolling(10).mean()
    ibov_above = ibov_m > ibov_ma10

    # Multifactor composite (momentum + low-vol)
    mom_sig = log_ret.shift(1).rolling(lookback).sum()
    mom_sig[has_glitch == 1] = np.nan
    vol_sig_mf = -ret.shift(1).rolling(lookback).std()
    vol_sig_mf[has_glitch == 1] = np.nan
    mf_composite = (
        mom_sig.rank(axis=1, pct=True) * 0.5
        + vol_sig_mf.rank(axis=1, pct=True) * 0.5
    )

    # Additional signals for Res.MultiFactor
    # Note: names reflect monthly-resampled rolling windows, not daily windows.
    # vol_5m = 5-period (month) rolling std, vol_2m = 2-period (month) rolling std.
    vol_5m = ret.rolling(5).std()
    daily_ret_abs = adj_close.pct_change().abs()
    atr_proxy = daily_ret_abs.ewm(span=14, min_periods=14).mean()
    atr_m = atr_proxy.resample(freq).last()
    vol_2m = ret.rolling(2).std()

    # IBOV vol series (monthly, for AdaptiveLowVol regime)
    ibov_vol_monthly = ibov_vol_m.copy()

    return {
        # ── price / return data ───────────────────────────────────────────────
        "adj_close": adj_close,
        "close_px": close_px,
        "fin_vol": fin_vol,
        "px": px,
        "ret": ret,
        "log_ret": log_ret,
        "raw_close": raw_close,
        "adtv": adtv,
        "has_glitch": has_glitch,
        # ── benchmark ─────────────────────────────────────────────────────────
        "ibov_ret": ibov_ret,
        "ibov_px": ibov_px,
        "cdi_monthly": cdi_monthly,
        "cdi_daily": cdi_daily,
        # ── regime signals ────────────────────────────────────────────────────
        "is_easing": is_easing,
        "above_ma200": above_ma200,
        "dist_ma200": dist_ma200,
        "ma200_m": ma200_m,
        "ibov_calm": ibov_calm,
        "ibov_uptrend": ibov_uptrend,
        "ibov_above": ibov_above,
        "ibov_vol_monthly": ibov_vol_monthly,
        "ibov_vol_pctrank": ibov_vol_pctrank,
        # ── composite signals ─────────────────────────────────────────────────
        "mf_composite": mf_composite,
        "vol_5m": vol_5m,
        "vol_60d": vol_5m,   # backward-compat alias
        "atr_m": atr_m,
        "vol_2m": vol_2m,
        "vol_20d": vol_2m,   # backward-compat alias
    }

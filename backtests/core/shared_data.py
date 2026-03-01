"""
Shared Data Builder
===================
Loads and precomputes all DataFrames needed by strategy plugins.
Mirrors the data loading done in strategy_returns.py lines 464-549,
but is exposed as a standalone function for use by the UI backtest service.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from backtests.core.data import load_b3_data, download_benchmark, download_cdi_daily

logger = logging.getLogger(__name__)

# Common split ratios (N:1 forward splits)
_SPLIT_RATIOS = [2, 3, 4, 5, 8, 10]
_SPLIT_TOLERANCE = 0.04  # 4% tolerance for ratio matching


def _detect_and_fix_unrecorded_splits(
    adj_close: pd.DataFrame,
    close_px: pd.DataFrame,
) -> pd.DataFrame:
    """
    Detect stock splits that were not adjusted by the pipeline and correct adj_close.

    When B3's API doesn't return historical split data (common for events before ~2010),
    adj_close == close_px and the split shows as an overnight price drop matching a common
    split ratio (e.g., exactly -50% for a 2:1 split).

    Algorithm:
    1. For each stock, compute daily price ratio (today/yesterday) for both adj_close
       and close_px.
    2. If both series show the same ratio AND that ratio matches a common split pattern
       (1/2, 1/3, 1/4, etc. within tolerance), flag it as an unrecorded split.
    3. Correct adj_close by dividing all pre-split prices by the split ratio.

    This is conservative: only fires when both adjusted and raw prices show identical
    split-like patterns (confirming no adjustment was applied).
    """
    corrected = adj_close.copy()
    total_fixes = 0

    for col in corrected.columns:
        adj_s = corrected[col].dropna()
        raw_s = close_px[col].reindex(adj_s.index).dropna()

        common_idx = adj_s.index.intersection(raw_s.index)
        if len(common_idx) < 2:
            continue

        adj_s = adj_s.loc[common_idx]
        raw_s = raw_s.loc[common_idx]

        # Daily ratio: today / yesterday
        adj_ratio = (adj_s / adj_s.shift(1)).iloc[1:]
        raw_ratio = (raw_s / raw_s.shift(1)).iloc[1:]

        # Find unrecorded split dates
        split_events = []  # list of (date, split_ratio_N)

        for i in range(len(adj_ratio)):
            ar = adj_ratio.iloc[i]
            rr = raw_ratio.iloc[i]

            if pd.isna(ar) or pd.isna(rr) or rr == 0:
                continue

            # Both adj and raw must show nearly identical ratio
            # (confirms no adjustment was applied)
            if abs(ar - rr) / max(abs(rr), 0.001) > 0.05:
                continue

            # Check forward splits: ratio ≈ 1/N
            for n in _SPLIT_RATIOS:
                target = 1.0 / n
                if abs(ar - target) / target < _SPLIT_TOLERANCE:
                    split_events.append((adj_ratio.index[i], n, "forward"))
                    break
                # Check reverse splits: ratio ≈ N
                if abs(ar - float(n)) / n < _SPLIT_TOLERANCE:
                    split_events.append((adj_ratio.index[i], n, "reverse"))
                    break

        if not split_events:
            continue

        # Apply corrections: for a forward N:1 split at date D,
        # all prices BEFORE D need to be divided by N (backward adjustment)
        for dt, n, direction in split_events:
            mask = corrected.index < dt
            if direction == "forward":
                corrected.loc[mask, col] /= n
            else:  # reverse split
                corrected.loc[mask, col] *= n

        total_fixes += len(split_events)
        logger.info(
            f"Fixed {len(split_events)} unrecorded splits for {col}: "
            + ", ".join(f"{d.date()} {dir} {n}:1" for d, n, dir in split_events)
        )

    if total_fixes > 0:
        logger.info(f"Total unrecorded splits fixed: {total_fixes}")
    return corrected


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

    # ── 1b. Fix unrecorded splits ────────────────────────────────────────────
    adj_close = _detect_and_fix_unrecorded_splits(adj_close, close_px)

    # ── 2. Resample ───────────────────────────────────────────────────────────
    ibov_ret = ibov_px.resample(freq).last().pct_change().dropna()
    cdi_ret = (1 + cdi_daily).resample(freq).prod() - 1
    cdi_monthly = cdi_ret.copy()

    px = adj_close.resample(freq).last()
    ret = px.pct_change()
    raw_close = close_px.resample(freq).last()
    adtv = fin_vol.resample(freq).mean()

    log_ret = np.log1p(ret)
    # Threshold -0.45 catches 2:1 splits (-50%) and 3:1 splits (-67%)
    # that slip past the heuristic detector
    has_glitch = ((ret > 1.0) | (ret < -0.45)).rolling(lookback).max()

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

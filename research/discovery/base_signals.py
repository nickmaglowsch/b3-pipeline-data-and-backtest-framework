"""
Base signal library for feature discovery.
Implements ~15 signal categories with parametric sweep.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.discovery import config


def compute_momentum_return(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Momentum returns: price change over window."""
    return adj_close.pct_change(window)


def compute_distance_to_ma(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Distance to moving average: price / MA - 1."""
    ma = adj_close.rolling(window, min_periods=max(1, window // 2)).mean()
    return adj_close / ma - 1


def compute_rolling_vol(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling volatility (shifted to avoid lookahead)."""
    daily_ret = adj_close.pct_change(1)
    return daily_ret.rolling(window, min_periods=max(5, window // 2)).std().shift(1)


def compute_atr(
    split_adj_high: pd.DataFrame,
    split_adj_low: pd.DataFrame,
    split_adj_close: pd.DataFrame,
    span: int,
) -> pd.DataFrame:
    """ATR normalized: exponential moving average of true range / close."""
    prev_close = split_adj_close.shift(1)
    true_range = np.maximum(
        np.maximum(
            split_adj_high - split_adj_low,
            (split_adj_high - prev_close).abs(),
        ),
        (split_adj_low - prev_close).abs(),
    )
    atr = true_range.ewm(span=span, min_periods=span).mean()
    return atr / split_adj_close


def compute_drawdown(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Drawdown: current price / max price - 1."""
    running_max = adj_close.rolling(window, min_periods=max(5, window // 2)).max()
    return adj_close / running_max - 1


def compute_volume_zscore(fin_vol: pd.DataFrame, window: int) -> pd.DataFrame:
    """Volume z-score."""
    mean = fin_vol.rolling(window, min_periods=max(5, window // 2)).mean()
    std = fin_vol.rolling(window, min_periods=max(5, window // 2)).std()
    return (fin_vol - mean) / std


def compute_volume_ratio(
    fin_vol: pd.DataFrame, short: int, long: int
) -> pd.DataFrame:
    """Short-term / long-term average volume ratio."""
    short_ma = fin_vol.rolling(short, min_periods=max(2, short // 2)).mean()
    long_ma = fin_vol.rolling(long, min_periods=max(5, long // 2)).mean()
    return short_ma / long_ma


def compute_beta(
    adj_close: pd.DataFrame, ibov: pd.Series, window: int
) -> pd.DataFrame:
    """Rolling beta vs IBOV."""
    daily_ret = adj_close.pct_change(1)
    ibov_ret = ibov.pct_change(1).reindex(daily_ret.index)
    min_periods = int(window * 0.6)
    
    rolling_cov = daily_ret.rolling(window, min_periods=min_periods).cov(ibov_ret)
    ibov_var = ibov_ret.rolling(window, min_periods=min_periods).var()
    return rolling_cov.div(ibov_var, axis=0)


def compute_skewness(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling skewness of daily returns (shifted)."""
    daily_ret = adj_close.pct_change(1)
    return daily_ret.rolling(window, min_periods=max(10, window // 2)).skew().shift(1)


def compute_kurtosis(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling kurtosis of daily returns (shifted)."""
    daily_ret = adj_close.pct_change(1)
    return daily_ret.rolling(window, min_periods=max(10, window // 2)).kurt().shift(1)


def compute_max_return(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Maximum daily return over window (shifted)."""
    daily_ret = adj_close.pct_change(1)
    return daily_ret.rolling(window, min_periods=max(5, window // 2)).max().shift(1)


def compute_min_return(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Minimum daily return over window (shifted)."""
    daily_ret = adj_close.pct_change(1)
    return daily_ret.rolling(window, min_periods=max(5, window // 2)).min().shift(1)


def compute_win_rate(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Fraction of positive daily returns (shifted)."""
    daily_ret = adj_close.pct_change(1)
    positive = (daily_ret > 0).astype(float)
    return positive.rolling(window, min_periods=max(5, window // 2)).mean().shift(1)


def compute_amihud(
    adj_close: pd.DataFrame, fin_vol: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Amihud illiquidity: |return| / volume."""
    daily_ret = adj_close.pct_change(1).abs()
    ratio = daily_ret / fin_vol.replace(0, np.nan)
    return ratio.rolling(window, min_periods=max(5, window // 2)).mean().shift(1)


def compute_autocorr(adj_close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return autocorrelation at lag 1 (shifted). Vectorized via rolling cov/var."""
    daily_ret = adj_close.pct_change(1)
    lagged_ret = daily_ret.shift(1)
    min_p = max(10, window // 2)
    rolling_cov = daily_ret.rolling(window, min_periods=min_p).cov(lagged_ret)
    rolling_var = daily_ret.rolling(window, min_periods=min_p).var()
    return (rolling_cov / rolling_var.replace(0, np.nan)).shift(1)


def compute_high_low_range(
    split_adj_high: pd.DataFrame,
    split_adj_low: pd.DataFrame,
    split_adj_close: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Average high-low range normalized by close (shifted)."""
    daily_range = (split_adj_high - split_adj_low) / split_adj_close
    return daily_range.rolling(window, min_periods=max(3, window // 2)).mean().shift(1)


def compute_ibov_return(ibov: pd.Series, window: int) -> pd.Series:
    """IBOV return over window."""
    return ibov.pct_change(window)


def compute_ibov_vol(ibov: pd.Series, window: int) -> pd.Series:
    """IBOV volatility (shifted)."""
    return ibov.pct_change(1).rolling(window).std().shift(1)


def compute_cdi_cumulative(cdi_daily: pd.Series, window: int) -> pd.Series:
    """CDI cumulative return over window."""
    cumprod = (1 + cdi_daily).cumprod()
    cumul = cumprod / cumprod.shift(window) - 1
    valid_count = cdi_daily.expanding().count()
    cumul[valid_count < max(20, window * 0.6)] = np.nan
    return cumul


def generate_all_base_signals(data: dict):
    """
    Generate all base signals via parametric sweep.
    
    Yields (feature_id, category, params, wide_df) for each signal.
    Memory-efficient: yields one at a time, does not hold all in memory.
    
    Args:
        data: dict from load_all_data() with keys:
              adj_close, split_adj_high, split_adj_low, split_adj_close,
              fin_vol, ibov, cdi_daily
    """
    adj_close = data["adj_close"]
    split_adj_high = data.get("split_adj_high")
    split_adj_low = data.get("split_adj_low")
    split_adj_close = data.get("split_adj_close")
    fin_vol = data.get("fin_vol")
    ibov = data.get("ibov")
    cdi_daily = data.get("cdi_daily")

    # 1. Momentum returns
    for w in config.MOMENTUM_WINDOWS:
        yield (f"Return_{w}d", "momentum", {"window": w},
               compute_momentum_return(adj_close, w))

    # 2. Distance to MA
    for w in config.MA_WINDOWS:
        yield (f"Distance_to_MA{w}", "momentum", {"window": w},
               compute_distance_to_ma(adj_close, w))

    # 3. Rolling volatility
    for w in config.VOLATILITY_WINDOWS:
        yield (f"Rolling_vol_{w}d", "volatility", {"window": w},
               compute_rolling_vol(adj_close, w))

    # 4. ATR
    if split_adj_high is not None and split_adj_low is not None:
        for s in [7, 14, 28]:
            yield (f"ATR_{s}", "volatility", {"span": s},
                   compute_atr(split_adj_high, split_adj_low, split_adj_close, s))

    # 5. Drawdown
    for w in [20, 60, 120]:
        yield (f"Drawdown_{w}d", "momentum", {"window": w},
               compute_drawdown(adj_close, w))

    # 6. Volume z-score
    if fin_vol is not None:
        for w in config.VOLUME_WINDOWS:
            yield (f"Volume_zscore_{w}d", "volume", {"window": w},
                   compute_volume_zscore(fin_vol, w))

        # 7. Volume ratio
        for s, l in [(5, 20), (5, 60), (10, 40), (20, 60)]:
            yield (f"Volume_ratio_{s}d_{l}d", "volume", {"short": s, "long": l},
                   compute_volume_ratio(fin_vol, s, l))

        # 13. Amihud
        for w in config.AMIHUD_WINDOWS:
            yield (f"Amihud_{w}d", "liquidity", {"window": w},
                   compute_amihud(adj_close, fin_vol, w))

    # 8. Beta
    if ibov is not None:
        for w in config.BETA_WINDOWS:
            yield (f"Beta_{w}d", "beta", {"window": w},
                   compute_beta(adj_close, ibov, w))

    # 9. Skewness
    for w in config.SKEW_KURT_WINDOWS:
        yield (f"Skewness_{w}d", "distribution", {"window": w},
               compute_skewness(adj_close, w))

    # 10. Kurtosis
    for w in config.SKEW_KURT_WINDOWS:
        yield (f"Kurtosis_{w}d", "distribution", {"window": w},
               compute_kurtosis(adj_close, w))

    # 11. Max/Min return
    for w in config.MAX_MIN_RET_WINDOWS:
        yield (f"Max_return_{w}d", "extremes", {"window": w},
               compute_max_return(adj_close, w))
        yield (f"Min_return_{w}d", "extremes", {"window": w},
               compute_min_return(adj_close, w))

    # 12. Win rate
    for w in config.WIN_RATE_WINDOWS:
        yield (f"Win_rate_{w}d", "distribution", {"window": w},
               compute_win_rate(adj_close, w))

    # 14. Autocorrelation
    for w in config.AUTOCORR_WINDOWS:
        yield (f"Autocorr_{w}d", "autocorr", {"window": w},
               compute_autocorr(adj_close, w))

    # 15. High-low range
    if split_adj_high is not None and split_adj_low is not None:
        for w in config.HIGH_LOW_RANGE_WINDOWS:
            yield (f"High_low_range_{w}d", "volatility", {"window": w},
                   compute_high_low_range(split_adj_high, split_adj_low, split_adj_close, w))

    # 16. IBOV returns
    if ibov is not None:
        for w in config.IBOV_WINDOWS:
            yield (f"Ibov_return_{w}d", "market_ibov", {"window": w},
                   compute_ibov_return(ibov, w))

        # 17. IBOV volatility
        for w in config.IBOV_WINDOWS:
            yield (f"Ibov_vol_{w}d", "market_ibov", {"window": w},
                   compute_ibov_vol(ibov, w))

    # 18. CDI cumulative
    if cdi_daily is not None:
        for w in config.CDI_WINDOWS:
            yield (f"CDI_cumulative_{w}d", "market_cdi", {"window": w},
                   compute_cdi_cumulative(cdi_daily, w))

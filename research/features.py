"""
Feature engineering for B3 feature importance study.
All features are computed in wide format (date x ticker) then stacked.
"""

import numpy as np
import pandas as pd
from research import config


FEATURE_NAMES = [
    "Return_1d", "Return_5d", "Return_20d", "Return_60d",
    "Distance_to_MA20", "Distance_to_MA50", "Distance_to_MA200",
    "Rolling_vol_20d", "Rolling_vol_60d", "ATR_14", "Drawdown_60d",
    "Volume_zscore_20d", "Volume_ratio_5d_20d",
    "Rank_momentum_60d", "Rank_volatility_20d", "Rank_volume",
    "Ibovespa_return_20d", "Ibovespa_vol_20d", "CDI_3m_change",
]


def compute_price_features(adj_close: pd.DataFrame) -> dict:
    """
    Compute price-based features from adj_close (wide format).

    Returns dict of {feature_name: pd.DataFrame} where each DataFrame
    is date x ticker, same shape as adj_close.
    """
    daily_ret = adj_close.pct_change(1)

    features = {
        "Return_1d": adj_close.pct_change(1),
        "Return_5d": adj_close.pct_change(5),
        "Return_20d": adj_close.pct_change(20),
        "Return_60d": adj_close.pct_change(60),
        "Distance_to_MA20": adj_close / adj_close.rolling(20).mean() - 1,
        "Distance_to_MA50": adj_close / adj_close.rolling(50).mean() - 1,
        "Distance_to_MA200": adj_close / adj_close.rolling(200).mean() - 1,
        "Rolling_vol_20d": daily_ret.rolling(20).std(),
        "Rolling_vol_60d": daily_ret.rolling(60).std(),
        "Drawdown_60d": adj_close / adj_close.rolling(60).max() - 1,
    }
    return features


def compute_atr_feature(split_adj_high, split_adj_low, split_adj_close) -> pd.DataFrame:
    """
    Compute normalized ATR_14 from split-adjusted HLC.

    ATR = true_range.ewm(span=14).mean()
    Normalized: ATR / split_adj_close (to make it scale-invariant)

    Returns date x ticker DataFrame.
    """
    prev_close = split_adj_close.shift(1)
    tr1 = split_adj_high - split_adj_low
    tr2 = (split_adj_high - prev_close).abs()
    tr3 = (split_adj_low - prev_close).abs()

    # Element-wise max (avoids concat+groupby which triples memory)
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)

    atr = true_range.ewm(span=config.ATR_SPAN, min_periods=config.ATR_SPAN).mean()

    # Normalize by price to make scale-invariant
    atr_normalized = atr / split_adj_close
    return atr_normalized


def compute_volume_features(fin_vol: pd.DataFrame) -> dict:
    """
    Compute volume-based features from financial volume.

    Returns dict of {feature_name: pd.DataFrame}.
    """
    vol_mean_20 = fin_vol.rolling(config.VOLUME_ZSCORE_WINDOW, min_periods=10).mean()
    vol_std_20 = fin_vol.rolling(config.VOLUME_ZSCORE_WINDOW, min_periods=10).std()
    volume_zscore_20d = (fin_vol - vol_mean_20) / vol_std_20

    vol_ma_5 = fin_vol.rolling(config.VOLUME_RATIO_SHORT, min_periods=3).mean()
    vol_ma_20 = fin_vol.rolling(config.VOLUME_RATIO_LONG, min_periods=10).mean()
    volume_ratio_5d_20d = vol_ma_5 / vol_ma_20

    return {
        "Volume_zscore_20d": volume_zscore_20d,
        "Volume_ratio_5d_20d": volume_ratio_5d_20d,
    }


def compute_cross_sectional_features(
    return_60d: pd.DataFrame,
    rolling_vol_20d: pd.DataFrame,
    fin_vol: pd.DataFrame,
) -> dict:
    """
    Compute cross-sectional rank features.

    Each rank is computed PER DATE (axis=1), producing a percentile rank
    among all stocks on that date. This is naturally backward-looking
    since it uses only today's cross-section.

    Returns dict of {feature_name: pd.DataFrame}.
    """
    rank_momentum_60d = return_60d.rank(axis=1, pct=True)
    rank_volatility_20d = rolling_vol_20d.rank(axis=1, pct=True)
    rank_volume = fin_vol.rolling(20, min_periods=10).mean().rank(axis=1, pct=True)

    return {
        "Rank_momentum_60d": rank_momentum_60d,
        "Rank_volatility_20d": rank_volatility_20d,
        "Rank_volume": rank_volume,
    }


def compute_market_features(ibov: pd.Series, cdi_daily: pd.Series) -> dict:
    """
    Compute market regime features from IBOV and CDI.

    These are scalar (same value for all stocks on a given date).
    Returns dict of {feature_name: pd.Series}.
    """
    ibov_daily_ret = ibov.pct_change()
    ibovespa_return_20d = ibov.pct_change(config.IBOV_WINDOW)
    ibovespa_vol_20d = ibov_daily_ret.rolling(config.IBOV_WINDOW).std()

    # CDI cumulative 3-month return (vectorized via cumprod + shift, avoids slow .apply)
    window = config.CDI_CUMULATIVE_WINDOW
    cdi_cumprod = (1 + cdi_daily).cumprod()
    cdi_cumulative = cdi_cumprod / cdi_cumprod.shift(window) - 1
    # Respect min_periods=40: mask the first 39 valid entries
    valid_count = cdi_daily.expanding().count()
    cdi_cumulative[valid_count < 40] = np.nan

    return {
        "Ibovespa_return_20d": ibovespa_return_20d,
        "Ibovespa_vol_20d": ibovespa_vol_20d,
        "CDI_3m_change": cdi_cumulative,
    }


def compute_universe_mask(
    close_px: pd.DataFrame,
    fin_vol: pd.DataFrame,
    adj_close: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute boolean mask (date x ticker) where True = stock is in universe.

    Criteria:
    1. 20-day rolling average financial volume >= R$1M
    2. Raw close price >= R$1.00
    3. Stock has at least 200 days of non-NaN adj_close history
    """
    adtv_20 = fin_vol.rolling(config.ADTV_WINDOW, min_periods=10).mean()
    liquidity_ok = adtv_20 >= config.MIN_ADTV
    price_ok = close_px >= config.MIN_PRICE

    # Count non-NaN days of adj_close history per ticker, expanding
    history_count = adj_close.expanding(min_periods=1).count()
    history_ok = history_count >= config.MIN_HISTORY_DAYS

    universe_mask = liquidity_ok & price_ok & history_ok
    return universe_mask


def build_feature_matrix(data: dict) -> pd.DataFrame:
    """
    Master function: compute all features and assemble into long-format DataFrame.

    Args:
        data: dict from data_loader.load_all_data()

    Returns:
        DataFrame with columns: ['date', 'ticker', <19 feature columns>]
        Only rows where the stock is in the liquid universe are included.
        Rows with any NaN feature are dropped.
    """
    adj_close = data["adj_close"]
    split_adj_high = data["split_adj_high"]
    split_adj_low = data["split_adj_low"]
    split_adj_close = data["split_adj_close"]
    close_px = data["close_px"]
    fin_vol = data["fin_vol"]
    ibov = data["ibov"]
    cdi_daily = data["cdi_daily"]

    print("  Computing price features...")
    price_features = compute_price_features(adj_close)

    print("  Computing ATR feature...")
    atr_feature = compute_atr_feature(split_adj_high, split_adj_low, split_adj_close)

    print("  Computing volume features...")
    volume_features = compute_volume_features(fin_vol)

    print("  Computing cross-sectional features...")
    cross_features = compute_cross_sectional_features(
        price_features["Return_60d"],
        price_features["Rolling_vol_20d"],
        fin_vol,
    )

    print("  Computing market regime features...")
    market_features = compute_market_features(ibov, cdi_daily)

    # Combine stock-level feature DataFrames (date x ticker)
    stock_features = {}
    stock_features.update(price_features)
    stock_features["ATR_14"] = atr_feature
    stock_features.update(volume_features)
    stock_features.update(cross_features)

    # Compute universe mask
    print("  Computing universe mask...")
    universe_mask = compute_universe_mask(close_px, fin_vol, adj_close)

    # Apply universe mask and stack to long format in one pass (avoids 16 .copy() calls)
    print("  Applying universe mask and stacking to long format...")
    stock_feat_names = [
        "Return_1d", "Return_5d", "Return_20d", "Return_60d",
        "Distance_to_MA20", "Distance_to_MA50", "Distance_to_MA200",
        "Rolling_vol_20d", "Rolling_vol_60d", "ATR_14", "Drawdown_60d",
        "Volume_zscore_20d", "Volume_ratio_5d_20d",
        "Rank_momentum_60d", "Rank_volatility_20d", "Rank_volume",
    ]
    stacked_parts = []
    for feat_name in stock_feat_names:
        # .where() returns NaN where mask is False -- no copy needed
        s = stock_features[feat_name].where(universe_mask).stack().astype("float32")
        s.name = feat_name
        s.index.names = ["date", "ticker"]
        stacked_parts.append(s)
        # Free the wide-format feature immediately
        del stock_features[feat_name]

    # Build long-format DataFrame by concatenating all stacked series
    long_df = pd.concat(stacked_parts, axis=1)
    del stacked_parts  # free intermediate list
    long_df = long_df.reset_index()

    # Join market regime features (Series indexed by date) onto the long DataFrame
    market_df = pd.DataFrame(market_features)
    market_df.index.name = "date"
    market_df = market_df.reset_index()

    long_df = long_df.merge(market_df, on="date", how="left")

    # Drop rows with any NaN feature
    feature_cols = FEATURE_NAMES
    long_df = long_df.dropna(subset=feature_cols)

    # Ensure date column is datetime
    long_df["date"] = pd.to_datetime(long_df["date"])

    # Cast market feature columns to float32 (stock features already float32 from stacking)
    for col in ["Ibovespa_return_20d", "Ibovespa_vol_20d", "CDI_3m_change"]:
        if col in long_df.columns:
            long_df[col] = long_df[col].astype("float32")

    print(f"  Feature matrix built: {long_df.shape[0]:,} rows, {long_df.shape[1]} columns")
    return long_df

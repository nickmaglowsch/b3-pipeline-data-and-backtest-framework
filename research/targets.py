"""
Target variable computation for B3 feature importance study.
Computes forward returns and binary classification targets.
"""

import numpy as np
import pandas as pd
from research import config


def compute_forward_returns(adj_close: pd.DataFrame, periods: int) -> pd.DataFrame:
    """
    Compute forward N-day returns in wide format.

    forward_return(t) = adj_close(t + periods) / adj_close(t) - 1

    This uses shift(-periods) which looks FORWARD in time.
    The last `periods` rows will be NaN (no future data available).

    Args:
        adj_close: wide DataFrame (date x ticker), forward-filled
        periods: number of trading days to look forward

    Returns:
        wide DataFrame of forward returns, same shape as adj_close
    """
    return adj_close.shift(-periods) / adj_close - 1


def compute_binary_target(forward_returns: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Convert forward returns to binary target.
    1 if forward_return > threshold, 0 otherwise.
    NaN where forward_return is NaN.
    """
    target = (forward_returns > threshold).astype(float)
    target[forward_returns.isna()] = np.nan
    return target


def compute_median_target(forward_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Convert forward returns to binary target relative to cross-sectional median.

    For each date, compute the median forward return across all stocks,
    then target = 1 if stock's return > that date's median, else 0.

    This avoids lookahead because the median is computed per date
    (using only stocks available on that date), and the forward return
    itself is the target (not a feature).
    """
    # Compute median per date (across tickers)
    date_median = forward_returns.median(axis=1)
    # Broadcast: subtract median for each date row
    excess = forward_returns.sub(date_median, axis=0)
    target = (excess > 0).astype(float)
    target[forward_returns.isna()] = np.nan
    return target


def add_targets_to_feature_matrix(
    feature_matrix: pd.DataFrame,
    adj_close: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add all three target columns to the feature matrix.

    The feature matrix has columns ['date', 'ticker', ...features...].
    This function:
    1. Computes forward 20d and 60d returns in wide format
    2. Converts to binary targets
    3. Stacks to long format and merges onto feature matrix by (date, ticker)
    4. Drops rows where primary target (target_20d) is NaN

    Returns:
        DataFrame with original feature columns plus:
        - 'target_20d': primary binary target (forward 20d return > 0)
        - 'target_60d': robustness target A (forward 60d return > 0)
        - 'target_20d_median': robustness target B (forward 20d return > date median)
        - 'forward_return_20d': raw forward 20d return (for diagnostics)
    """
    # 1. Compute forward returns in wide format
    fwd_20d = compute_forward_returns(adj_close, config.FORWARD_PERIOD_20D)
    fwd_60d = compute_forward_returns(adj_close, config.FORWARD_PERIOD_60D)

    # 2. Compute binary targets in wide format
    target_20d = compute_binary_target(fwd_20d)
    target_60d = compute_binary_target(fwd_60d)
    target_20d_median = compute_median_target(fwd_20d)

    # 3. Stack all targets in one pass using concat (avoids 4 separate stack+merge)
    target_wide = {
        "target_20d": target_20d,
        "target_60d": target_60d,
        "target_20d_median": target_20d_median,
        "forward_return_20d": fwd_20d,
    }
    stacked = []
    for name, df in target_wide.items():
        s = df.stack()
        s.name = name
        s.index.names = ["date", "ticker"]
        stacked.append(s)
    targets_long = pd.concat(stacked, axis=1).reset_index()
    del stacked, target_wide, target_20d, target_60d, target_20d_median, fwd_20d, fwd_60d

    # 4. Single merge onto feature matrix
    result = feature_matrix.merge(targets_long, on=["date", "ticker"], how="left")
    del targets_long

    # 5. Drop rows where primary target is NaN (no future data)
    result = result.dropna(subset=["target_20d"])

    return result


def get_dataset_for_target(
    full_df: pd.DataFrame,
    target_col: str,
    feature_names: list,
) -> tuple:
    """
    Extract X (features) and y (target) for a specific target column.
    Drops rows where target is NaN (relevant for target_60d which has
    more NaN rows at the end than target_20d).

    Returns:
        (X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame)
        where meta contains 'date' and 'ticker' for each row
    """
    df = full_df.dropna(subset=[target_col] + feature_names).copy()
    X = df[feature_names]
    y = df[target_col].astype(int)
    meta = df[["date", "ticker"]]
    return X, y, meta

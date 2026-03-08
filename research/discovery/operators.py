"""
Operators for feature synthesis.
Each operator takes wide-format DataFrames and returns wide-format DataFrames.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import cotahist_rs as _rs
    _RUST_OPS = True
except ImportError:
    _RUST_OPS = False


def _wide_to_batch(df: pd.DataFrame):
    """Convert a wide DataFrame (DatetimeIndex × tickers) to a pyarrow RecordBatch."""
    import pyarrow as pa
    tmp = df.reset_index()
    # The reset_index column name may be None or the index name; rename to "date"
    tmp = tmp.rename(columns={tmp.columns[0]: "date"})
    tmp["date"] = tmp["date"].astype(str)
    return pa.RecordBatch.from_pandas(tmp, preserve_index=False)


def _batch_to_wide(batch, original_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a RecordBatch back to a wide DataFrame, restoring the original index."""
    result = batch.to_pandas().set_index("date")
    result.index = pd.to_datetime(result.index)
    return result.reindex(index=original_df.index, columns=original_df.columns)


# ── Unary operators ──────────────────────────────────────────────────────────


def op_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional percentile rank per date (axis=1)."""
    if _RUST_OPS and df.shape[0] * df.shape[1] > 10_000:
        batch = _wide_to_batch(df)
        result_batch = _rs.cross_sectional_rank(batch)
        return _batch_to_wide(result_batch, df)
    return df.rank(axis=1, pct=True)


def op_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per date (axis=1)."""
    if _RUST_OPS and df.shape[0] * df.shape[1] > 10_000:
        batch = _wide_to_batch(df)
        result_batch = _rs.cross_sectional_zscore(batch)
        return _batch_to_wide(result_batch, df)
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    # Broadcast: subtract row mean, divide by row std
    result = df.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
    return result


def op_delta(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Change from N days ago: signal(t) - signal(t-period)."""
    return df - df.shift(period)


def op_ratio_to_mean(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """Signal / rolling mean of signal."""
    rolling_mean = df.rolling(period, min_periods=max(3, period // 2)).mean()
    return df / rolling_mean.replace(0, np.nan)


# ── Binary operators ─────────────────────────────────────────────────────────


def op_ratio(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """A / B. Replace zero B with NaN to avoid inf."""
    return a / b.replace(0, np.nan)


def op_product(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """A * B (interaction term)."""
    return a * b


def op_diff(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    """A - B."""
    return a - b


# ── Operator registry ────────────────────────────────────────────────────────

UNARY_OPS = {
    "rank": op_rank,
    "zscore": op_zscore,
}

BINARY_OPS = {
    "ratio": op_ratio,
    "product": op_product,
}

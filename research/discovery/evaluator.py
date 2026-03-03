"""
IC-based feature evaluator.
Computes Information Coefficient and related metrics for each feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from research.discovery import config
from research.discovery.store import FeatureStore


def compute_forward_returns_wide(
    adj_close: pd.DataFrame, horizons: list[int]
) -> dict[int, pd.DataFrame]:
    """
    Compute forward N-day returns for multiple horizons.
    Returns dict of {horizon: wide DataFrame}.
    """
    result = {}
    for h in horizons:
        result[h] = adj_close.shift(-h) / adj_close - 1
    return result


def compute_ic_series_fast(
    feature_wide: pd.DataFrame,
    forward_return_wide: pd.DataFrame,
    universe_mask: pd.DataFrame,
    fwd_rank_precomputed: pd.DataFrame = None,
) -> pd.Series:
    """
    Vectorized IC computation using rank correlation.

    Spearman correlation = Pearson correlation of ranks.
    We rank across columns (stocks) per row (date), then compute
    row-wise Pearson correlation between feature ranks and return ranks.

    Args:
        feature_wide: wide DataFrame (date x ticker)
        forward_return_wide: wide DataFrame (date x ticker) - can be None if fwd_rank_precomputed given
        universe_mask: boolean DataFrame (date x ticker)
        fwd_rank_precomputed: pre-ranked forward returns (avoids re-ranking for every feature)

    Returns:
        pd.Series of IC values, indexed by date. NaN for dates with < 10 valid stocks.
    """
    # Apply universe mask and rank feature
    feat = feature_wide.where(universe_mask)
    feat_rank = feat.rank(axis=1, pct=True)

    # Use precomputed forward return ranks if available
    if fwd_rank_precomputed is not None:
        fwd_rank = fwd_rank_precomputed
    else:
        fwd = forward_return_wide.where(universe_mask)
        fwd_rank = fwd.rank(axis=1, pct=True)

    # Mask both to only include positions where both have valid data
    both_valid = feat_rank.notna() & fwd_rank.notna()
    n = both_valid.sum(axis=1)

    feat_masked = feat_rank.where(both_valid)
    fwd_masked = fwd_rank.where(both_valid)

    # Demean ranks for correlation calculation
    feat_dm = feat_masked.sub(feat_masked.mean(axis=1), axis=0).fillna(0)
    fwd_dm = fwd_masked.sub(fwd_masked.mean(axis=1), axis=0).fillna(0)

    # Cross product per row
    numerator = (feat_dm * fwd_dm).sum(axis=1)
    denom = (feat_dm**2).sum(axis=1).pow(0.5) * (fwd_dm**2).sum(axis=1).pow(0.5)

    ic = numerator / denom.replace(0, np.nan)

    # Mask dates with too few stocks
    ic[n < 10] = np.nan

    return ic


def compute_evaluation_summary(
    ic_series: pd.Series, train_cutoff_date
) -> dict:
    """
    Compute summary stats from an IC time series.

    Returns dict with metrics including train/test split.
    """
    ic = ic_series.dropna()
    n = len(ic)

    # Guard: empty IC series
    if n == 0:
        return {
            "mean_ic": 0.0, "ic_std": 0.0, "ic_ir": 0.0, "ic_t_stat": 0.0,
            "pct_positive_ic": 0.0, "mean_ic_5y": None, "n_dates": 0,
            "mean_ic_train": None, "ic_ir_train": 0.0,
            "mean_ic_test": None, "ic_ir_test": 0.0,
        }

    mean_ic = ic.mean()
    ic_std = ic.std()
    ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
    ic_t_stat = mean_ic / (ic_std / np.sqrt(n)) if ic_std > 0 else 0.0
    pct_positive = (ic > 0).mean()

    # Recency metric: mean IC over last 5 years
    recency_cutoff = ic.index.max() - pd.DateOffset(years=config.IC_RECENCY_YEARS)
    recent_ic = ic[ic.index >= recency_cutoff]
    mean_ic_5y = recent_ic.mean() if len(recent_ic) > 0 else np.nan

    # Train/test split
    train = ic[ic.index <= train_cutoff_date]
    test = ic[ic.index > train_cutoff_date]

    mean_ic_train = train.mean() if len(train) > 0 else np.nan
    ic_std_train = train.std() if len(train) > 0 else np.nan
    ic_ir_train = mean_ic_train / ic_std_train if pd.notna(ic_std_train) and ic_std_train > 0 else 0.0

    mean_ic_test = test.mean() if len(test) > 0 else np.nan
    ic_std_test = test.std() if len(test) > 0 else np.nan
    ic_ir_test = mean_ic_test / ic_std_test if pd.notna(ic_std_test) and ic_std_test > 0 else 0.0

    return {
        "mean_ic": round(float(mean_ic), 6),
        "ic_std": round(float(ic_std), 6),
        "ic_ir": round(float(ic_ir), 4),
        "ic_t_stat": round(float(ic_t_stat), 2),
        "pct_positive_ic": round(float(pct_positive), 4),
        "mean_ic_5y": round(float(mean_ic_5y), 6) if not np.isnan(mean_ic_5y) else None,
        "n_dates": int(n),
        "mean_ic_train": round(float(mean_ic_train), 6) if not np.isnan(mean_ic_train) else None,
        "ic_ir_train": round(float(ic_ir_train), 4),
        "mean_ic_test": round(float(mean_ic_test), 6) if not np.isnan(mean_ic_test) else None,
        "ic_ir_test": round(float(ic_ir_test), 4),
    }


def compute_decay(
    feature_wide: pd.DataFrame,
    forward_return_wide: pd.DataFrame,
    universe_mask: pd.DataFrame,
    lags: list[int],
    fwd_rank_precomputed: pd.DataFrame = None,
) -> dict[int, float]:
    """
    Compute mean IC when feature is lagged by N days.
    Returns dict of {lag: mean_ic}.
    """
    result = {}
    for lag in lags:
        lagged = feature_wide.shift(lag)
        ic = compute_ic_series_fast(lagged, forward_return_wide, universe_mask, fwd_rank_precomputed)
        result[lag] = round(float(ic.dropna().mean()), 6)
    return result


def compute_turnover(feature_wide: pd.DataFrame, universe_mask: pd.DataFrame) -> float:
    """
    Compute average day-to-day rank correlation of the feature.
    High turnover = feature values change a lot day-to-day = more trading.

    turnover = 1 - mean(spearman_corr(rank(t), rank(t-1)))
    """
    feat = feature_wide.where(universe_mask)
    ranks = feat.rank(axis=1, pct=True)

    # Compute row-wise correlation between consecutive dates
    ranks_shifted = ranks.shift(1)

    # Vectorized correlation
    r = ranks.fillna(0)
    rs = ranks_shifted.fillna(0)
    r_dm = r.sub(r.mean(axis=1), axis=0)
    rs_dm = rs.sub(rs.mean(axis=1), axis=0)

    num = (r_dm * rs_dm).sum(axis=1)
    den = (r_dm**2).sum(axis=1).pow(0.5) * (rs_dm**2).sum(axis=1).pow(0.5)

    daily_corr = num / den.replace(0, np.nan)

    # Turnover = 1 - average auto-correlation of ranks
    mean_autocorr = daily_corr.dropna().mean()
    return round(float(1.0 - mean_autocorr), 4)


def evaluate_feature(
    feature_id: str,
    feature_wide: pd.DataFrame,
    forward_returns: dict[int, pd.DataFrame],
    universe_mask: pd.DataFrame,
    train_cutoff_date,
    fwd_ranks_precomputed: dict = None,
) -> tuple[dict, list[dict]]:
    """
    Full evaluation of a single feature.

    Returns tuple of (evaluation_dict, ic_records_list):
    - evaluation_dict keyed by horizon: {"fwd_5d": {mean_ic, ic_ir, ..., decay, turnover}, ...}
    - ic_records_list: list of dicts with keys [feature_id, horizon, date, ic] for primary horizon only
    """
    result = {}
    ic_records = []

    # Compute turnover once (applies to all horizons)
    turnover = compute_turnover(feature_wide, universe_mask)

    # Evaluate for each horizon
    for horizon, fwd_wide in forward_returns.items():
        fwd_rank = fwd_ranks_precomputed.get(horizon) if fwd_ranks_precomputed else None
        ic_series = compute_ic_series_fast(feature_wide, fwd_wide, universe_mask, fwd_rank)
        summary = compute_evaluation_summary(ic_series, train_cutoff_date)

        # Compute decay for this horizon
        decay = compute_decay(feature_wide, fwd_wide, universe_mask, config.IC_DECAY_LAGS, fwd_rank)

        # Combine metrics
        summary["decay"] = decay
        summary["turnover"] = turnover

        horizon_key = f"fwd_{horizon}d"
        result[horizon_key] = summary

        # Collect IC records for primary horizon only
        if horizon == config.PRIMARY_HORIZON:
            ic_clean = ic_series.dropna()
            for dt, val in ic_clean.items():
                ic_records.append({
                    "feature_id": feature_id,
                    "horizon": horizon_key,
                    "date": dt,
                    "ic": float(val),
                })

    return result, ic_records


def evaluate_all_features(
    store: FeatureStore,
    data: dict,
    universe_mask: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    """
    Evaluate all features in the store.
    Skips features that already have evaluation results (unless force=True).

    Returns DataFrame with one row per feature, columns for all metrics.

    Args:
        store: FeatureStore instance
        data: dict from load_all_data()
        universe_mask: boolean DataFrame (date x ticker)
        force: if True, recompute all evaluations
    """
    from research.targets import compute_forward_returns_multi

    adj_close = data["adj_close"]

    # Compute forward returns once (all horizons)
    print(f"    Computing forward returns...")
    forward_returns = compute_forward_returns_multi(adj_close, config.FORWARD_HORIZONS)

    # Pre-rank forward returns once (huge speedup: avoids re-ranking per feature)
    print(f"    Pre-ranking forward returns...")
    fwd_ranks_precomputed = {}
    for horizon, fwd_wide in forward_returns.items():
        fwd_masked = fwd_wide.where(universe_mask)
        fwd_ranks_precomputed[horizon] = fwd_masked.rank(axis=1, pct=True)

    # Compute train_cutoff_date
    dates = adj_close.index.sort_values()
    n_dates = len(dates)
    cutoff_idx = int(n_dates * config.TRAIN_FRACTION)
    train_cutoff_date = dates[cutoff_idx]

    # Evaluate each feature
    registry = store.get_registry()
    all_features = list(registry["features"].keys())
    n_total = len(all_features)

    rows = []
    all_ic_records = []

    for idx, feature_id in enumerate(all_features):
        if idx % 50 == 0:
            print(f"    Evaluating: {idx}/{n_total} features...")

        # Skip if already evaluated (unless force)
        if not force and store.has_evaluation(feature_id):
            continue

        try:
            # Load feature
            long_df = store.load_feature(feature_id)
            wide_df = long_df.pivot_table(
                index="date", columns="ticker", values="value"
            )

            # Evaluate (with pre-ranked forward returns)
            evaluation, ic_records = evaluate_feature(
                feature_id, wide_df, forward_returns, universe_mask,
                train_cutoff_date, fwd_ranks_precomputed,
            )
            all_ic_records.extend(ic_records)

            # Save evaluation
            store.save_evaluation(feature_id, evaluation)

            # Build row for results DataFrame
            row = {
                "feature_id": feature_id,
                "category": registry["features"][feature_id].get("category"),
                "level": registry["features"][feature_id].get("level"),
                "formula": registry["features"][feature_id].get("formula"),
            }

            # Flatten evaluation metrics
            for horizon, metrics in evaluation.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        col = f"{metric_name}_{horizon}"
                        row[col] = value

            rows.append(row)

            del wide_df, long_df

            # Persist IC records periodically to avoid data loss on interruption
            if len(all_ic_records) >= 50000:
                store.save_ic_timeseries_batch(all_ic_records)
                all_ic_records = []

        except Exception as e:
            print(f"    WARNING: Failed to evaluate {feature_id}: {e}")
            continue

    if all_ic_records:
        print(f"    Persisting IC time series ({len(all_ic_records)} records)...")
        store.save_ic_timeseries_batch(all_ic_records)

    store.save_registry()

    # Always return full evaluations from store (not just newly computed)
    return store.get_all_evaluations()

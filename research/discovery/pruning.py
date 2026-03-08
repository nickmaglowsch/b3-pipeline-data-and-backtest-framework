"""
Pruning and deduplication of features.
Multi-step pipeline to remove noise and redundancy.
"""
from __future__ import annotations

import pandas as pd
import numpy as np

from research.discovery import config
from research.discovery.store import FeatureStore

try:
    import cotahist_rs as _rs
    _RUST_CORR = True
except ImportError:
    _RUST_CORR = False


def filter_nan_and_variance(
    store: FeatureStore,
    feature_ids: list[str],
    universe_mask: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Remove features with excessive NaN rates or zero variance.

    Returns:
        (kept_feature_ids, removed_feature_ids)
    """
    removed = []

    for feature_id in feature_ids:
        try:
            wide_df = store.load_feature(feature_id)

            # Compute NaN rate over all cells
            total_cells = wide_df.size
            nan_cells = wide_df.isna().sum().sum()
            nan_rate = nan_cells / total_cells if total_cells > 0 else 0

            # Check zero-variance dates
            zero_var_count = (wide_df.std(axis=1) == 0).sum()
            zero_var_rate = zero_var_count / len(wide_df) if len(wide_df) > 0 else 0

            # Apply filters
            if nan_rate > config.MAX_NAN_RATE:
                removed.append(feature_id)
            elif zero_var_rate > (1 - config.MIN_VARIANCE_DATES_FRAC):
                removed.append(feature_id)

        except Exception as e:
            print(f"    WARNING: Error checking {feature_id}: {e}")
            removed.append(feature_id)

    kept = [fid for fid in feature_ids if fid not in removed]
    return kept, removed


def filter_by_ic(
    evaluations_df: pd.DataFrame,
    feature_ids: list[str],
    horizon: str = "fwd_20d",
    min_ic: float = None,
) -> list[str]:
    """
    Remove features where abs(mean_ic) < MIN_IC_THRESHOLD on the primary horizon.
    """
    if min_ic is None:
        min_ic = config.MIN_IC_THRESHOLD

    # Find the IC column
    col = f"mean_ic_{horizon}"
    if col not in evaluations_df.columns:
        # Fallback: look for any mean_ic column
        ic_cols = [c for c in evaluations_df.columns if "mean_ic" in c]
        if ic_cols:
            col = ic_cols[0]
        else:
            return feature_ids  # No IC data, keep all

    mask = evaluations_df["feature_id"].isin(feature_ids)
    valid = evaluations_df[mask]

    if len(valid) == 0:
        return feature_ids

    kept = valid[valid[col].abs() >= min_ic]["feature_id"].tolist()
    return kept


def compute_feature_correlation_matrix(
    store: FeatureStore,
    feature_ids: list[str],
    universe_mask: pd.DataFrame,
    sample_every_n_dates: int = 10,
) -> pd.DataFrame:
    """
    Compute pairwise Spearman correlation between features.

    Uses cross-sectional rank approach: for each sampled date, rank all stocks
    per feature, then compute correlation across the flattened rank vectors.

    Samples every Nth date for speed. Returns NxN DataFrame.
    """
    print(f"    Loading {len(feature_ids)} features for correlation computation...")

    # Sample dates first to reduce data volume
    all_dates = universe_mask.index
    sampled_dates = all_dates[::sample_every_n_dates]

    # Get the common set of tickers from universe mask
    common_tickers = universe_mask.columns.tolist()

    # Load features and extract cross-sectional ranks at sampled dates
    rank_data = {}
    valid_ids = []

    for i, fid in enumerate(feature_ids):
        if i % 100 == 0 and i > 0:
            print(f"      Loaded {i}/{len(feature_ids)} features...")
        try:
            wide_df = store.load_feature(fid)
            # Reindex to common grid (sampled_dates x common_tickers) so all arrays same length
            sampled = wide_df.reindex(index=sampled_dates, columns=common_tickers)
            ranked = sampled.rank(axis=1, pct=True)
            # Flatten to a single vector (date*ticker)
            rank_data[fid] = ranked.values.flatten()
            valid_ids.append(fid)
            del wide_df, sampled, ranked
        except Exception as e:
            print(f"      WARNING: Could not load {fid}: {e}")
            continue

    if not valid_ids:
        return pd.DataFrame()

    print(f"    Computing correlations for {len(valid_ids)} features on {len(sampled_dates)} dates...")

    if _RUST_CORR and len(valid_ids) >= 2:
        import pyarrow as pa
        arrow_vectors = [pa.array(rank_data[fid], type=pa.float64()) for fid in valid_ids]
        result_batch = _rs.compute_pairwise_spearman(arrow_vectors, valid_ids)
        corr_df = result_batch.to_pandas().set_index("feature_id")
        corr_df.index.name = None
        return corr_df

    # Python fallback: build rank matrix and use pandas .corr()
    rank_matrix = pd.DataFrame(rank_data)
    corr_matrix = rank_matrix.corr()
    return corr_matrix


def deduplicate_by_correlation(
    store: FeatureStore,
    feature_ids: list[str],
    evaluations_df: pd.DataFrame,
    universe_mask: pd.DataFrame,
    max_corr: float = None,
) -> tuple[list[str], list[dict], pd.DataFrame]:
    """
    Remove highly correlated features, keeping the one with higher IC_IR.

    Algorithm:
    1. Sort features by abs(IC_IR) descending (best first)
    2. For each feature (in order):
       a. If already marked as removed, skip
       b. Compare against all remaining features
       c. If correlation > max_corr with any already-kept feature, remove it
    3. Return the kept set and list of removed pairs

    Returns:
        (kept_feature_ids, removed_pairs_list)
    """
    if max_corr is None:
        max_corr = config.MAX_CORRELATION

    # Compute correlation matrix
    corr_matrix = compute_feature_correlation_matrix(
        store, feature_ids, universe_mask
    )

    if corr_matrix.empty:
        return feature_ids, [], corr_matrix

    # Sort by IC_IR (best first)
    ic_col = None
    for col in evaluations_df.columns:
        if "ic_ir" in col:
            ic_col = col
            break

    if ic_col is None:
        # No IC data, return all
        return feature_ids, [], corr_matrix

    relevant = evaluations_df[evaluations_df["feature_id"].isin(feature_ids)].copy()
    relevant = relevant.sort_values(ic_col, ascending=False, key=abs)
    sorted_ids = relevant["feature_id"].tolist()

    # Greedy deduplication
    kept = []
    removed_pairs = []

    for feature_id in sorted_ids:
        if feature_id not in corr_matrix.index:
            kept.append(feature_id)
            continue

        # Check correlation with already-kept features
        should_remove = False
        for kept_id in kept:
            if kept_id not in corr_matrix.index:
                continue

            corr_val = abs(corr_matrix.loc[feature_id, kept_id])
            if corr_val > max_corr:
                # Get IC_IR to determine which one to keep
                feat_ir = relevant[relevant["feature_id"] == feature_id][ic_col].values
                kept_ir = relevant[relevant["feature_id"] == kept_id][ic_col].values

                feat_ir = float(feat_ir[0]) if len(feat_ir) > 0 else 0
                kept_ir = float(kept_ir[0]) if len(kept_ir) > 0 else 0

                if abs(feat_ir) < abs(kept_ir):
                    removed_pairs.append({
                        "removed": feature_id,
                        "kept": kept_id,
                        "correlation": round(corr_val, 4),
                    })
                    should_remove = True
                    break

        if not should_remove:
            kept.append(feature_id)

    return kept, removed_pairs, corr_matrix


def enforce_cap(
    feature_ids: list[str],
    evaluations_df: pd.DataFrame,
    max_features: int = None,
    horizon: str = "fwd_20d",
) -> list[str]:
    """
    If more than MAX_FEATURES remain, keep only the top MAX_FEATURES by IC_IR.
    """
    if max_features is None:
        max_features = config.MAX_FEATURES

    if len(feature_ids) <= max_features:
        return feature_ids

    # Sort by IC_IR on primary horizon
    col = f"ic_ir_{horizon}"
    if col not in evaluations_df.columns:
        # Fallback
        ic_cols = [c for c in evaluations_df.columns if "ic_ir" in c]
        if ic_cols:
            col = ic_cols[0]
        else:
            return feature_ids

    relevant = evaluations_df[evaluations_df["feature_id"].isin(feature_ids)].copy()
    relevant = relevant.sort_values(col, ascending=False, key=abs)
    return relevant.head(max_features)["feature_id"].tolist()


def run_pruning_pipeline(
    store: FeatureStore,
    universe_mask: pd.DataFrame,
) -> tuple[list[str], dict]:
    """
    Run the full pruning pipeline.

    Returns:
        (kept_feature_ids, pruning_summary)
    """
    print("\n  Step 1: NaN/Variance Filter")
    registry = store.get_registry()
    all_features = list(registry["features"].keys())
    initial_count = len(all_features)

    kept, removed_nan = filter_nan_and_variance(store, all_features, universe_mask)
    after_nan_count = len(kept)
    print(f"    Removed {len(removed_nan)} features with excessive NaN/zero variance")
    print(f"    Kept: {after_nan_count}")

    print("\n  Step 2: IC Threshold Filter")
    evaluations_df = store.get_all_evaluations()
    kept_before_ic = list(kept)
    if not evaluations_df.empty:
        kept = filter_by_ic(evaluations_df, kept)
    removed_ic = [f for f in kept_before_ic if f not in kept]
    after_ic_count = len(kept)
    print(f"    Removed {len(removed_ic)} features with |mean_ic| < {config.MIN_IC_THRESHOLD}")
    print(f"    Kept: {after_ic_count}")

    print("\n  Step 3: Correlation Deduplication")
    kept_after_corr, removed_pairs, corr_matrix = deduplicate_by_correlation(
        store, kept, evaluations_df, universe_mask
    )
    print(f"    Removed {len(removed_pairs)} features due to high correlation")
    print(f"    Kept: {len(kept_after_corr)}")

    print("\n  Step 4: Cap Enforcement")
    kept_final = enforce_cap(kept_after_corr, evaluations_df)
    removed_cap = [f for f in kept_after_corr if f not in kept_final]
    print(f"    Enforcing MAX_FEATURES={config.MAX_FEATURES}")
    print(f"    Removed {len(removed_cap)} features by cap")
    print(f"    Final kept: {len(kept_final)}")

    # Subset correlation matrix for final kept features (reuse from dedup)
    final_corr = corr_matrix
    if not corr_matrix.empty:
        common = [f for f in kept_final if f in corr_matrix.index]
        if common:
            final_corr = corr_matrix.loc[common, common]

    # Build summary
    summary = {
        "initial_count": initial_count,
        "after_nan_filter": after_nan_count,
        "after_ic_filter": after_ic_count,
        "after_correlation_dedup": len(kept_after_corr),
        "after_cap": len(kept_final),
        "removed_by_nan": removed_nan,
        "removed_by_ic": removed_ic,
        "removed_by_correlation": removed_pairs,
        "removed_by_cap": removed_cap,
        "correlation_matrix": final_corr,
    }

    return kept_final, summary

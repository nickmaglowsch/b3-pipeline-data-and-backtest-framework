"""
Top-level worker functions for ProcessPoolExecutor-based parallelism.

These functions MUST be module-level (not closures or lambdas) so that
multiprocessing can pickle them. Workers are read-only with respect to the
FeatureStore registry — they return dicts and the main process does all writes.
"""
from __future__ import annotations

import sys
import traceback
import pandas as pd

from research.discovery.store import FeatureStore


def evaluate_feature_worker(args: tuple):
    """
    Worker function for parallel feature evaluation.

    Arguments tuple:
        (feature_id, store_dir_str, fwd_rank_parquet_paths, universe_mask_parquet,
         train_cutoff_date_str, horizons, decay_lags)

    Returns:
        dict with keys: feature_id, evaluation, ic_records
        OR None on any exception.
    """
    (
        feature_id,
        store_dir_str,
        fwd_rank_parquet_paths,
        universe_mask_parquet,
        train_cutoff_date_str,
        horizons,
        decay_lags,
    ) = args

    try:
        from pathlib import Path
        from research.discovery.evaluator import evaluate_feature

        store = FeatureStore(store_dir=Path(store_dir_str))

        # Load feature from Parquet
        long_df = store.load_feature(feature_id)
        wide_df = long_df.pivot_table(index="date", columns="ticker", values="value")

        # Load pre-computed forward rank DataFrames
        fwd_ranks_precomputed = {}
        for horizon, path_str in fwd_rank_parquet_paths.items():
            fwd_ranks_precomputed[horizon] = pd.read_parquet(path_str, engine="pyarrow")

        # Load universe mask
        universe_mask = pd.read_parquet(universe_mask_parquet, engine="pyarrow")

        # We need forward_returns dict for evaluate_feature (used for decay).
        # Since fwd_rank_precomputed is provided for all horizons, forward_return_wide
        # is only used where fwd_rank is None. We pass empty dict for forward_returns
        # and rely solely on precomputed ranks.
        # However, evaluate_feature passes forward_returns[horizon] to compute_decay,
        # so we need a stub. We reconstruct fwd_wide from fwd_rank (approximate).
        # A cleaner approach: pass fwd_rank as fwd_wide too — compute_decay will
        # re-rank it, but ranks of ranks == ranks, so it's equivalent.
        forward_returns = {h: fwd_ranks_precomputed[h] for h in horizons}

        train_cutoff_date = pd.Timestamp(train_cutoff_date_str)

        evaluation, ic_records = evaluate_feature(
            feature_id,
            wide_df,
            forward_returns,
            universe_mask,
            train_cutoff_date,
            fwd_ranks_precomputed,
        )

        return {
            "feature_id": feature_id,
            "evaluation": evaluation,
            "ic_records": ic_records,
        }

    except Exception:
        print(
            f"    [worker] ERROR evaluating {feature_id}:\n"
            + traceback.format_exc(),
            file=sys.stderr,
        )
        return None


def generate_unary_feature_worker(args: tuple):
    """
    Worker function for parallel Level 1 unary feature generation.

    Arguments tuple:
        (feature_id, op_name, new_id, store_dir_str, universe_mask_parquet, category)

    Returns:
        dict with keys: new_id, df (long DataFrame), metadata
        OR None on any exception.
    """
    feature_id, op_name, new_id, store_dir_str, universe_mask_parquet, category = args

    try:
        from pathlib import Path
        from research.discovery.operators import UNARY_OPS

        store = FeatureStore(store_dir=Path(store_dir_str))

        # Load feature from Parquet
        long_df = store.load_feature(feature_id)
        wide_df = long_df.pivot_table(index="date", columns="ticker", values="value")

        # Load universe mask
        universe_mask = pd.read_parquet(universe_mask_parquet, engine="pyarrow")

        # Apply unary operator
        op_func = UNARY_OPS[op_name]
        result = op_func(wide_df)

        # Mask and convert to long format
        masked = result.where(universe_mask)
        long_result = masked.stack().reset_index()
        long_result.columns = ["date", "ticker", "value"]
        long_result = long_result.dropna(subset=["value"])
        long_result["value"] = long_result["value"].astype("float32")

        metadata = {
            "category": category,
            "level": 1,
            "formula": new_id,
            "params": {},
        }

        return {"new_id": new_id, "df": long_result, "metadata": metadata}

    except Exception:
        print(
            f"    [worker] ERROR generating {new_id}:\n" + traceback.format_exc(),
            file=sys.stderr,
        )
        return None


def generate_delta_feature_worker(args: tuple):
    """
    Worker function for delta operator in Level 2 generation.

    Arguments tuple:
        (feature_id, period, new_id, store_dir_str, universe_mask_parquet, category)

    Returns:
        dict with keys: new_id, df (long DataFrame), metadata
        OR None on any exception.
    """
    feature_id, period, new_id, store_dir_str, universe_mask_parquet, category = args

    try:
        from pathlib import Path
        from research.discovery.operators import op_delta

        store = FeatureStore(store_dir=Path(store_dir_str))

        long_df = store.load_feature(feature_id)
        wide_df = long_df.pivot_table(index="date", columns="ticker", values="value")

        universe_mask = pd.read_parquet(universe_mask_parquet, engine="pyarrow")

        result = op_delta(wide_df, period)

        masked = result.where(universe_mask)
        long_result = masked.stack().reset_index()
        long_result.columns = ["date", "ticker", "value"]
        long_result = long_result.dropna(subset=["value"])
        long_result["value"] = long_result["value"].astype("float32")

        metadata = {
            "category": category,
            "level": 2,
            "formula": new_id,
            "params": {"period": period},
        }

        return {"new_id": new_id, "df": long_result, "metadata": metadata}

    except Exception:
        print(
            f"    [worker] ERROR generating {new_id}:\n" + traceback.format_exc(),
            file=sys.stderr,
        )
        return None


def generate_ratio_to_mean_feature_worker(args: tuple):
    """
    Worker function for ratio_to_mean operator in Level 2 generation.

    Arguments tuple:
        (feature_id, period, new_id, store_dir_str, universe_mask_parquet, category)

    Returns:
        dict with keys: new_id, df (long DataFrame), metadata
        OR None on any exception.
    """
    feature_id, period, new_id, store_dir_str, universe_mask_parquet, category = args

    try:
        from pathlib import Path
        from research.discovery.operators import op_ratio_to_mean

        store = FeatureStore(store_dir=Path(store_dir_str))

        long_df = store.load_feature(feature_id)
        wide_df = long_df.pivot_table(index="date", columns="ticker", values="value")

        universe_mask = pd.read_parquet(universe_mask_parquet, engine="pyarrow")

        result = op_ratio_to_mean(wide_df, period)

        masked = result.where(universe_mask)
        long_result = masked.stack().reset_index()
        long_result.columns = ["date", "ticker", "value"]
        long_result = long_result.dropna(subset=["value"])
        long_result["value"] = long_result["value"].astype("float32")

        metadata = {
            "category": category,
            "level": 2,
            "formula": new_id,
            "params": {"period": period},
        }

        return {"new_id": new_id, "df": long_result, "metadata": metadata}

    except Exception:
        print(
            f"    [worker] ERROR generating {new_id}:\n" + traceback.format_exc(),
            file=sys.stderr,
        )
        return None


def generate_binary_feature_worker(args: tuple):
    """
    Worker function for binary operators in Level 2 generation.

    Arguments tuple:
        (feat_a, feat_b, op_name, new_id, store_dir_str, universe_mask_parquet)

    Returns:
        dict with keys: new_id, df (long DataFrame), metadata
        OR None on any exception.
    """
    feat_a, feat_b, op_name, new_id, store_dir_str, universe_mask_parquet = args

    try:
        from pathlib import Path
        from research.discovery.operators import BINARY_OPS

        store = FeatureStore(store_dir=Path(store_dir_str))

        long_a = store.load_feature(feat_a)
        long_b = store.load_feature(feat_b)

        wide_a = long_a.pivot_table(index="date", columns="ticker", values="value")
        wide_b = long_b.pivot_table(index="date", columns="ticker", values="value")

        universe_mask = pd.read_parquet(universe_mask_parquet, engine="pyarrow")

        op_func = BINARY_OPS[op_name]
        result = op_func(wide_a, wide_b)

        masked = result.where(universe_mask)
        long_result = masked.stack().reset_index()
        long_result.columns = ["date", "ticker", "value"]
        long_result = long_result.dropna(subset=["value"])
        long_result["value"] = long_result["value"].astype("float32")

        metadata = {
            "category": "composite",
            "level": 2,
            "formula": new_id,
            "params": {},
        }

        return {"new_id": new_id, "df": long_result, "metadata": metadata}

    except Exception:
        print(
            f"    [worker] ERROR generating {new_id}:\n" + traceback.format_exc(),
            file=sys.stderr,
        )
        return None

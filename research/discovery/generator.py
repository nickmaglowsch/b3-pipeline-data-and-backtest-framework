"""
Feature generator engine for composite feature synthesis.
Combines base signals with operators to produce Level 1 and Level 2 features.
"""
from __future__ import annotations

import itertools
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import pandas as pd

from research.discovery import config
from research.discovery.store import FeatureStore
from research.discovery.operators import UNARY_OPS, BINARY_OPS


def generate_level1_features(
    store: FeatureStore,
    data: dict,
    universe_mask: pd.DataFrame,
) -> int:
    """
    Generate Level 1 features: apply rank and zscore to all base signals.
    Uses ProcessPoolExecutor for parallelism.

    For each base signal in the store, apply each unary operator and save
    the result to the store (if not already present).

    Args:
        store: FeatureStore instance (already populated with Level 0 base signals)
        data: dict from load_all_data() -- needed for universe mask
        universe_mask: boolean DataFrame (date x ticker)

    Returns:
        Number of new features generated.
    """
    from research.discovery._worker import generate_unary_feature_worker

    n_new = 0

    # Get all Level 0 features from registry
    registry = store.get_registry()
    level0_features = [
        fid for fid, meta in registry["features"].items()
        if meta.get("level") == 0
    ]

    print(f"  Generating Level 1 features for {len(level0_features)} base signals...")

    # Write universe mask to a tmp Parquet file shared across workers
    tmp_dir = Path(store.store_dir) / "tmp_gen_l1"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    universe_mask_path = str(tmp_dir / "universe_mask.parquet")
    universe_mask.to_parquet(universe_mask_path, engine="pyarrow")

    # Build args list, filtering already-computed features
    args_list = []
    for feature_id in level0_features:
        category = registry["features"][feature_id].get("category")
        if category in ["market_ibov", "market_cdi"]:
            continue
        for op_name in UNARY_OPS:
            new_id = f"{op_name}__{feature_id}"
            if store.has_feature(new_id):
                continue
            args_list.append((feature_id, op_name, new_id, str(store.store_dir), universe_mask_path, category))

    n_todo = len(args_list)
    print(f"    {n_todo} Level 1 features to generate (parallel, max_workers={config.MAX_WORKERS})...")

    with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
        futures = {pool.submit(generate_unary_feature_worker, args): args[2] for args in args_list}
        for future in as_completed(futures):
            new_id = futures[future]
            try:
                result = future.result()
            except Exception as e:
                print(f"    WARNING: Future for {new_id} raised: {e}")
                result = None

            if result is None:
                continue

            store.save_feature(result["new_id"], result["df"], result["metadata"])
            n_new += 1

            if n_new % 50 == 0:
                print(f"    Generated {n_new} Level 1 features...")

    store.save_registry()

    # Clean up tmp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return n_new


def generate_level2_features(
    store: FeatureStore,
    data: dict,
    universe_mask: pd.DataFrame,
    top_features_for_delta: list[str],
    top_features_for_binary: list[str],
    top_features_for_ratio_to_mean: list[str] = None,
) -> int:
    """
    Generate Level 2 features using ProcessPoolExecutor for parallelism:
    1. Apply delta(20) to top_features_for_delta
    2. Apply ratio_to_mean to top_features_for_ratio_to_mean
    3. Apply ratio and product to all pairs from top_features_for_binary

    Args:
        store: FeatureStore instance
        data: dict from load_all_data()
        universe_mask: boolean DataFrame (date x ticker)
        top_features_for_delta: top N feature IDs by IC_IR for delta operator
        top_features_for_binary: top N feature IDs by IC_IR for binary operators

    Returns:
        Number of new features generated.
    """
    from research.discovery._worker import (
        generate_delta_feature_worker,
        generate_ratio_to_mean_feature_worker,
        generate_binary_feature_worker,
    )

    n_new = 0
    registry = store.get_registry()

    print(f"  Generating Level 2 features (delta + binary)...")

    # Write universe mask to a tmp Parquet file shared across workers
    tmp_dir = Path(store.store_dir) / "tmp_gen_l2"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    universe_mask_path = str(tmp_dir / "universe_mask.parquet")
    universe_mask.to_parquet(universe_mask_path, engine="pyarrow")

    # --- Phase 1: Delta operator on top features ---
    delta_args = []
    for feature_id in top_features_for_delta:
        if feature_id not in registry["features"]:
            continue
        category = registry["features"][feature_id].get("category")
        for period in config.DELTA_PERIODS:
            new_id = f"delta{period}__{feature_id}"
            if store.has_feature(new_id):
                continue
            delta_args.append((feature_id, period, new_id, str(store.store_dir), universe_mask_path, category))

    if delta_args:
        print(f"    {len(delta_args)} delta features to generate (parallel)...")
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futures = {pool.submit(generate_delta_feature_worker, args): args[2] for args in delta_args}
            for future in as_completed(futures):
                new_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"    WARNING: Future for {new_id} raised: {e}")
                    result = None
                if result is None:
                    continue
                store.save_feature(result["new_id"], result["df"], result["metadata"])
                n_new += 1

    # --- Phase 1b: ratio_to_mean operator on top features ---
    if top_features_for_ratio_to_mean:
        rtm_args = []
        for feature_id in top_features_for_ratio_to_mean:
            if feature_id not in registry["features"]:
                continue
            category = registry["features"][feature_id].get("category")
            for period in config.RATIO_TO_MEAN_PERIODS:
                new_id = f"ratio_to_mean{period}__{feature_id}"
                if store.has_feature(new_id):
                    continue
                rtm_args.append((feature_id, period, new_id, str(store.store_dir), universe_mask_path, category))

        if rtm_args:
            print(f"    {len(rtm_args)} ratio_to_mean features to generate (parallel)...")
            with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
                futures = {pool.submit(generate_ratio_to_mean_feature_worker, args): args[2] for args in rtm_args}
                for future in as_completed(futures):
                    new_id = futures[future]
                    try:
                        result = future.result()
                    except Exception as e:
                        print(f"    WARNING: Future for {new_id} raised: {e}")
                        result = None
                    if result is None:
                        continue
                    store.save_feature(result["new_id"], result["df"], result["metadata"])
                    n_new += 1

    # --- Phase 2: Binary operators on top feature pairs ---
    binary_args = []
    for feat_a, feat_b in itertools.combinations(top_features_for_binary, 2):
        if feat_a not in registry["features"] or feat_b not in registry["features"]:
            continue
        cat_a = registry["features"][feat_a].get("category")
        cat_b = registry["features"][feat_b].get("category")
        if cat_a == cat_b:
            continue
        for op_name in BINARY_OPS:
            new_id = f"{op_name}__{feat_a}__{feat_b}"
            if store.has_feature(new_id):
                continue
            binary_args.append((feat_a, feat_b, op_name, new_id, str(store.store_dir), universe_mask_path))

    if binary_args:
        print(f"    {len(binary_args)} binary features to generate (parallel)...")
        with ProcessPoolExecutor(max_workers=config.MAX_WORKERS) as pool:
            futures = {pool.submit(generate_binary_feature_worker, args): args[3] for args in binary_args}
            completed_binary = 0
            for future in as_completed(futures):
                new_id = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"    WARNING: Future for {new_id} raised: {e}")
                    result = None
                if result is None:
                    continue
                store.save_feature(result["new_id"], result["df"], result["metadata"])
                n_new += 1
                completed_binary += 1
                if completed_binary % 50 == 0:
                    print(f"    Generated {n_new} Level 2 features...")

    store.save_registry()

    # Clean up tmp files
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return n_new


def run_generation_pipeline(
    store: FeatureStore,
    data: dict,
    universe_mask: pd.DataFrame,
) -> None:
    """
    Full generation pipeline:
    1. Generate Level 0 (base signals) and save to store
    2. Generate Level 1 (unary operators on all base signals)

    Level 2 generation is triggered separately AFTER evaluation of Level 0+1,
    because it depends on IC rankings.

    Args:
        store: FeatureStore instance
        data: dict from load_all_data()
        universe_mask: boolean DataFrame (date x ticker)
    """
    from research.discovery.base_signals import generate_all_base_signals

    print("\n  Phase 1: Generating Level 0 base signals...")
    n_new = 0
    n_skipped = 0

    for feature_id, category, params, wide_df in generate_all_base_signals(data):
        if store.has_feature(feature_id):
            n_skipped += 1
            continue

        try:
            # Handle market-level signals (Series) vs stock-level (DataFrame)
            if isinstance(wide_df, pd.Series):
                # Broadcast Series across all tickers in universe
                broadcast_df = pd.DataFrame(
                    {col: wide_df for col in universe_mask.columns},
                    index=wide_df.index,
                ).reindex(universe_mask.index)
                masked = broadcast_df.where(universe_mask).astype("float32")
            else:
                masked = wide_df.where(universe_mask).astype("float32")

            metadata = {
                "category": category,
                "level": 0,
                "formula": feature_id,
                "params": params,
            }
            store.save_feature(feature_id, masked, metadata)
            n_new += 1

        except Exception as e:
            print(f"    WARNING: Failed to generate {feature_id}: {e}")
            continue

        # Free memory
        del wide_df, masked

        if n_new % 50 == 0:
            print(f"    Generated {n_new} base signals...")

    print(f"  Level 0 complete: {n_new} new, {n_skipped} skipped")
    store.save_registry()

    # Generate Level 1
    print("\n  Phase 2: Generating Level 1 features...")
    n_level1 = generate_level1_features(store, data, universe_mask)
    print(f"  Level 1 complete: {n_level1} new features")

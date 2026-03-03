"""
Feature generator engine for composite feature synthesis.
Combines base signals with operators to produce Level 1 and Level 2 features.
"""
from __future__ import annotations

import itertools
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

    For each base signal in the store, apply each unary operator and save
    the result to the store (if not already present).

    Args:
        store: FeatureStore instance (already populated with Level 0 base signals)
        data: dict from load_all_data() -- needed for universe mask
        universe_mask: boolean DataFrame (date x ticker)

    Returns:
        Number of new features generated.
    """
    n_new = 0

    # Get all Level 0 features from registry
    registry = store.get_registry()
    level0_features = [
        fid for fid, meta in registry["features"].items()
        if meta.get("level") == 0
    ]

    print(f"  Generating Level 1 features for {len(level0_features)} base signals...")

    for feature_id in level0_features:
        # Skip market-level signals (Series, not DataFrames)
        category = registry["features"][feature_id].get("category")
        if category in ["market_ibov", "market_cdi"]:
            continue

        # Load wide-format feature
        long_df = store.load_feature(feature_id)

        # Pivot to wide (date x ticker)
        wide_df = long_df.pivot_table(
            index="date", columns="ticker", values="value"
        )

        # Apply unary operators
        for op_name, op_func in UNARY_OPS.items():
            new_id = f"{op_name}__{feature_id}"

            if store.has_feature(new_id):
                continue

            try:
                # Apply operator
                result = op_func(wide_df)

                # Mask and convert to long format
                masked = result.where(universe_mask)
                long_result = masked.stack().reset_index()
                long_result.columns = ["date", "ticker", "value"]
                long_result = long_result.dropna(subset=["value"])
                long_result["value"] = long_result["value"].astype("float32")

                # Save to store
                metadata = {
                    "category": category,
                    "level": 1,
                    "formula": new_id,
                    "params": {},
                }
                store.save_feature(new_id, long_result, metadata)
                n_new += 1

            except Exception as e:
                print(f"    WARNING: Failed to generate {new_id}: {e}")
                continue

        # Free memory
        del wide_df, long_df

        if n_new % 50 == 0:
            print(f"    Generated {n_new} Level 1 features...")

    store.save_registry()
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
    Generate Level 2 features:
    1. Apply delta(20) to top_features_for_delta
    2. Apply ratio and product to all pairs from top_features_for_binary

    Args:
        store: FeatureStore instance
        data: dict from load_all_data()
        universe_mask: boolean DataFrame (date x ticker)
        top_features_for_delta: top N feature IDs by IC_IR for delta operator
        top_features_for_binary: top N feature IDs by IC_IR for binary operators

    Returns:
        Number of new features generated.
    """
    from research.discovery.operators import op_delta

    n_new = 0
    registry = store.get_registry()

    print(f"  Generating Level 2 features (delta + binary)...")

    # Phase 1: Delta operator on top features
    for feature_id in top_features_for_delta:
        if feature_id not in registry["features"]:
            continue

        long_df = store.load_feature(feature_id)
        wide_df = long_df.pivot_table(
            index="date", columns="ticker", values="value"
        )
        category = registry["features"][feature_id].get("category")

        for period in config.DELTA_PERIODS:
            new_id = f"delta{period}__{feature_id}"

            if store.has_feature(new_id):
                continue

            try:
                result = op_delta(wide_df, period)

                # Mask and convert to long format
                masked = result.where(universe_mask)
                long_result = masked.stack().reset_index()
                long_result.columns = ["date", "ticker", "value"]
                long_result = long_result.dropna(subset=["value"])
                long_result["value"] = long_result["value"].astype("float32")

                # Save to store
                metadata = {
                    "category": category,
                    "level": 2,
                    "formula": new_id,
                    "params": {"period": period},
                }
                store.save_feature(new_id, long_result, metadata)
                n_new += 1

            except Exception as e:
                print(f"    WARNING: Failed to generate {new_id}: {e}")
                continue

        del wide_df, long_df

    # Phase 1b: ratio_to_mean operator on top features
    if top_features_for_ratio_to_mean:
        from research.discovery.operators import op_ratio_to_mean

        for feature_id in top_features_for_ratio_to_mean:
            if feature_id not in registry["features"]:
                continue

            long_df = store.load_feature(feature_id)
            wide_df = long_df.pivot_table(
                index="date", columns="ticker", values="value"
            )
            category = registry["features"][feature_id].get("category")

            for period in config.RATIO_TO_MEAN_PERIODS:
                new_id = f"ratio_to_mean{period}__{feature_id}"

                if store.has_feature(new_id):
                    continue

                try:
                    result = op_ratio_to_mean(wide_df, period)

                    # Mask and convert to long format
                    masked = result.where(universe_mask)
                    long_result = masked.stack().reset_index()
                    long_result.columns = ["date", "ticker", "value"]
                    long_result = long_result.dropna(subset=["value"])
                    long_result["value"] = long_result["value"].astype("float32")

                    # Save to store
                    metadata = {
                        "category": category,
                        "level": 2,
                        "formula": new_id,
                        "params": {"period": period},
                    }
                    store.save_feature(new_id, long_result, metadata)
                    n_new += 1

                except Exception as e:
                    print(f"    WARNING: Failed to generate {new_id}: {e}")
                    continue

            del wide_df, long_df

    # Phase 2: Binary operators on top feature pairs
    for feat_a, feat_b in itertools.combinations(top_features_for_binary, 2):
        if feat_a not in registry["features"] or feat_b not in registry["features"]:
            continue

        # Skip if both from same category (redundant)
        cat_a = registry["features"][feat_a].get("category")
        cat_b = registry["features"][feat_b].get("category")
        if cat_a == cat_b:
            continue

        # Load both features
        long_a = store.load_feature(feat_a)
        long_b = store.load_feature(feat_b)

        wide_a = long_a.pivot_table(index="date", columns="ticker", values="value")
        wide_b = long_b.pivot_table(index="date", columns="ticker", values="value")

        # Apply binary operators
        for op_name, op_func in BINARY_OPS.items():
            new_id = f"{op_name}__{feat_a}__{feat_b}"

            if store.has_feature(new_id):
                continue

            try:
                result = op_func(wide_a, wide_b)

                # Mask and convert to long format
                masked = result.where(universe_mask)
                long_result = masked.stack().reset_index()
                long_result.columns = ["date", "ticker", "value"]
                long_result = long_result.dropna(subset=["value"])
                long_result["value"] = long_result["value"].astype("float32")

                # Save to store
                metadata = {
                    "category": "composite",
                    "level": 2,
                    "formula": new_id,
                    "params": {},
                }
                store.save_feature(new_id, long_result, metadata)
                n_new += 1

            except Exception as e:
                print(f"    WARNING: Failed to generate {new_id}: {e}")
                continue

        del wide_a, wide_b, long_a, long_b

        if n_new % 50 == 0:
            print(f"    Generated {n_new} Level 2 features...")

    store.save_registry()
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
                masked = broadcast_df.where(universe_mask)
            else:
                masked = wide_df.where(universe_mask)

            long_df = masked.stack().reset_index()
            long_df.columns = ["date", "ticker", "value"]
            long_df = long_df.dropna(subset=["value"])
            long_df["value"] = long_df["value"].astype("float32")

            metadata = {
                "category": category,
                "level": 0,
                "formula": feature_id,
                "params": params,
            }
            store.save_feature(feature_id, long_df, metadata)
            n_new += 1

        except Exception as e:
            print(f"    WARNING: Failed to generate {feature_id}: {e}")
            continue

        # Free memory
        del wide_df, long_df

        if n_new % 50 == 0:
            print(f"    Generated {n_new} base signals...")

    print(f"  Level 0 complete: {n_new} new, {n_skipped} skipped")
    store.save_registry()

    # Generate Level 1
    print("\n  Phase 2: Generating Level 1 features...")
    n_level1 = generate_level1_features(store, data, universe_mask)
    print(f"  Level 1 complete: {n_level1} new features")

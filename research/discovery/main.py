#!/usr/bin/env python3
"""
B3 Feature Discovery Engine -- Main Orchestrator

Automatic feature generation, evaluation, and ranking.

Usage:
    python -m research.discovery.main                  # full run
    python -m research.discovery.main --incremental    # skip already-computed
    python -m research.discovery.main --force-recompute  # start fresh

Pipeline:
1. Load data from SQLite + external sources
2. Initialize feature store (check data hash)
3. Generate Level 0 base signals (~120-150)
4. Generate Level 1 features (rank + zscore on base signals)
5. Evaluate Level 0 + Level 1 features (IC computation)
6. Select top features for Level 2 generation
7. Generate Level 2 features (delta, ratio, product)
8. Evaluate Level 2 features
9. Run pruning pipeline
10. Export feature catalog JSON
11. Generate plots and report
12. Update feature store registry
"""
from __future__ import annotations

import argparse
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    """Main discovery pipeline orchestrator."""
    print("\n" + "=" * 80)
    print("B3 FEATURE DISCOVERY ENGINE")
    print("=" * 80)

    start_time = time.time()

    # Parse arguments
    parser = argparse.ArgumentParser(description="B3 Feature Discovery Engine")
    parser.add_argument(
        "--incremental", action="store_true",
        help="Skip already-computed features (default: recompute all)"
    )
    parser.add_argument(
        "--force-recompute", action="store_true",
        help="Invalidate feature store and recompute everything"
    )
    args = parser.parse_args()

    if args.incremental and args.force_recompute:
        parser.error("--incremental and --force-recompute are mutually exclusive")

    # Import modules
    from research.data_loader import load_all_data, print_data_summary
    from research.features import compute_universe_mask
    from research.discovery.store import FeatureStore
    from research.discovery.generator import (
        run_generation_pipeline, generate_level2_features
    )
    from research.discovery.evaluator import evaluate_all_features
    from research.discovery.pruning import run_pruning_pipeline
    from research.discovery.catalog import export_catalog
    from research.discovery.plots import generate_all_discovery_plots
    from research.discovery.report import generate_discovery_report
    from research.discovery import config

    # Step 1: Load data
    print("\nStep 1: Loading data...")
    t0 = time.time()
    data = load_all_data()
    print_data_summary(data)
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 2: Initialize feature store
    print("\nStep 2: Initializing feature store...")
    t0 = time.time()
    store = FeatureStore()
    data_hash = store.compute_data_hash(data)

    if args.force_recompute:
        print("  Force recompute: invalidating feature store...")
        store.invalidate()

    if not store.is_valid(data_hash):
        print("  Data has changed since last run. Invalidating feature store...")
        store.invalidate()

    # Update the data hash in the store
    store._registry["data_hash"] = data_hash
    store.save_registry()
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 3: Compute universe mask
    print("\nStep 3: Computing universe mask...")
    t0 = time.time()
    universe_mask = compute_universe_mask(
        data["close_px"], data["fin_vol"], data["adj_close"]
    )
    print(f"  Universe mask shape: {universe_mask.shape}")
    print(f"  Average stocks per date: {universe_mask.sum(axis=1).mean():.0f}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 4: Generate Level 0 + Level 1
    print("\nStep 4: Generating Level 0 and Level 1 features...")
    t0 = time.time()
    run_generation_pipeline(store, data, universe_mask)
    print(f"  Total features in store: {store.feature_count()}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 5: Evaluate Level 0 + Level 1
    print("\nStep 5: Evaluating Level 0 + Level 1 features...")
    t0 = time.time()
    force_eval = not args.incremental
    evaluations_df = evaluate_all_features(store, data, universe_mask, force=force_eval)
    store.save_registry()
    print(f"  Features with evaluation: {len(evaluations_df)}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 6: Select top features for Level 2
    print("\nStep 6: Selecting top features for Level 2 generation...")
    t0 = time.time()

    # Find IC_IR column
    ic_col = None
    for col in evaluations_df.columns:
        if "ic_ir" in col and "fwd_20d" in col:
            ic_col = col
            break
    if not ic_col:
        for col in evaluations_df.columns:
            if "ic_ir" in col:
                ic_col = col
                break

    if ic_col:
        # Top N for delta: stock-level features only (exclude market-level)
        stock_level = evaluations_df[evaluations_df["level"].isin([0, 1])]
        stock_level_sorted = stock_level.sort_values(ic_col, ascending=False, key=abs)
        top_for_delta = stock_level_sorted.head(config.TOP_N_FOR_DELTA)["feature_id"].tolist()

        # Top N for binary ops: Level 0 features only
        level0_sorted = evaluations_df[evaluations_df["level"] == 0].sort_values(
            ic_col, ascending=False, key=abs
        )
        top_for_binary = level0_sorted.head(config.TOP_N_FOR_BINARY_OPS)["feature_id"].tolist()

        # Top N for ratio_to_mean: stock-level features
        top_for_ratio_to_mean = stock_level_sorted.head(config.TOP_N_FOR_RATIO_TO_MEAN)["feature_id"].tolist()
    else:
        top_for_delta = []
        top_for_binary = []
        top_for_ratio_to_mean = []

    print(f"  Top {len(top_for_delta)} for delta, top {len(top_for_binary)} for binary ops, top {len(top_for_ratio_to_mean)} for ratio_to_mean")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 7: Generate Level 2
    print("\nStep 7: Generating Level 2 features...")
    t0 = time.time()
    if top_for_delta or top_for_binary or top_for_ratio_to_mean:
        n_level2 = generate_level2_features(
            store, data, universe_mask, top_for_delta, top_for_binary,
            top_features_for_ratio_to_mean=top_for_ratio_to_mean,
        )
        print(f"  Level 2 features generated: {n_level2}")
    else:
        print(f"  Skipping Level 2 (no top features selected)")
        n_level2 = 0
    store.save_registry()
    print(f"  Total features in store: {store.feature_count()}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 8: Evaluate Level 2
    print("\nStep 8: Evaluating Level 2 features...")
    t0 = time.time()
    if n_level2 > 0:
        evaluations_df = evaluate_all_features(store, data, universe_mask, force=force_eval)
        store.save_registry()
        print(f"  Features with evaluation: {len(evaluations_df)}")
    else:
        print(f"  Skipping (no Level 2 features)")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 9: Pruning
    print("\nStep 9: Running pruning pipeline...")
    t0 = time.time()
    evaluations_df = store.get_all_evaluations()
    kept_ids, pruning_summary = run_pruning_pipeline(store, universe_mask)
    print(f"  Features surviving pruning: {len(kept_ids)}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 10: Export catalog
    print("\nStep 10: Exporting feature catalog...")
    t0 = time.time()
    catalog_path = export_catalog(store, kept_ids, pruning_summary, adj_close=data["adj_close"])
    print(f"  Catalog exported to {catalog_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Step 11: Generate plots and report
    print("\nStep 11: Generating plots and report...")
    t0 = time.time()
    try:
        generate_all_discovery_plots(store, kept_ids, evaluations_df, pruning_summary, data, universe_mask)
    except Exception as e:
        print(f"  WARNING: Plot generation failed: {e}")

    report_text = generate_discovery_report(evaluations_df, kept_ids, pruning_summary)
    report_path = config.DISCOVERY_REPORT_PATH
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Report saved to {report_path}")
    print(f"  Completed in {time.time() - t0:.1f}s")

    # Final summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("DISCOVERY PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Total runtime: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Features generated: {store.feature_count()}")
    print(f"  Features after pruning: {len(kept_ids)}")
    print(f"\n  Output files:")
    print(f"    Feature catalog: {catalog_path}")
    print(f"    Report: {report_path}")
    print(f"    Plots: {config.OUTPUT_DIR}")
    print(f"    Feature store: {config.FEATURE_STORE_DIR}")
    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

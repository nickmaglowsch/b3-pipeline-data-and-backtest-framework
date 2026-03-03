"""
Text report generation for discovery results.
"""
from __future__ import annotations

import pandas as pd

from research.discovery import config


def generate_discovery_report(
    evaluations_df: pd.DataFrame,
    kept_feature_ids: list[str],
    pruning_summary: dict,
    data_summary: dict = None,
) -> str:
    """
    Generate the plain-text discovery report.

    Sections:
    1. DATASET OVERVIEW
    2. DISCOVERY SUMMARY
    3. TOP 30 FEATURES
    4. CATEGORY BREAKDOWN
    5. STABILITY ANALYSIS
    """
    report = []

    report.append("=" * 80)
    report.append("B3 FEATURE DISCOVERY RESULTS")
    report.append("=" * 80)

    # Section 1: Dataset Overview
    report.append("\n" + "=" * 80)
    report.append("1. DATASET OVERVIEW")
    report.append("=" * 80)
    if data_summary:
        report.append(f"  Date Range:          {data_summary.get('date_range', 'N/A')}")
        report.append(f"  Number of Tickers:   {data_summary.get('n_tickers', 'N/A')}")
        report.append(f"  Total Observations:  {data_summary.get('n_obs', 'N/A')}")
    report.append(f"  Training Fraction:   {config.TRAIN_FRACTION:.1%}")
    report.append(f"  Min IC Threshold:    {config.MIN_IC_THRESHOLD:.4f}")
    report.append(f"  Max Correlation:     {config.MAX_CORRELATION:.2f}")

    # Section 2: Discovery Summary
    report.append("\n" + "=" * 80)
    report.append("2. DISCOVERY SUMMARY")
    report.append("=" * 80)
    report.append(f"  Initial Features Generated:     {pruning_summary.get('initial_count', 0):>6d}")
    report.append(f"  After NaN Filter:              {pruning_summary.get('after_nan_filter', 0):>6d}")
    report.append(f"  After IC Filter:               {pruning_summary.get('after_ic_filter', 0):>6d}")
    report.append(f"  After Correlation Dedup:       {pruning_summary.get('after_correlation_dedup', 0):>6d}")
    report.append(f"  Final Features (after cap):    {pruning_summary.get('after_cap', 0):>6d}")
    report.append(f"\n  Total Removed Features:")
    report.append(f"    By NaN/Variance:   {len(pruning_summary.get('removed_by_nan', []))}")
    report.append(f"    By IC Threshold:   {len(pruning_summary.get('removed_by_ic', []))}")
    report.append(f"    By Correlation:    {len(pruning_summary.get('removed_by_correlation', []))}")
    report.append(f"    By Cap:            {len(pruning_summary.get('removed_by_cap', []))}")

    # Section 3: Top 30 Features
    if not evaluations_df.empty:
        report.append("\n" + "=" * 80)
        report.append("3. TOP 30 FEATURES BY IC_IR")
        report.append("=" * 80)

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
            plot_data = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)].copy()
            plot_data = plot_data.sort_values(ic_col, ascending=False, key=abs).head(30)

            report.append("\n  Rank | Feature ID (truncated) | Category      | Level | IC_IR")
            report.append("  " + "-" * 76)

            for idx, (_, row) in enumerate(plot_data.iterrows(), start=1):
                fid = row["feature_id"][:22]
                cat = str(row.get("category", "?"))[:13]
                level = row.get("level", 0)
                ic_ir = row.get(ic_col, 0.0)

                report.append(
                    f"  {idx:4d} | {fid:<22} | {cat:<13} | {level:5d} | {ic_ir:6.4f}"
                )

        # Section 4: Category Breakdown
        report.append("\n" + "=" * 80)
        report.append("4. CATEGORY BREAKDOWN")
        report.append("=" * 80)

        registry = {}  # Placeholder since we don't have store here
        categories = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)][
            "category"
        ].value_counts()

        report.append("\n  Category              | Count | % of Total")
        report.append("  " + "-" * 50)

        total = len(kept_feature_ids)
        for cat, count in categories.items():
            pct = 100.0 * count / total if total > 0 else 0
            report.append(f"  {str(cat):<20} | {count:5d} | {pct:6.2f}%")

        # Section 5: Stability Analysis
        report.append("\n" + "=" * 80)
        report.append("5. STABILITY ANALYSIS (Train vs Test)")
        report.append("=" * 80)

        train_cols = [c for c in evaluations_df.columns if "train" in c and "mean_ic" in c]
        test_cols = [c for c in evaluations_df.columns if "test" in c and "mean_ic" in c]

        if train_cols and test_cols:
            train_col = train_cols[0]
            test_col = test_cols[0]

            plot_data = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)].copy()
            plot_data = plot_data.dropna(subset=[train_col, test_col])

            if not plot_data.empty:
                report.append(f"\n  Train IC (mean):       {plot_data[train_col].mean():8.6f}")
                report.append(f"  Test IC (mean):        {plot_data[test_col].mean():8.6f}")
                report.append(f"  Train IC (std):        {plot_data[train_col].std():8.6f}")
                report.append(f"  Test IC (std):         {plot_data[test_col].std():8.6f}")

                # Overfit detection
                plot_data["overfit_ratio"] = plot_data[test_col] / (plot_data[train_col] + 1e-8)
                overfitted = (plot_data["overfit_ratio"] < 0.5).sum()
                report.append(f"\n  Features potentially overfitted (test IC < 50% train): {overfitted}")

    # Final section
    report.append("\n" + "=" * 80)
    report.append("FEATURE DISCOVERY COMPLETE")
    report.append("=" * 80)
    report.append(f"Report generated for {len(kept_feature_ids)} features")
    report.append(f"Primary horizon: {config.PRIMARY_HORIZON} days")
    report.append(f"Min IC threshold: {config.MIN_IC_THRESHOLD}")

    return "\n".join(report)

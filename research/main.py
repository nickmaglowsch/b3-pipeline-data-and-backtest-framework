#!/usr/bin/env python3
"""
B3 Feature Importance Discovery -- Main Orchestrator

Runs the complete research pipeline:
1. Load data from SQLite + external sources
2. Engineer 19 features (price, volume, cross-sectional, market regime)
3. Compute 3 binary classification targets
4. Train RandomForest and XGBoost classifiers
5. Extract and compare feature importance rankings
6. Generate visualizations and research report

Usage:
    python -m research.main
"""

import sys
import os
import time
from datetime import datetime

import numpy as np

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research import config
from research.data_loader import load_all_data, print_data_summary
from research.features import build_feature_matrix, FEATURE_NAMES
from research.targets import add_targets_to_feature_matrix
from research.modeling import run_all_experiments, save_results
from research.visualization import generate_all_plots


def generate_report(all_results: dict, full_df, feature_names: list) -> str:
    """
    Generate the plain-text research summary.

    Answers:
    1. Which features consistently appear important?
    2. Are they momentum-like, volatility-like, regime-like?
    3. Are results stable across targets?
    4. Is predictive power marginal or meaningful?
    5. Is there evidence of structural signal or mostly noise?
    """
    lines = []
    lines.append("=" * 70)
    lines.append("B3 FEATURE IMPORTANCE DISCOVERY -- RESEARCH SUMMARY")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")

    # Section 1: Dataset overview
    lines.append("1. DATASET OVERVIEW")
    lines.append("-" * 40)
    lines.append(f"Date range: {full_df['date'].min().date()} to {full_df['date'].max().date()}")
    lines.append(f"Total observations: {len(full_df):,}")
    lines.append(f"Unique tickers: {full_df['ticker'].nunique()}")
    lines.append(f"Features: {len(feature_names)}")
    lines.append(f"Target (20d >0) class balance: {full_df['target_20d'].mean():.3f}")
    lines.append("")

    # Section 2: Model performance summary
    lines.append("2. MODEL PERFORMANCE")
    lines.append("-" * 40)
    for exp_name, exp_data in all_results.items():
        lines.append(f"\n  Target: {exp_name}")
        for model_name, metrics in exp_data["metrics"].items():
            lines.append(f"    {model_name}:")
            lines.append(f"      Accuracy:  {metrics['accuracy']:.4f}")
            lines.append(f"      ROC AUC:   {metrics['roc_auc']:.4f}")
            lines.append(f"      Precision: {metrics['precision']:.4f}")
            lines.append(f"      Recall:    {metrics['recall']:.4f}")
            lines.append(f"      F1:        {metrics['f1']:.4f}")
    lines.append("")

    # Section 3: Top features (primary target)
    lines.append("3. TOP FEATURES (Primary Target: 20-Day Forward Return > 0)")
    lines.append("-" * 40)
    primary = all_results.get("target_20d", {})
    for model_name in ["RandomForest", "XGBoost"]:
        if model_name in primary.get("importance", {}):
            imp = primary["importance"][model_name]
            lines.append(f"\n  {model_name} -- Top 10:")
            # Find the builtin importance column
            builtin_col = [
                c for c in imp.columns
                if "importance" in c and "permutation" not in c and "std" not in c and c != "rank"
            ]
            builtin_col = builtin_col[0] if builtin_col else None
            for _, row in imp.head(10).iterrows():
                feat = row["feature"]
                perm = row["permutation_importance_mean"]
                if builtin_col:
                    builtin_val = row[builtin_col]
                    lines.append(
                        f"    {int(row['rank']):2d}. {feat:<25s}  builtin={builtin_val:.4f}  perm={perm:.4f}"
                    )
                else:
                    lines.append(f"    {int(row['rank']):2d}. {feat:<25s}  perm={perm:.4f}")
    lines.append("")

    # Section 4: Feature overlap between RF and XGB
    lines.append("4. RF vs XGB OVERLAP (Primary Target)")
    lines.append("-" * 40)
    rf_top10 = set(primary["importance"]["RandomForest"].head(10)["feature"]) if "RandomForest" in primary.get("importance", {}) else set()
    xgb_top10 = set(primary["importance"]["XGBoost"].head(10)["feature"]) if "XGBoost" in primary.get("importance", {}) else set()
    overlap = rf_top10 & xgb_top10
    lines.append(f"  RF top 10:  {sorted(rf_top10)}")
    lines.append(f"  XGB top 10: {sorted(xgb_top10)}")
    lines.append(f"  Overlap ({len(overlap)} features): {sorted(overlap)}")
    lines.append("")

    # Section 5: Stability across targets
    lines.append("5. FEATURE STABILITY ACROSS TARGETS")
    lines.append("-" * 40)

    for model_name in ["RandomForest", "XGBoost"]:
        top10_per_target = []
        for exp_name in ["target_20d", "target_60d", "target_20d_median"]:
            if exp_name in all_results and model_name in all_results[exp_name].get("importance", {}):
                top10 = set(all_results[exp_name]["importance"][model_name].head(10)["feature"])
                top10_per_target.append(top10)

        if len(top10_per_target) == 3:
            stable = top10_per_target[0] & top10_per_target[1] & top10_per_target[2]
            lines.append(f"\n  {model_name} -- stable top-10 across all 3 targets ({len(stable)}):")
            for f in sorted(stable):
                lines.append(f"    - {f}")
            if not stable:
                lines.append("    (no features appear in top 10 for all three targets)")
        else:
            lines.append(f"\n  {model_name} -- insufficient data for stability analysis")
    lines.append("")

    # Section 6: Key findings
    lines.append("6. KEY FINDINGS")
    lines.append("-" * 40)

    # Categorize features by type
    momentum_features = {
        "Return_1d", "Return_5d", "Return_20d", "Return_60d",
        "Distance_to_MA20", "Distance_to_MA50", "Distance_to_MA200",
        "Rank_momentum_60d",
    }
    volatility_features = {
        "Rolling_vol_20d", "Rolling_vol_60d", "ATR_14", "Drawdown_60d",
        "Rank_volatility_20d",
    }
    volume_features = {"Volume_zscore_20d", "Volume_ratio_5d_20d", "Rank_volume"}
    regime_features = {"Ibovespa_return_20d", "Ibovespa_vol_20d", "CDI_3m_change"}

    # Determine average AUC on primary target
    primary_aucs = []
    for model_name in ["RandomForest", "XGBoost"]:
        if model_name in primary.get("metrics", {}):
            primary_aucs.append(primary["metrics"][model_name]["roc_auc"])

    avg_auc = float(np.mean(primary_aucs)) if primary_aucs else 0.5

    lines.append(f"\n  a) Average ROC AUC on primary target: {avg_auc:.4f}")
    if avg_auc > 0.55:
        lines.append("     -> Meaningful predictive signal detected (AUC > 0.55)")
    elif avg_auc > 0.52:
        lines.append("     -> Marginal predictive signal (AUC 0.52-0.55)")
    else:
        lines.append("     -> Weak/no predictive signal (AUC near 0.50)")

    # Classify which feature categories dominate the overlap
    if overlap:
        mom_count = len(overlap & momentum_features)
        vol_count = len(overlap & volatility_features)
        vol_feat_count = len(overlap & volume_features)
        reg_count = len(overlap & regime_features)
        lines.append(f"\n  b) Overlapping top features by category:")
        lines.append(f"     Momentum/trend: {mom_count}")
        lines.append(f"     Volatility:     {vol_count}")
        lines.append(f"     Volume:         {vol_feat_count}")
        lines.append(f"     Regime/macro:   {reg_count}")
    lines.append("")

    # Section 7: Conclusion
    lines.append("7. CONCLUSION")
    lines.append("-" * 40)
    if avg_auc > 0.55:
        lines.append("  There is evidence of structural signal in the feature set.")
        lines.append("  The predictive power, while not large in absolute terms,")
        lines.append("  is consistent across models and exceeds random classification.")
    elif avg_auc > 0.52:
        lines.append("  There is marginal evidence of predictive signal.")
        lines.append("  Results should be interpreted cautiously -- the AUC is only")
        lines.append("  slightly above random, though feature importance rankings")
        lines.append("  may still reveal structural patterns.")
    else:
        lines.append("  There is little evidence of meaningful predictive signal")
        lines.append("  from these features for binary return classification.")
        lines.append("  This does not mean the features are useless -- they may")
        lines.append("  still be valuable in a portfolio construction context")
        lines.append("  (ranking/sorting) rather than point prediction.")

    lines.append("")
    lines.append("  Note: This is a hypothesis discovery exercise. These results")
    lines.append("  should inform further research, not drive trading decisions.")
    lines.append("")
    lines.append("=" * 70)
    lines.append("END OF REPORT")
    lines.append("=" * 70)

    report_text = "\n".join(lines)

    # Also print to stdout
    print(report_text)

    return report_text


def main():
    """Run the full feature importance discovery pipeline."""
    pipeline_start = time.time()

    print("\n" + "=" * 70)
    print("  B3 FEATURE IMPORTANCE DISCOVERY")
    print("  Hypothesis Discovery Exercise")
    print("=" * 70)
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Date range: {config.START_DATE} to {config.END_DATE}")
    print(f"  Database:   {config.DB_PATH}")
    print(f"  Output:     {config.OUTPUT_DIR}")
    print()

    # -- Step 1: Load data -----------------------------------------------
    print("-" * 70)
    print("  STEP 1/6: Loading data")
    print("-" * 70)
    step_start = time.time()
    try:
        data = load_all_data()
    except FileNotFoundError:
        print(f"ERROR: Database not found at {config.DB_PATH}")
        print("Run the B3 pipeline first: python -m b3_pipeline.main")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        raise

    print_data_summary(data)
    print(f"  Step 1 completed in {time.time() - step_start:.1f}s\n")

    # -- Step 2: Feature engineering -------------------------------------
    print("-" * 70)
    print("  STEP 2/6: Computing features")
    print("-" * 70)
    step_start = time.time()
    try:
        feature_matrix = build_feature_matrix(data)
    except Exception as e:
        print(f"ERROR computing features: {e}")
        raise

    print(f"  Feature matrix shape: {feature_matrix.shape}")
    print(f"  Date range: {feature_matrix['date'].min().date()} to {feature_matrix['date'].max().date()}")
    print(f"  Unique tickers: {feature_matrix['ticker'].nunique()}")
    print(f"  Step 2 completed in {time.time() - step_start:.1f}s\n")

    # -- Step 3: Target computation --------------------------------------
    print("-" * 70)
    print("  STEP 3/6: Computing targets")
    print("-" * 70)
    step_start = time.time()
    adj_close_for_targets = data["adj_close"]
    # Free wide-format data no longer needed (saves ~1+ GB)
    del data
    try:
        full_df = add_targets_to_feature_matrix(feature_matrix, adj_close_for_targets)
    except Exception as e:
        print(f"ERROR computing targets: {e}")
        raise
    del feature_matrix, adj_close_for_targets  # only full_df needed from here

    print(f"  Full dataset shape: {full_df.shape}")
    print(f"  Target_20d class balance: {full_df['target_20d'].mean():.3f}")
    print(f"  Target_60d class balance: {full_df['target_60d'].dropna().mean():.3f}")
    print(f"  Target_20d_median balance: {full_df['target_20d_median'].dropna().mean():.3f}")
    print(f"  Step 3 completed in {time.time() - step_start:.1f}s\n")

    # -- Step 4: Modeling and evaluation ---------------------------------
    print("-" * 70)
    print("  STEP 4/6: Training models and evaluating")
    print("-" * 70)
    step_start = time.time()
    try:
        all_results = run_all_experiments(full_df, FEATURE_NAMES)
    except Exception as e:
        print(f"ERROR during modeling: {e}")
        raise

    print(f"  Step 4 completed in {time.time() - step_start:.1f}s\n")

    # -- Step 5: Save results and generate plots -------------------------
    print("-" * 70)
    print("  STEP 5/6: Saving results and generating plots")
    print("-" * 70)
    step_start = time.time()
    try:
        save_results(all_results, FEATURE_NAMES)
        generate_all_plots(all_results)
    except Exception as e:
        print(f"ERROR saving results: {e}")
        raise

    print(f"  Step 5 completed in {time.time() - step_start:.1f}s\n")

    # -- Step 6: Generate report -----------------------------------------
    print("-" * 70)
    print("  STEP 6/6: Generating research report")
    print("-" * 70)
    step_start = time.time()
    try:
        report = generate_report(all_results, full_df, FEATURE_NAMES)
    except Exception as e:
        print(f"ERROR generating report: {e}")
        raise

    report_path = config.OUTPUT_DIR / config.SUMMARY_TXT
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {report_path}")
    print(f"  Step 6 completed in {time.time() - step_start:.1f}s\n")

    # -- Summary ---------------------------------------------------------
    total_time = time.time() - pipeline_start
    print("=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Total runtime: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"  Output files:")
    for f in sorted(config.OUTPUT_DIR.glob("*")):
        if not f.name.startswith("."):
            print(f"    {f.name}")
    print()


if __name__ == "__main__":
    main()

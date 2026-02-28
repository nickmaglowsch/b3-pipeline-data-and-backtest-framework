"""
Visualization for B3 feature importance study.
Dark-themed plots matching the existing backtest aesthetic.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research import config


# Reuse the project's established color palette
PALETTE = {
    "bg": "#0D1117",
    "panel": "#161B22",
    "grid": "#21262D",
    "text": "#E6EDF3",
    "sub": "#8B949E",
    "rf_color": "#00D4AA",    # teal (reuse "pretax" color for RandomForest)
    "xgb_color": "#7B61FF",   # purple (reuse "aftertax" color for XGBoost)
    "accent1": "#FF6B35",     # orange
    "accent2": "#FFC947",     # yellow
    "accent3": "#FF4C6A",     # red
}


def setup_style():
    """Set global matplotlib style to match backtest plots."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
        "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["sub"],
        "xtick.color": PALETTE["sub"],
        "ytick.color": PALETTE["sub"],
        "legend.facecolor": PALETTE["panel"],
        "legend.edgecolor": PALETTE["grid"],
        "legend.labelcolor": PALETTE["text"],
    })


def _get_builtin_col(model_name: str) -> str:
    """Return the column name for built-in importance for a given model."""
    return "gini_importance" if model_name == "RandomForest" else "gain_importance"


def _fmt_ax(ax):
    """Apply dark theme formatting to an axis."""
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["sub"], labelsize=8.5)
    ax.spines[:].set_color(PALETTE["grid"])
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.6, ls="--")
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.3, ls=":")


def plot_feature_importance_top15(
    importance_df: pd.DataFrame,
    experiment_name: str = "target_20d",
    metrics_dict: dict = None,
    out_path: str = None,
) -> None:
    """
    Horizontal bar chart comparing RF (Gini) vs XGB (Gain) importance
    for the top 15 features.

    Args:
        importance_df: DataFrame with columns: experiment, model, feature,
                       gini_importance/gain_importance, permutation_importance_mean, rank
        experiment_name: which experiment to plot (default: primary target)
        metrics_dict: optional dict with model metrics for subtitle
        out_path: output file path (default: config.OUTPUT_DIR / config.IMPORTANCE_PLOT)
    """
    if out_path is None:
        out_path = str(config.OUTPUT_DIR / config.IMPORTANCE_PLOT)

    # Filter to primary experiment
    rf_data = importance_df[
        (importance_df["experiment"] == experiment_name) &
        (importance_df["model"] == "RandomForest")
    ].copy()
    xgb_data = importance_df[
        (importance_df["experiment"] == experiment_name) &
        (importance_df["model"] == "XGBoost")
    ].copy()

    if rf_data.empty or xgb_data.empty:
        print(f"  WARNING: No data for experiment '{experiment_name}' -- skipping top15 plot")
        return

    # Use builtin_importance if available (unified column), else use model-specific
    rf_col = "builtin_importance" if "builtin_importance" in rf_data.columns else "gini_importance"
    xgb_col = "builtin_importance" if "builtin_importance" in xgb_data.columns else "gain_importance"

    # Normalize each model's importance to [0, 1] for comparable plotting
    rf_imp = rf_data.set_index("feature")[rf_col]
    xgb_imp = xgb_data.set_index("feature")[xgb_col]

    rf_norm = rf_imp / rf_imp.sum() if rf_imp.sum() > 0 else rf_imp
    xgb_norm = xgb_imp / xgb_imp.sum() if xgb_imp.sum() > 0 else xgb_imp

    # Merge and find top 15 by average normalized importance
    all_features = rf_norm.index.union(xgb_norm.index)
    avg_importance = (
        rf_norm.reindex(all_features).fillna(0) +
        xgb_norm.reindex(all_features).fillna(0)
    ) / 2
    top15_features = avg_importance.nlargest(15).index.tolist()
    # Reverse so most important is at top of horizontal bar chart
    top15_features = top15_features[::-1]

    rf_values = [rf_norm.get(f, 0) for f in top15_features]
    xgb_values = [xgb_norm.get(f, 0) for f in top15_features]

    # Build subtitle from metrics
    subtitle = ""
    if metrics_dict and experiment_name in metrics_dict:
        rf_auc = metrics_dict[experiment_name].get("RandomForest", {}).get("roc_auc", None)
        xgb_auc = metrics_dict[experiment_name].get("XGBoost", {}).get("roc_auc", None)
        if rf_auc and xgb_auc:
            subtitle = f"RF AUC: {rf_auc:.4f}  |  XGB AUC: {xgb_auc:.4f}"

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(PALETTE["bg"])

    y_pos = np.arange(len(top15_features))
    bar_height = 0.35

    ax.barh(y_pos - bar_height / 2, rf_values, height=bar_height,
            color=PALETTE["rf_color"], label="RF (Gini)", alpha=0.9)
    ax.barh(y_pos + bar_height / 2, xgb_values, height=bar_height,
            color=PALETTE["xgb_color"], label="XGB (Gain)", alpha=0.9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15_features, color=PALETTE["text"], fontsize=9)
    ax.set_xlabel("Normalized Importance (Gini / Gain)", color=PALETTE["sub"])
    ax.set_title(
        "Top 15 Feature Importances -- 20-Day Forward Return",
        color=PALETTE["text"], fontsize=13, pad=15
    )
    if subtitle:
        ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center",
                color=PALETTE["sub"], fontsize=9)

    ax.legend(loc="lower right")
    _fmt_ax(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved top-15 importance plot to {out_path}")


def plot_robustness_comparison(
    importance_df: pd.DataFrame,
    out_path: str = None,
) -> None:
    """
    Multi-panel chart showing feature importance stability across all three targets.

    Layout: 2 rows x 3 columns
    - Top row: RF importance for target_20d, target_60d, target_20d_median
    - Bottom row: XGB importance for target_20d, target_60d, target_20d_median

    Highlights features that appear in the top 10 across ALL targets.

    Args:
        importance_df: full DataFrame from importance_results.csv or all_results
        out_path: output file path
    """
    if out_path is None:
        out_path = str(config.OUTPUT_DIR / config.ROBUSTNESS_PLOT)

    models = ["RandomForest", "XGBoost"]
    targets = ["target_20d", "target_60d", "target_20d_median"]
    target_labels = ["20d Ret > 0", "60d Ret > 0", "20d Ret > Median"]

    # Use builtin_importance column if present, otherwise fall back to model-specific
    has_unified = "builtin_importance" in importance_df.columns

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        "Feature Importance Robustness Across Targets",
        color=PALETTE["text"], fontsize=14, y=1.01
    )

    for row, model in enumerate(models):
        model_color = PALETTE["rf_color"] if model == "RandomForest" else PALETTE["xgb_color"]

        # Find features in top 10 for ALL targets for this model
        top10_sets = []
        for target in targets:
            subset = importance_df[
                (importance_df["experiment"] == target) &
                (importance_df["model"] == model)
            ]
            if not subset.empty:
                top10 = set(subset.nsmallest(10, "rank")["feature"])
                top10_sets.append(top10)

        stable_features = set()
        if len(top10_sets) == 3:
            stable_features = top10_sets[0].intersection(top10_sets[1]).intersection(top10_sets[2])

        for col, (target, label) in enumerate(zip(targets, target_labels)):
            ax = axes[row, col]
            subset = importance_df[
                (importance_df["experiment"] == target) &
                (importance_df["model"] == model)
            ]

            if subset.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color=PALETTE["sub"])
                ax.set_facecolor(PALETTE["panel"])
                continue

            top10 = subset.nsmallest(10, "rank").copy()

            # Get importance values
            if has_unified:
                imp_col = "builtin_importance"
            else:
                imp_col = "gini_importance" if model == "RandomForest" else "gain_importance"

            if imp_col not in top10.columns:
                # fallback
                imp_col = [c for c in top10.columns if "importance" in c and "permutation" not in c and "std" not in c]
                imp_col = imp_col[0] if imp_col else "rank"

            top10 = top10.sort_values(imp_col, ascending=True)  # ascending for barh

            colors = [
                PALETTE["accent2"] if f in stable_features else model_color
                for f in top10["feature"]
            ]

            ax.barh(range(len(top10)), top10[imp_col], color=colors, alpha=0.9)
            ax.set_yticks(range(len(top10)))
            ax.set_yticklabels(top10["feature"], fontsize=7.5, color=PALETTE["text"])
            ax.set_title(
                f"{model}\n{label}",
                color=PALETTE["text"], fontsize=9
            )
            ax.set_xlabel("Importance", color=PALETTE["sub"], fontsize=8)
            _fmt_ax(ax)

    # Add legend for stable feature highlight
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["accent2"], label="Stable (top-10 all targets)"),
        Patch(facecolor=PALETTE["rf_color"], label="RandomForest"),
        Patch(facecolor=PALETTE["xgb_color"], label="XGBoost"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        facecolor=PALETTE["panel"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PALETTE["bg"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved robustness comparison plot to {out_path}")


def generate_all_plots(all_results: dict) -> None:
    """
    Convenience function: generate all plots from the results dict.
    Called from main.py after all experiments complete.

    Args:
        all_results: dict from modeling.run_all_experiments()
    """
    setup_style()

    # Build a consolidated DataFrame from all_results for convenience
    rows = []
    for exp_name, exp_data in all_results.items():
        for model_name, imp_df in exp_data["importance"].items():
            imp_copy = imp_df.copy()
            imp_copy["experiment"] = exp_name
            imp_copy["model"] = model_name
            rows.append(imp_copy)

    if not rows:
        print("  WARNING: No importance data found -- skipping plots")
        return

    importance_df = pd.concat(rows, ignore_index=True)

    # Normalize column names: create unified builtin_importance column
    if "gini_importance" in importance_df.columns and "gain_importance" not in importance_df.columns:
        importance_df["builtin_importance"] = importance_df["gini_importance"]
    elif "gain_importance" in importance_df.columns and "gini_importance" not in importance_df.columns:
        importance_df["builtin_importance"] = importance_df["gain_importance"]
    elif "gini_importance" in importance_df.columns and "gain_importance" in importance_df.columns:
        importance_df["builtin_importance"] = importance_df["gini_importance"].fillna(
            importance_df["gain_importance"]
        )

    # Build metrics dict for subtitle
    metrics_dict = {
        exp_name: exp_data["metrics"]
        for exp_name, exp_data in all_results.items()
    }

    plot_feature_importance_top15(importance_df, metrics_dict=metrics_dict)
    plot_robustness_comparison(importance_df)

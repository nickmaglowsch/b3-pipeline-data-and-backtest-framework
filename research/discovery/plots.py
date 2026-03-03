"""
Enhanced plots for feature discovery results.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from research.discovery import config
from research.visualization import PALETTE, setup_style, _fmt_ax
from research.discovery.store import FeatureStore


def plot_ic_bar_chart(
    evaluations_df: pd.DataFrame,
    kept_feature_ids: list[str],
    horizon: str = "fwd_20d",
    top_n: int = 30,
    out_path: str = None,
) -> None:
    """Horizontal bar chart of top N features by IC_IR."""
    if evaluations_df.empty or len(kept_feature_ids) == 0:
        return

    # Find IC_IR column
    ic_col = f"ic_ir_{horizon}"
    if ic_col not in evaluations_df.columns:
        ic_cols = [c for c in evaluations_df.columns if "ic_ir" in c]
        if not ic_cols:
            return
        ic_col = ic_cols[0]

    # Filter to kept features
    plot_data = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)].copy()
    plot_data = plot_data.sort_values(ic_col, ascending=False, key=abs).head(top_n)

    if plot_data.empty:
        return

    # Category color mapping
    cat_colors = {
        "momentum": PALETTE["rf_color"],
        "volatility": PALETTE["xgb_color"],
        "volume": PALETTE["accent1"],
        "beta": PALETTE["accent2"],
        "market_ibov": PALETTE["accent3"],
        "market_cdi": PALETTE["accent3"],
        "composite": "#4FC3F7",
    }

    fig, ax = plt.subplots(figsize=(10, 12))
    setup_style()

    colors = [cat_colors.get(cat, PALETTE["sub"]) for cat in plot_data["category"]]
    ax.barh(range(len(plot_data)), plot_data[ic_col], color=colors)

    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels([id[:35] for id in plot_data["feature_id"]], fontsize=8)
    ax.set_xlabel("IC Information Ratio", fontsize=10)
    ax.set_title(f"Top {top_n} Features by IC_IR -- {horizon}", fontsize=12, fontweight="bold")
    ax.axvline(0, color="white", linestyle="-", linewidth=0.5)

    _fmt_ax(ax)
    plt.tight_layout()

    out_path = out_path or config.OUTPUT_DIR / f"discovery_ic_top{top_n}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ic_decay(
    evaluations_df: pd.DataFrame,
    kept_feature_ids: list[str],
    top_n: int = 20,
    out_path: str = None,
) -> None:
    """Grouped bar chart showing IC at lag 0, 1, 5, 20 for top N features."""
    if evaluations_df.empty:
        return

    # Find IC column
    ic_col = None
    for col in evaluations_df.columns:
        if "mean_ic" in col and "fwd" in col:
            ic_col = col
            break

    if not ic_col:
        return

    plot_data = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)].copy()
    plot_data = plot_data.sort_values(ic_col, ascending=False, key=abs).head(top_n)

    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    setup_style()

    # Extract decay data from evaluation columns
    decay_cols = {col: col for col in plot_data.columns if col.startswith("decay_")}
    lag_labels = []
    lag_values = {}

    # Try to extract IC value + decay lag values per feature
    feature_labels = []
    ic_at_lag0 = []

    for _, row in plot_data.iterrows():
        fid = row["feature_id"][:20]
        feature_labels.append(fid)
        ic_val = row.get(ic_col, 0.0)
        ic_at_lag0.append(float(ic_val) if pd.notna(ic_val) else 0.0)

    x = np.arange(len(feature_labels))
    width = 0.6

    # Single bar for IC_IR
    ax.bar(x, ic_at_lag0, width, label="IC_IR", alpha=0.8, color=PALETTE["rf_color"])

    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("IC_IR", fontsize=10)
    ax.set_title(f"Top {top_n} Features by IC_IR", fontsize=12, fontweight="bold")
    ax.legend()

    _fmt_ax(ax)
    plt.tight_layout()

    out_path = out_path or config.OUTPUT_DIR / "discovery_ic_decay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_turnover_vs_ic(
    evaluations_df: pd.DataFrame,
    kept_feature_ids: list[str],
    horizon: str = "fwd_20d",
    out_path: str = None,
) -> None:
    """Scatter plot: mean_ic (x) vs turnover (y) for all surviving features."""
    if evaluations_df.empty:
        return

    ic_col = f"mean_ic_{horizon}"
    if ic_col not in evaluations_df.columns:
        ic_cols = [c for c in evaluations_df.columns if "mean_ic" in c]
        if not ic_cols:
            return
        ic_col = ic_cols[0]

    plot_data = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)].copy()

    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    setup_style()

    # Category color mapping
    cat_colors = {
        "momentum": PALETTE["rf_color"],
        "volatility": PALETTE["xgb_color"],
        "volume": PALETTE["accent1"],
        "beta": PALETTE["accent2"],
        "composite": "#4FC3F7",
    }

    # Find turnover column
    turnover_col = None
    for col in plot_data.columns:
        if "turnover" in col.lower():
            turnover_col = col
            break

    if turnover_col is None:
        # No turnover data; skip plot
        plt.close()
        return

    for category in plot_data["category"].unique():
        cat_data = plot_data[plot_data["category"] == category]
        color = cat_colors.get(category, PALETTE["sub"])
        cat_data_valid = cat_data.dropna(subset=[ic_col, turnover_col])
        if not cat_data_valid.empty:
            ax.scatter(cat_data_valid[ic_col], cat_data_valid[turnover_col],
                       label=category, alpha=0.6, s=100, color=color)

    ax.axvline(0, color="white", linestyle="-", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="High turnover")
    ax.set_xlabel("Mean IC", fontsize=10)
    ax.set_ylabel("Turnover", fontsize=10)
    ax.set_title("Feature IC vs Turnover Trade-off", fontsize=12, fontweight="bold")
    ax.legend()

    _fmt_ax(ax)
    plt.tight_layout()

    out_path = out_path or config.OUTPUT_DIR / "discovery_turnover_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_train_test_ic(
    evaluations_df: pd.DataFrame,
    kept_feature_ids: list[str],
    horizon: str = "fwd_20d",
    out_path: str = None,
) -> None:
    """Scatter plot: train IC (x-axis) vs test IC (y-axis)."""
    if evaluations_df.empty:
        return

    train_col = f"mean_ic_train_{horizon}"
    test_col = f"mean_ic_test_{horizon}"

    if train_col not in evaluations_df.columns or test_col not in evaluations_df.columns:
        # Try to find any train/test columns
        train_cols = [c for c in evaluations_df.columns if "train" in c and "mean_ic" in c]
        test_cols = [c for c in evaluations_df.columns if "test" in c and "mean_ic" in c]
        if train_cols and test_cols:
            train_col = train_cols[0]
            test_col = test_cols[0]
        else:
            return

    plot_data = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)].copy()
    plot_data = plot_data.dropna(subset=[train_col, test_col])

    if plot_data.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 10))
    setup_style()

    # Category color mapping
    cat_colors = {
        "momentum": PALETTE["rf_color"],
        "volatility": PALETTE["xgb_color"],
        "volume": PALETTE["accent1"],
        "composite": "#4FC3F7",
    }

    for category in plot_data["category"].unique():
        cat_data = plot_data[plot_data["category"] == category]
        color = cat_colors.get(category, PALETTE["sub"])

        ax.scatter(cat_data[train_col], cat_data[test_col],
                   label=category, alpha=0.6, s=80, color=color)

    # 45-degree reference line
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, "white", linestyle="--", linewidth=1, label="No overfit")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    ax.set_xlabel("Train IC", fontsize=10)
    ax.set_ylabel("Test IC", fontsize=10)
    ax.set_title("Train vs Test IC -- Overfit Detection", fontsize=12, fontweight="bold")
    ax.legend()

    _fmt_ax(ax)
    plt.tight_layout()

    out_path = out_path or config.OUTPUT_DIR / "discovery_train_test_scatter.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_ic_timeseries(
    store: FeatureStore,
    kept_feature_ids: list[str],
    evaluations_df: pd.DataFrame,
    horizon: str = "fwd_20d",
    top_n: int = 10,
    out_path: str = None,
) -> None:
    """Plot rolling 1-year IC time series for top N features."""
    if evaluations_df.empty or len(kept_feature_ids) == 0:
        return

    # Find IC_IR column
    ic_col = f"ic_ir_{horizon}"
    if ic_col not in evaluations_df.columns:
        ic_cols = [c for c in evaluations_df.columns if "ic_ir" in c]
        if not ic_cols:
            return
        ic_col = ic_cols[0]

    # Get top N features by IC_IR
    plot_data = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)].copy()
    plot_data = plot_data.sort_values(ic_col, ascending=False, key=abs).head(top_n)

    if plot_data.empty:
        print(f"    No features to plot for {horizon}")
        return

    top_ids = plot_data["feature_id"].tolist()

    # Load IC time series from store
    ic_df = store.load_ic_timeseries(feature_ids=top_ids, horizon=horizon)

    if ic_df.empty:
        print(f"    ! IC time series not persisted; cannot plot")
        return

    # Pivot to wide format
    ic_wide = ic_df.pivot(index="date", columns="feature_id", values="ic")

    # Compute rolling 252-day mean IC (1-year rolling average)
    rolling_ic = ic_wide.rolling(252, min_periods=126).mean()

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    setup_style()

    # Color palette for up to 10 lines
    colors = [
        PALETTE["rf_color"],
        PALETTE["xgb_color"],
        PALETTE["accent1"],
        PALETTE["accent2"],
        PALETTE["accent3"],
        "#4FC3F7",
        "#81C784",
        "#FFB74D",
        "#BA68C8",
        "#4DD0E1",
    ]

    for idx, col in enumerate(rolling_ic.columns):
        color = colors[idx % len(colors)]
        ax.plot(rolling_ic.index, rolling_ic[col], label=col[:25], linewidth=2, color=color)

    # Horizontal line at y=0
    ax.axhline(0, color="white", linestyle="-", linewidth=0.5)

    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Rolling 1-Year IC", fontsize=10)
    ax.set_title(f"Rolling 1-Year IC -- Top {top_n} Features ({horizon})", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")

    _fmt_ax(ax)
    plt.tight_layout()

    out_path = out_path or config.OUTPUT_DIR / "discovery_ic_timeseries.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    evaluations_df: pd.DataFrame = None,
    kept_feature_ids: list[str] = None,
    top_n: int = 50,
    out_path: str = None,
) -> None:
    """Plot correlation clustering heatmap."""
    if corr_matrix.empty or corr_matrix.shape[0] < 2:
        return

    # Subset to top N if needed
    if corr_matrix.shape[0] > top_n and evaluations_df is not None and kept_feature_ids is not None:
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
            # Get top N features by IC_IR
            top_data = evaluations_df[evaluations_df["feature_id"].isin(kept_feature_ids)]
            top_data = top_data.sort_values(ic_col, ascending=False, key=abs).head(top_n)
            top_ids = top_data["feature_id"].tolist()

            # Intersect with correlation matrix index
            valid_ids = [fid for fid in top_ids if fid in corr_matrix.index]
            if valid_ids:
                corr_matrix = corr_matrix.loc[valid_ids, valid_ids]

    if corr_matrix.shape[0] < 2:
        return

    # Replace NaN with 0
    corr_matrix = corr_matrix.fillna(0)

    # Perform hierarchical clustering
    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import squareform

        dist_matrix = 1 - np.abs(corr_matrix.values)
        np.fill_diagonal(dist_matrix, 0)
        condensed = squareform(dist_matrix, checks=False)
        Z = linkage(condensed, method="ward")
        order = leaves_list(Z)
    except Exception as e:
        print(f"    ! Clustering failed: {e}; using unclustered order")
        order = np.arange(corr_matrix.shape[0])

    # Reorder both axes
    ordered_ids = corr_matrix.index[order].tolist()
    corr_ordered = corr_matrix.loc[ordered_ids, ordered_ids]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    fig.patch.set_facecolor(PALETTE["background"])
    setup_style()

    # Heatmap
    im = ax.imshow(corr_ordered.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    # Set tick labels
    ax.set_xticks(np.arange(len(ordered_ids)))
    ax.set_yticks(np.arange(len(ordered_ids)))
    ax.set_xticklabels([fid[:20] for fid in ordered_ids], rotation=90, fontsize=8)
    ax.set_yticklabels([fid[:20] for fid in ordered_ids], fontsize=8)

    ax.set_title("Feature Correlation Matrix (Clustered)", fontsize=12, fontweight="bold")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Correlation", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["sub"])
    cbar.outline.set_edgecolor(PALETTE["grid"])

    _fmt_ax(ax)
    plt.tight_layout()

    out_path = out_path or config.OUTPUT_DIR / "discovery_correlation_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_all_discovery_plots(
    store: FeatureStore,
    kept_feature_ids: list[str],
    evaluations_df: pd.DataFrame,
    pruning_summary: dict,
    data: dict,
    universe_mask: pd.DataFrame,
) -> None:
    """Generate all discovery plots."""
    setup_style()

    print("\n  Generating discovery plots...")

    try:
        plot_ic_bar_chart(evaluations_df, kept_feature_ids)
        print(f"    ✓ IC bar chart")
    except Exception as e:
        print(f"    ! IC bar chart failed: {e}")

    try:
        plot_ic_decay(evaluations_df, kept_feature_ids)
        print(f"    ✓ IC decay chart")
    except Exception as e:
        print(f"    ! IC decay chart failed: {e}")

    try:
        plot_turnover_vs_ic(evaluations_df, kept_feature_ids)
        print(f"    ✓ Turnover vs IC scatter")
    except Exception as e:
        print(f"    ! Turnover vs IC scatter failed: {e}")

    try:
        plot_train_test_ic(evaluations_df, kept_feature_ids)
        print(f"    ✓ Train vs test IC scatter")
    except Exception as e:
        print(f"    ! Train vs test IC scatter failed: {e}")

    try:
        plot_ic_timeseries(store, kept_feature_ids, evaluations_df)
        print(f"    ✓ IC time series chart")
    except Exception as e:
        print(f"    ! IC time series chart failed: {e}")

    try:
        corr_matrix = pruning_summary.get("correlation_matrix", pd.DataFrame())
        plot_correlation_heatmap(corr_matrix, evaluations_df, kept_feature_ids)
        print(f"    ✓ Correlation heatmap")
    except Exception as e:
        print(f"    ! Correlation heatmap failed: {e}")

    print(f"  Plots saved to {config.OUTPUT_DIR}")

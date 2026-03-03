"""
Discovery Chart Components
==========================
Plotly chart functions for the Feature Discovery page.
All functions return go.Figure for use with st.plotly_chart().
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ui.components.charts import PALETTE, _apply_dark_theme

CATEGORY_COLORS = {
    "momentum": "#00D4AA",
    "volatility": "#7B61FF",
    "volume": "#FF6B35",
    "beta": "#FFC947",
    "composite": "#4FC3F7",
    "distribution": "#FF4C6A",
    "extremes": "#4ECDC4",
    "liquidity": "#A8E6CF",
    "autocorr": "#8B949E",
    "market_ibov": "#F7DC6F",
    "market_cdi": "#F7DC6F",
    "mean_reversion": "#82E0AA",
}


def _get_category_color(category: str) -> str:
    """Return color for a category, with fallback."""
    return CATEGORY_COLORS.get(category, PALETTE["sub"])


def plot_ic_bar_chart(
    features: list[dict],
    horizon: str = "fwd_20d",
    top_n: int = 30,
) -> go.Figure:
    """
    Horizontal bar chart of top N features sorted by abs(IC_IR) for the given horizon.

    Args:
        features: The catalog["features"] list.
        horizon: Evaluation horizon (e.g. "fwd_20d").
        top_n: Number of top features to show.

    Returns:
        Plotly Figure.
    """
    if not features:
        return _apply_dark_theme(go.Figure())

    # Build rows with IC_IR for the given horizon
    rows = []
    for f in features:
        metrics = f.get("metrics", {}).get(horizon, {})
        if not metrics:
            continue
        rows.append({
            "id": f["id"],
            "category": f.get("category", ""),
            "level": f.get("level", 0),
            "formula_human": f.get("formula_human", f["id"]),
            "ic_ir": metrics.get("ic_ir", 0.0),
            "mean_ic": metrics.get("mean_ic", 0.0),
            "pct_positive_ic": metrics.get("pct_positive_ic", 0.0),
        })

    if not rows:
        return _apply_dark_theme(go.Figure())

    df = pd.DataFrame(rows)
    df = df.reindex(df["ic_ir"].abs().sort_values(ascending=False).index)
    df = df.head(top_n).iloc[::-1]  # Reverse so highest is on top

    labels = [s[:35] for s in df["formula_human"]]
    colors = [_get_category_color(c) for c in df["category"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["ic_ir"],
        y=labels,
        orientation="h",
        marker_color=colors,
        customdata=list(zip(
            df["id"],
            df["category"],
            df["level"],
            df["mean_ic"],
            df["pct_positive_ic"],
        )),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Category: %{customdata[1]} | Level: %{customdata[2]}<br>"
            "IC_IR: %{x:.4f}<br>"
            "Mean IC: %{customdata[3]:.6f}<br>"
            "% Positive IC: %{customdata[4]:.1%}<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line_color=PALETTE["sub"], line_width=1)

    _apply_dark_theme(
        fig,
        title=f"Top {top_n} Features by IC_IR -- {horizon}",
        xaxis_title="IC_IR",
        height=max(400, top_n * 22),
    )
    fig.update_layout(yaxis=dict(autorange="reversed" if top_n <= 1 else True))
    return fig


def plot_ic_decay_chart(
    features: list[dict],
    horizon: str = "fwd_20d",
    top_n: int = 15,
) -> go.Figure:
    """
    Grouped bar chart showing IC at lag 0, 1, 5, 20 for the top N features.

    Args:
        features: The catalog["features"] list.
        horizon: Evaluation horizon.
        top_n: Number of top features to show.

    Returns:
        Plotly Figure.
    """
    if not features:
        return _apply_dark_theme(go.Figure())

    rows = []
    for f in features:
        metrics = f.get("metrics", {}).get(horizon, {})
        if not metrics:
            continue
        decay = metrics.get("decay", {})
        rows.append({
            "id": f["id"],
            "formula": f.get("formula_human", f["id"])[:25],
            "category": f.get("category", ""),
            "ic_ir": metrics.get("ic_ir", 0.0),
            "lag0": metrics.get("mean_ic", 0.0),
            "lag1": decay.get("1", None),
            "lag5": decay.get("5", None),
            "lag20": decay.get("20", None),
        })

    if not rows:
        return _apply_dark_theme(go.Figure())

    df = pd.DataFrame(rows)
    df = df.reindex(df["ic_ir"].abs().sort_values(ascending=False).index).head(top_n)

    feature_labels = df["formula"].tolist()
    cat_colors = [_get_category_color(c) for c in df["category"]]

    # Shade factor for decay bars
    def _fade(hex_color: str, alpha: float) -> str:
        """Return rgba string with reduced opacity."""
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure()

    # Lag 0 (mean_ic at lag 0)
    fig.add_trace(go.Bar(
        name="Lag 0",
        x=feature_labels,
        y=df["lag0"],
        marker_color=cat_colors,
        hovertemplate="<b>Lag 0 (Mean IC)</b><br>%{x}<br>IC: %{y:.6f}<extra></extra>",
    ))

    # Lag 1
    if df["lag1"].notna().any():
        fig.add_trace(go.Bar(
            name="Lag 1",
            x=feature_labels,
            y=df["lag1"],
            marker_color=[_fade(c, 0.75) for c in cat_colors],
            hovertemplate="<b>Lag 1</b><br>%{x}<br>IC: %{y:.6f}<extra></extra>",
        ))

    # Lag 5
    if df["lag5"].notna().any():
        fig.add_trace(go.Bar(
            name="Lag 5",
            x=feature_labels,
            y=df["lag5"],
            marker_color=[_fade(c, 0.55) for c in cat_colors],
            hovertemplate="<b>Lag 5</b><br>%{x}<br>IC: %{y:.6f}<extra></extra>",
        ))

    # Lag 20
    if df["lag20"].notna().any():
        fig.add_trace(go.Bar(
            name="Lag 20",
            x=feature_labels,
            y=df["lag20"],
            marker_color=[_fade(c, 0.35) for c in cat_colors],
            hovertemplate="<b>Lag 20</b><br>%{x}<br>IC: %{y:.6f}<extra></extra>",
        ))

    fig.update_layout(barmode="group")
    _apply_dark_theme(
        fig,
        title=f"IC Decay Analysis -- Top {top_n} Features",
        xaxis_title="Feature",
        yaxis_title="Mean IC",
        height=500,
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def plot_turnover_vs_ic(
    features: list[dict],
    horizon: str = "fwd_20d",
) -> go.Figure:
    """
    Scatter plot: mean_ic (x) vs turnover (y), colored by category.

    Args:
        features: The catalog["features"] list.
        horizon: Evaluation horizon.

    Returns:
        Plotly Figure.
    """
    if not features:
        return _apply_dark_theme(go.Figure())

    rows = []
    for f in features:
        metrics = f.get("metrics", {}).get(horizon, {})
        if not metrics:
            continue
        rows.append({
            "id": f["id"],
            "category": f.get("category", ""),
            "mean_ic": metrics.get("mean_ic", 0.0),
            "ic_ir": metrics.get("ic_ir", 0.0),
            "turnover": metrics.get("turnover", f.get("turnover", 0.0)),
        })

    if not rows:
        return _apply_dark_theme(go.Figure())

    df = pd.DataFrame(rows)
    fig = go.Figure()

    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        color = _get_category_color(cat)
        fig.add_trace(go.Scatter(
            x=sub["mean_ic"],
            y=sub["turnover"],
            mode="markers",
            name=cat,
            marker=dict(color=color, size=8, opacity=0.75),
            customdata=list(zip(sub["id"], sub["ic_ir"])),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"Category: {cat}<br>"
                "Mean IC: %{x:.6f}<br>"
                "Turnover: %{y:.4f}<br>"
                "IC_IR: %{customdata[1]:.4f}<extra></extra>"
            ),
        ))

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color=PALETTE["sub"],
        annotation_text="High turnover",
        annotation_position="top right",
    )
    fig.add_vline(x=0, line_color=PALETTE["sub"], line_width=1)

    _apply_dark_theme(
        fig,
        title="Feature IC vs Turnover Trade-off",
        xaxis_title="Mean IC",
        yaxis_title="Turnover",
    )
    return fig


def plot_train_test_scatter(
    features: list[dict],
    horizon: str = "fwd_20d",
) -> go.Figure:
    """
    Scatter plot: train IC (x) vs test IC (y) for overfit detection.

    Args:
        features: The catalog["features"] list.
        horizon: Evaluation horizon.

    Returns:
        Plotly Figure.
    """
    if not features:
        return _apply_dark_theme(go.Figure())

    rows = []
    for f in features:
        metrics = f.get("metrics", {}).get(horizon, {})
        if not metrics:
            continue
        train_ic = metrics.get("mean_ic_train")
        test_ic = metrics.get("mean_ic_test")
        if train_ic is None or test_ic is None:
            continue
        rows.append({
            "id": f["id"],
            "category": f.get("category", ""),
            "train_ic": train_ic,
            "test_ic": test_ic,
            "overfit_ratio": (test_ic / train_ic) if train_ic != 0 else None,
        })

    if not rows:
        return _apply_dark_theme(go.Figure())

    df = pd.DataFrame(rows)
    fig = go.Figure()

    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        color = _get_category_color(cat)
        fig.add_trace(go.Scatter(
            x=sub["train_ic"],
            y=sub["test_ic"],
            mode="markers",
            name=cat,
            marker=dict(color=color, size=8, opacity=0.75),
            customdata=list(zip(sub["id"], sub["overfit_ratio"].fillna(0))),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"Category: {cat}<br>"
                "Train IC: %{x:.6f}<br>"
                "Test IC: %{y:.6f}<br>"
                "Overfit Ratio: %{customdata[1]:.3f}<extra></extra>"
            ),
        ))

    # 45-degree no-overfit line
    all_vals = list(df["train_ic"]) + list(df["test_ic"])
    if all_vals:
        lo, hi = min(all_vals), max(all_vals)
        fig.add_trace(go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(color="white", dash="dash", width=1),
            name="No overfit",
            showlegend=True,
            hoverinfo="skip",
        ))

    _apply_dark_theme(
        fig,
        title="Train vs Test IC -- Overfit Detection",
        xaxis_title="Train IC",
        yaxis_title="Test IC",
    )
    return fig


def plot_category_breakdown(catalog: dict) -> go.Figure:
    """
    Horizontal bar chart with one bar per category, sorted by count descending.
    Annotates each bar with avg_ic_ir from category_summary.

    Args:
        catalog: Full catalog dict (has category_summary key).

    Returns:
        Plotly Figure.
    """
    cat_summary = catalog.get("category_summary", {})
    if not cat_summary:
        return _apply_dark_theme(go.Figure())

    rows = []
    for cat, stats in cat_summary.items():
        rows.append({
            "category": cat,
            "count": stats.get("count", 0),
            "avg_ic_ir": stats.get("avg_ic_ir", 0.0),
        })

    df = pd.DataFrame(rows).sort_values("count", ascending=True)

    colors = [_get_category_color(c) for c in df["category"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["count"],
        y=df["category"],
        orientation="h",
        marker_color=colors,
        text=[f"avg IC_IR: {v:.4f}" for v in df["avg_ic_ir"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Features: %{x}<br>"
            "Avg IC_IR: %{text}<extra></extra>"
        ),
    ))

    _apply_dark_theme(
        fig,
        title="Feature Count by Category",
        xaxis_title="Feature Count",
        height=max(350, len(df) * 30),
    )
    return fig


def plot_pruning_funnel(
    catalog: dict,
    report_text: Optional[str] = None,
) -> go.Figure:
    """
    Horizontal bar chart showing feature count at each pruning stage.

    Args:
        catalog: Full catalog dict (has total_generated, total_after_pruning).
        report_text: Optional discovery report text to parse intermediate counts.

    Returns:
        Plotly Figure.
    """
    total_gen = catalog.get("total_generated", 0)
    total_pruned = catalog.get("total_after_pruning", 0)

    stages = [("Generated", total_gen)]
    colors = ["#00D4AA"]

    # Try to parse intermediate counts from the report
    if report_text:
        patterns = {
            "After NaN Filter": r"After NaN filter[:\s]+(\d+)",
            "After IC Filter": r"After IC filter[:\s]+(\d+)",
            "After Correlation Dedup": r"After [Cc]orrelation\b.*?(\d+)",
        }
        for label, pattern in patterns.items():
            m = re.search(pattern, report_text, re.IGNORECASE)
            if m:
                stages.append((label, int(m.group(1))))

    stages.append(("After Pruning", total_pruned))

    # Assign progressively darker greens
    stage_colors = [
        "#00D4AA",  # generated
        "#00A882",  # nan filter
        "#007D61",  # ic filter
        "#005240",  # correlation
        "#003328",  # final
    ]

    labels = [s[0] for s in stages]
    values = [s[1] for s in stages]
    bar_colors = stage_colors[: len(stages)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=bar_colors,
        text=[f"{v:,}" for v in values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Features: %{x:,}<extra></extra>",
    ))

    _apply_dark_theme(
        fig,
        title="Feature Pruning Pipeline",
        xaxis_title="Feature Count",
        height=max(250, len(stages) * 50),
    )
    return fig


def plot_feature_timeseries(
    feature_df: pd.DataFrame,
    feature_id: str,
) -> go.Figure:
    """
    Time series of cross-sectional median feature value with 25th/75th percentile band.

    Args:
        feature_df: Long-format DataFrame (date, ticker, value).
        feature_id: Feature identifier for chart title.

    Returns:
        Plotly Figure.
    """
    if feature_df is None or feature_df.empty:
        return _apply_dark_theme(go.Figure())

    try:
        grouped = feature_df.groupby("date")["value"]
        median = grouped.median()
        p25 = grouped.quantile(0.25)
        p75 = grouped.quantile(0.75)
    except Exception:
        return _apply_dark_theme(go.Figure())

    fig = go.Figure()

    # Percentile band
    fig.add_trace(go.Scatter(
        x=list(p75.index) + list(p25.index[::-1]),
        y=list(p75.values) + list(p25.values[::-1]),
        fill="toself",
        fillcolor="rgba(0,212,170,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th-75th pct",
        hoverinfo="skip",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=median.index,
        y=median.values,
        mode="lines",
        name="Median",
        line=dict(color=PALETTE["pretax"], width=2),
        hovertemplate="Date: %{x}<br>Median: %{y:.4f}<extra></extra>",
    ))

    _apply_dark_theme(
        fig,
        title=f"Cross-Sectional Median -- {feature_id}",
        xaxis_title="Date",
        yaxis_title="Feature Value",
    )
    return fig


def plot_feature_histogram(
    feature_df: pd.DataFrame,
    feature_id: str,
) -> go.Figure:
    """
    Histogram of feature values across all tickers at the most recent date.

    Args:
        feature_df: Long-format DataFrame (date, ticker, value).
        feature_id: Feature identifier for chart title.

    Returns:
        Plotly Figure.
    """
    if feature_df is None or feature_df.empty:
        return _apply_dark_theme(go.Figure())

    try:
        latest_date = feature_df["date"].max()
        latest = feature_df[feature_df["date"] == latest_date]["value"].dropna()
    except Exception:
        return _apply_dark_theme(go.Figure())

    if latest.empty:
        return _apply_dark_theme(go.Figure())

    mean_val = float(latest.mean())
    median_val = float(latest.median())
    date_str = str(latest_date)[:10]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=latest.values,
        name="Distribution",
        marker_color=PALETTE["pretax"],
        opacity=0.75,
        hovertemplate="Value: %{x:.4f}<br>Count: %{y}<extra></extra>",
    ))

    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=PALETTE["ibov"],
        annotation_text=f"Mean: {mean_val:.4f}",
        annotation_position="top right",
    )
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=PALETTE["aftertax"],
        annotation_text=f"Median: {median_val:.4f}",
        annotation_position="top left",
    )

    _apply_dark_theme(
        fig,
        title=f"Cross-Sectional Distribution -- {feature_id} ({date_str})",
        xaxis_title="Feature Value",
        yaxis_title="Count",
    )
    return fig

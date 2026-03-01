"""
Plotly Chart Library
====================
Reusable Plotly chart functions for the B3 Data Pipeline UI.
Reproduces the key visualisations from the matplotlib tear sheets
with full interactivity (zoom, hover tooltips, pan).

All functions return go.Figure objects ready for st.plotly_chart().
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour Palette (mirrors backtests/core/plotting.py) ──────────────────────
PALETTE = {
    "pretax": "#00D4AA",
    "aftertax": "#7B61FF",
    "ibov": "#FF6B35",
    "tax": "#FF4C6A",
    "loss_cf": "#FFC947",
    "bg": "#0D1117",
    "panel": "#161B22",
    "grid": "#21262D",
    "text": "#E6EDF3",
    "sub": "#8B949E",
    "cdi": "#8B949E",
}

_STRATEGY_COLORS = [
    "#00D4AA", "#7B61FF", "#FF6B35", "#FFC947",
    "#FF4C6A", "#4ECDC4", "#A8E6CF", "#FFD93D",
    "#C3B1E1", "#F7DC6F", "#82E0AA", "#F1948A",
]


# ── Dark Theme Helper ─────────────────────────────────────────────────────────

def _apply_dark_theme(fig: go.Figure, **extra_layout) -> go.Figure:
    """Apply consistent dark theme to any figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PALETTE["bg"],
        plot_bgcolor=PALETTE["panel"],
        font=dict(family="monospace", color=PALETTE["text"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["grid"]),
        **extra_layout,
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["grid"])
    return fig


def _to_cumret(values: pd.Series) -> pd.Series:
    """Convert value series to cumulative return (starts at 0%)."""
    if values.empty or values.iloc[0] == 0:
        return pd.Series(dtype=float)
    return values / values.iloc[0] * 100 - 100


def _drawdown_series(values: pd.Series) -> pd.Series:
    """Compute drawdown (%) from peak."""
    if values.empty:
        return pd.Series(dtype=float)
    peak = values.cummax()
    return (values / peak - 1) * 100


# ── 1. Equity Curves ──────────────────────────────────────────────────────────

def plot_equity_curves(
    pretax_values: pd.Series,
    aftertax_values: pd.Series,
    ibov_ret: pd.Series,
    cdi_ret: Optional[pd.Series] = None,
) -> go.Figure:
    """
    Cumulative return chart: pre-tax, after-tax, IBOV, CDI.

    Args:
        pretax_values:    Equity curve (BRL), DatetimeIndex.
        aftertax_values:  After-tax equity curve (BRL), DatetimeIndex.
        ibov_ret:         IBOV period returns (not cumulative).
        cdi_ret:          CDI period returns (optional).

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # Pre-tax
    pretax_cum = _to_cumret(pretax_values)
    fig.add_trace(go.Scatter(
        x=pretax_values.index, y=pretax_cum,
        name="Pre-Tax",
        line=dict(color=PALETTE["pretax"], width=2),
        hovertemplate=(
            "<b>Pre-Tax</b><br>Date: %{x}<br>"
            "Cum. Return: %{y:.2f}%<extra></extra>"
        ),
    ))

    # After-tax
    aftertax_cum = _to_cumret(aftertax_values)
    fig.add_trace(go.Scatter(
        x=aftertax_values.index, y=aftertax_cum,
        name="After-Tax",
        line=dict(color=PALETTE["aftertax"], width=2),
        fill="tonexty",
        fillcolor="rgba(123,97,255,0.1)",
        hovertemplate=(
            "<b>After-Tax</b><br>Date: %{x}<br>"
            "Cum. Return: %{y:.2f}%<extra></extra>"
        ),
    ))

    # IBOV benchmark
    if not ibov_ret.empty:
        ibov_cum = (1 + ibov_ret).cumprod()
        ibov_cum = ibov_cum / ibov_cum.iloc[0] * 100 - 100
        fig.add_trace(go.Scatter(
            x=ibov_cum.index, y=ibov_cum,
            name="IBOV",
            line=dict(color=PALETTE["ibov"], width=1.5, dash="dot"),
            hovertemplate=(
                "<b>IBOV</b><br>Date: %{x}<br>"
                "Cum. Return: %{y:.2f}%<extra></extra>"
            ),
        ))

    # CDI (optional)
    if cdi_ret is not None and not cdi_ret.empty:
        cdi_cum = (1 + cdi_ret).cumprod()
        cdi_cum = cdi_cum / cdi_cum.iloc[0] * 100 - 100
        fig.add_trace(go.Scatter(
            x=cdi_cum.index, y=cdi_cum,
            name="CDI",
            line=dict(color=PALETTE["cdi"], width=1.5, dash="dash"),
            hovertemplate=(
                "<b>CDI</b><br>Date: %{x}<br>"
                "Cum. Return: %{y:.2f}%<extra></extra>"
            ),
        ))

    _apply_dark_theme(fig, title="Cumulative Return (%)", yaxis_title="Return (%)")
    return fig


# ── 2. Drawdown ───────────────────────────────────────────────────────────────

def plot_drawdown(
    pretax_values: pd.Series,
    aftertax_values: pd.Series,
    ibov_ret: pd.Series,
    cdi_ret: Optional[pd.Series] = None,
) -> go.Figure:
    """Drawdown chart for pre-tax, after-tax, IBOV, CDI."""
    fig = go.Figure()

    dd_pt = _drawdown_series(pretax_values)
    fig.add_trace(go.Scatter(
        x=dd_pt.index, y=dd_pt,
        name="Pre-Tax",
        line=dict(color=PALETTE["pretax"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,170,0.15)",
        hovertemplate="<b>Pre-Tax DD</b><br>Date: %{x}<br>DD: %{y:.2f}%<extra></extra>",
    ))

    dd_at = _drawdown_series(aftertax_values)
    fig.add_trace(go.Scatter(
        x=dd_at.index, y=dd_at,
        name="After-Tax",
        line=dict(color=PALETTE["aftertax"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(123,97,255,0.15)",
        hovertemplate="<b>After-Tax DD</b><br>Date: %{x}<br>DD: %{y:.2f}%<extra></extra>",
    ))

    # IBOV
    if not ibov_ret.empty:
        ibov_cum = (1 + ibov_ret).cumprod()
        dd_ibov = _drawdown_series(ibov_cum)
        fig.add_trace(go.Scatter(
            x=dd_ibov.index, y=dd_ibov,
            name="IBOV",
            line=dict(color=PALETTE["ibov"], width=1, dash="dot"),
            hovertemplate="<b>IBOV DD</b><br>Date: %{x}<br>DD: %{y:.2f}%<extra></extra>",
        ))

    if cdi_ret is not None and not cdi_ret.empty:
        cdi_cum = (1 + cdi_ret).cumprod()
        dd_cdi = _drawdown_series(cdi_cum)
        fig.add_trace(go.Scatter(
            x=dd_cdi.index, y=dd_cdi,
            name="CDI",
            line=dict(color=PALETTE["cdi"], width=1, dash="dash"),
            hovertemplate="<b>CDI DD</b><br>Date: %{x}<br>DD: %{y:.2f}%<extra></extra>",
        ))

    _apply_dark_theme(fig, title="Drawdown (%)", yaxis_title="Drawdown (%)")
    return fig


# ── 3. Tax Detail ─────────────────────────────────────────────────────────────

def plot_tax_detail(
    tax_paid: pd.Series,
    loss_cf: pd.Series,
    turnover: pd.Series,
) -> go.Figure:
    """Three-panel chart: tax paid bars, loss carryforward area, turnover bars."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Tax Paid (BRL)", "Loss Carryforward (BRL)", "Portfolio Turnover"),
        shared_xaxes=False,
    )

    # Tax paid
    fig.add_trace(go.Bar(
        x=tax_paid.index, y=tax_paid,
        name="Tax Paid",
        marker_color=PALETTE["tax"],
        hovertemplate="<b>Tax</b><br>Date: %{x}<br>Amount: R$%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    # Loss carryforward
    fig.add_trace(go.Scatter(
        x=loss_cf.index, y=loss_cf,
        name="Loss CF",
        line=dict(color=PALETTE["loss_cf"], width=2),
        fill="tozeroy",
        fillcolor="rgba(255,201,71,0.2)",
        hovertemplate="<b>Loss CF</b><br>Date: %{x}<br>Amount: R$%{y:,.0f}<extra></extra>",
    ), row=1, col=2)

    # Turnover
    fig.add_trace(go.Bar(
        x=turnover.index, y=turnover * 100,
        name="Turnover",
        marker_color=PALETTE["sub"],
        hovertemplate="<b>Turnover</b><br>Date: %{x}<br>%{y:.1f}%<extra></extra>",
    ), row=1, col=3)

    _apply_dark_theme(fig, title="Tax Detail")
    return fig


# ── 4. Tax Drag ───────────────────────────────────────────────────────────────

def plot_tax_drag(
    pretax_values: pd.Series,
    aftertax_values: pd.Series,
    total_tax_brl: float,
) -> go.Figure:
    """Cumulative tax drag spread between pre-tax and after-tax equity curves."""
    drag = pretax_values - aftertax_values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drag.index, y=drag,
        name="Tax Drag (BRL)",
        line=dict(color=PALETTE["tax"], width=2),
        fill="tozeroy",
        fillcolor="rgba(255,76,106,0.15)",
        hovertemplate="<b>Tax Drag</b><br>Date: %{x}<br>R$%{y:,.0f}<extra></extra>",
    ))
    fig.add_annotation(
        text=f"Total Tax Paid: R${total_tax_brl:,.0f}",
        xref="paper", yref="paper",
        x=0.02, y=0.95,
        showarrow=False,
        font=dict(color=PALETTE["tax"], size=12),
    )
    _apply_dark_theme(fig, title="Cumulative Tax Drag", yaxis_title="BRL")
    return fig


# ── 5. Metrics Table ──────────────────────────────────────────────────────────

def plot_metrics_table(metrics_list: list[dict]) -> go.Figure:
    """Render performance metrics as a Plotly table figure."""
    if not metrics_list:
        return go.Figure()

    df = pd.DataFrame(metrics_list)
    header_vals = list(df.columns)
    cell_vals = [df[col].tolist() for col in df.columns]

    # Build format list: None for string columns, ",.2f" for numeric columns
    col_formats = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_formats.append(",.2f")
        else:
            col_formats.append(None)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_vals,
            fill_color=PALETTE["panel"],
            font=dict(color=PALETTE["text"]),
            align="left",
        ),
        cells=dict(
            values=cell_vals,
            fill_color=PALETTE["bg"],
            font=dict(color=PALETTE["text"]),
            align="left",
            format=col_formats,
        ),
    )])
    _apply_dark_theme(fig)
    return fig


# ── 6. Correlation Heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(returns_df: pd.DataFrame) -> go.Figure:
    """
    Lower-triangle correlation heatmap with annotated values.

    Args:
        returns_df: DataFrame with strategy names as columns, monthly returns as values.
    """
    corr = returns_df.corr()
    n = len(corr)

    # Mask upper triangle
    z = corr.values.copy().astype(float)
    for i in range(n):
        for j in range(i + 1, n):
            z[i, j] = np.nan

    # Text annotations
    annotations = []
    for i in range(n):
        for j in range(i + 1):
            annotations.append(dict(
                x=corr.columns[j],
                y=corr.index[i],
                text=f"{z[i, j]:.2f}",
                showarrow=False,
                font=dict(color=PALETTE["text"], size=11),
            ))

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(annotations=annotations)
    _apply_dark_theme(fig, title="Strategy Correlation Matrix")
    return fig


# ── 7. Strategy Comparison ────────────────────────────────────────────────────

def plot_strategy_comparison(equity_curves: dict[str, pd.Series]) -> go.Figure:
    """
    Overlay multiple strategy equity curves (normalised to 1.0 start) on one chart.

    Args:
        equity_curves: {strategy_name: pd.Series (normalised to 1.0)}.
    """
    fig = go.Figure()
    for i, (name, curve) in enumerate(equity_curves.items()):
        color = _STRATEGY_COLORS[i % len(_STRATEGY_COLORS)]
        pct = curve * 100 - 100
        fig.add_trace(go.Scatter(
            x=pct.index, y=pct,
            name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name}</b><br>Date: %{{x}}<br>Return: %{{y:.2f}}%<extra></extra>",
        ))
    _apply_dark_theme(
        fig,
        title="Strategy Comparison (Normalised)",
        yaxis_title="Cumulative Return (%)",
    )
    return fig


# ── 8. Parameter Sensitivity Heatmap ─────────────────────────────────────────

def plot_param_heatmap(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str,
    title: str = "Parameter Sensitivity",
) -> go.Figure:
    """
    2D heatmap for parameter sensitivity analysis.

    Args:
        results_df: DataFrame with columns including param_x, param_y, and metric.
        param_x:    Column name for x-axis parameter.
        param_y:    Column name for y-axis parameter.
        metric:     Column name for the metric to display (e.g. 'Sharpe').
        title:      Chart title.
    """
    pivot = results_df.pivot(index=param_y, columns=param_x, values=metric)

    annotations = []
    for i, yi in enumerate(pivot.index):
        for j, xi in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                annotations.append(dict(
                    x=str(xi), y=str(yi),
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(color=PALETTE["text"], size=10),
                ))

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=[str(r) for r in pivot.index],
        colorscale="RdYlGn",
        hovertemplate=(
            f"<b>{param_x}</b>: %{{x}}<br>"
            f"<b>{param_y}</b>: %{{y}}<br>"
            f"<b>{metric}</b>: %{{z:.3f}}<extra></extra>"
        ),
    ))
    fig.update_layout(
        annotations=annotations,
        xaxis_title=param_x,
        yaxis_title=param_y,
    )
    _apply_dark_theme(fig, title=title)
    return fig

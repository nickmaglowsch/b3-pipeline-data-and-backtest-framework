"""
Parameter Sensitivity Scanner Framework
=========================================
Provides a generic framework for sweeping strategy parameters over a 2D grid
and visualizing the results as heatmaps.

Functions:
    scan_parameters()    -- Run a strategy across all parameter combinations
    plot_param_heatmap() -- Plot a 2D heatmap of a metric vs two parameters
"""

import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKTESTS_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKTESTS_DIR)
for _p in [_PROJECT_ROOT, _BACKTESTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.simulation import run_simulation
from core.metrics import build_metrics, value_to_ret

try:
    from core.plotting import PALETTE
except Exception:
    PALETTE = {
        "bg": "#0D1117",
        "panel": "#161B22",
        "grid": "#21262D",
        "text": "#E6EDF3",
        "sub": "#8B949E",
        "pretax": "#00D4AA",
        "aftertax": "#7B61FF",
        "ibov": "#FF6B35",
        "tax": "#FF4C6A",
        "loss_cf": "#FFC947",
    }


def scan_parameters(
    signal_fn,
    param_grid: dict,
    shared_data: dict,
    sim_config: dict,
) -> pd.DataFrame:
    """
    Run a strategy over all combinations of parameters in the grid.

    Parameters
    ----------
    signal_fn : callable(params_dict, shared_data) -> (returns_matrix, target_weights)
        A function that accepts a dict of parameter values and shared precomputed
        data, and returns the (returns_matrix, target_weights) tuple needed by
        run_simulation().
    param_grid : dict
        Mapping of parameter name -> list of values to sweep.
        E.g. {"lookback": [6, 9, 12], "top_pct": [0.05, 0.10, 0.15]}
    shared_data : dict
        Precomputed DataFrames passed to signal_fn (e.g. ret, adtv, log_ret).
    sim_config : dict
        Simulation parameters: capital, tax_rate, slippage, monthly_sales_exemption.

    Returns
    -------
    pd.DataFrame with columns: [param_names..., "sharpe", "ann_return",
                                 "max_dd", "avg_turnover"]
    Rows = one per parameter combination.
    """
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combos = list(itertools.product(*param_values))

    records = []
    total = len(combos)

    for i, combo in enumerate(combos):
        params = dict(zip(param_names, combo))
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        print(f"  [{i+1:3d}/{total}] {param_str}")

        try:
            ret_matrix, target_weights = signal_fn(params, shared_data)
            result = run_simulation(
                returns_matrix=ret_matrix.fillna(0.0),
                target_weights=target_weights,
                initial_capital=sim_config.get("capital", 100_000),
                tax_rate=sim_config.get("tax_rate", 0.15),
                slippage=sim_config.get("slippage", 0.001),
                monthly_sales_exemption=sim_config.get("monthly_sales_exemption", 20_000),
                name=param_str,
            )

            at_val = result["aftertax_values"]
            at_ret = value_to_ret(at_val).dropna()

            if len(at_ret) < 12:
                raise ValueError("Too few periods in result")

            m = build_metrics(at_ret, param_str, 12)
            turnover_series = result.get("turnover", pd.Series(dtype=float))
            avg_turnover = float(turnover_series.mean()) if len(turnover_series) > 0 else 0.0

            record = dict(params)
            record["sharpe"] = float(m.get("Sharpe", 0.0))
            record["ann_return"] = float(m.get("Ann. Return (%)", 0.0))
            record["max_dd"] = float(m.get("Max Drawdown (%)", 0.0))
            record["avg_turnover"] = round(avg_turnover * 100, 2)

        except Exception as exc:
            print(f"    WARNING: failed -- {exc}")
            record = dict(params)
            record["sharpe"] = np.nan
            record["ann_return"] = np.nan
            record["max_dd"] = np.nan
            record["avg_turnover"] = np.nan

        records.append(record)

    return pd.DataFrame(records)


def plot_param_heatmap(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str,
    title: str,
    out_path: str,
    ax=None,
    fig=None,
    robust_threshold: float = 0.80,
) -> list:
    """
    Plot a 2D heatmap of a metric across two parameter dimensions.

    Parameters
    ----------
    results_df : DataFrame
        Output from scan_parameters(); must contain param_x, param_y, and metric columns.
    param_x : str
        Column name for the x-axis parameter.
    param_y : str
        Column name for the y-axis parameter.
    metric : str
        Metric to plot: "sharpe", "ann_return", or "max_dd".
    title : str
        Plot title.
    out_path : str
        If ax is None, save standalone figure to this path.
    ax : matplotlib Axes, optional
        If provided, draw into this axes (for subplot layouts).
    robust_threshold : float
        Fraction of max metric value used to identify the "robust zone".
    """
    x_vals = sorted(results_df[param_x].unique())
    y_vals = sorted(results_df[param_y].unique())

    # Build pivot
    pivot = results_df.pivot_table(
        index=param_y, columns=param_x, values=metric, aggfunc="mean"
    )
    pivot = pivot.reindex(index=y_vals, columns=x_vals)

    data = pivot.values.astype(float)

    # Color maps: green=good for sharpe/return, green=small for max_dd
    if metric == "max_dd":
        cmap = "RdYlGn_r"
    else:
        cmap = "RdYlGn"

    standalone = ax is None
    if standalone:
        plt.rcParams.update({
            "font.family": "monospace",
            "figure.facecolor": PALETTE["bg"],
            "text.color": PALETTE["text"],
        })
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor(PALETTE["bg"])

    ax.set_facecolor(PALETTE["panel"])

    # Mask NaN
    masked = np.ma.masked_invalid(data)
    vmin = np.nanmin(data) if not np.all(np.isnan(data)) else 0
    vmax = np.nanmax(data) if not np.all(np.isnan(data)) else 1

    im = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Annotate cells
    for (yi, xi), val in np.ndenumerate(data):
        if np.isnan(val):
            txt = "n/a"
            text_color = "white"
        else:
            txt = f"{val:.2f}"
            norm = (val - vmin) / max(vmax - vmin, 1e-9)
            text_color = "black" if 0.3 < norm < 0.9 else "white"
        ax.text(
            xi, yi, txt,
            ha="center", va="center", fontsize=8,
            color=text_color,
            fontweight="bold",
        )

    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(v) for v in x_vals], color=PALETTE["sub"])
    ax.set_yticks(range(len(y_vals)))
    ax.set_yticklabels([str(v) for v in y_vals], color=PALETTE["sub"])
    ax.set_xlabel(param_x, color=PALETTE["sub"])
    ax.set_ylabel(param_y, color=PALETTE["sub"])
    ax.set_title(title, color=PALETTE["text"], fontsize=9, pad=6)
    ax.spines[:].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["sub"])

    if standalone:
        plt.colorbar(im, ax=ax).ax.yaxis.set_tick_params(color=PALETTE["sub"])
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        plt.close()
        print(f"  Heatmap saved -> {out_path}")

    # Identify robust zone: cells within threshold of peak
    if metric != "max_dd":
        peak = np.nanmax(data)
        robust_floor = peak * robust_threshold
        robust_cells = [
            (y_vals[yi], x_vals[xi])
            for (yi, xi), val in np.ndenumerate(data)
            if not np.isnan(val) and val >= robust_floor
        ]
    else:
        # For max_dd: robust = closest to 0
        best = np.nanmin(np.abs(data))
        robust_floor = best / robust_threshold if robust_threshold > 0 else best
        robust_cells = [
            (y_vals[yi], x_vals[xi])
            for (yi, xi), val in np.ndenumerate(data)
            if not np.isnan(val) and abs(val) <= abs(robust_floor)
        ]

    return robust_cells


def identify_robust_zone(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str = "sharpe",
    threshold: float = 0.80,
) -> str:
    """
    Return a text summary of the robust parameter zone.

    A parameter combination is 'robust' if its metric value is within
    `threshold` (80%) of the peak metric value.
    """
    x_vals = sorted(results_df[param_x].unique())
    y_vals = sorted(results_df[param_y].unique())

    pivot = results_df.pivot_table(
        index=param_y, columns=param_x, values=metric, aggfunc="mean"
    )
    pivot = pivot.reindex(index=y_vals, columns=x_vals)
    data = pivot.values.astype(float)

    if metric != "max_dd":
        peak = np.nanmax(data)
        robust_floor = peak * threshold
        robust_mask = data >= robust_floor
    else:
        best = np.nanmin(np.abs(data))
        robust_mask = np.abs(data) <= abs(best) / threshold

    robust_x = sorted(set(x_vals[xi] for yi, xi in zip(*np.where(robust_mask))))
    robust_y = sorted(set(y_vals[yi] for yi, xi in zip(*np.where(robust_mask))))

    lines = [
        f"  Metric: {metric}",
        f"  Peak: {np.nanmax(data) if metric != 'max_dd' else np.nanmax(data):.3f}",
        f"  Robust {param_x} values (within {threshold*100:.0f}% of peak): {robust_x}",
        f"  Robust {param_y} values (within {threshold*100:.0f}% of peak): {robust_y}",
    ]
    return "\n".join(lines)

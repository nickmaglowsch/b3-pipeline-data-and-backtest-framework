"""
Core plotting utilities for backtesting.
"""

import matplotlib.dates as mdates

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
}


def fmt_ax(ax, ylabel=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["sub"], labelsize=8.5)
    ax.spines[:].set_color(PALETTE["grid"])
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE["sub"], fontsize=9)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.6, ls="--")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.3, ls=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))



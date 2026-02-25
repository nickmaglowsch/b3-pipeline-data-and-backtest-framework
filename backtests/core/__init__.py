"""
Core backtesting components.
"""

from .data import load_b3_data, download_benchmark, download_cdi_daily
from .metrics import build_metrics, value_to_ret, display_metrics_table
from .plotting import plot_tax_backtest
from .portfolio import rebalance_positions, apply_returns, compute_tax

__all__ = [
    "load_b3_data",
    "download_benchmark",
    "download_cdi_daily",
    "build_metrics",
    "value_to_ret",
    "display_metrics_table",
    "plot_tax_backtest",
    "rebalance_positions",
    "apply_returns",
    "compute_tax",
]

"""
Core backtesting components.
"""

from .data import load_b3_data, load_b3_hlc_data, download_benchmark, download_cdi_daily
from .metrics import build_metrics, value_to_ret, display_metrics_table
from .simulation import run_simulation

__all__ = [
    "load_b3_data",
    "load_b3_hlc_data",
    "download_benchmark",
    "download_cdi_daily",
    "build_metrics",
    "value_to_ret",
    "display_metrics_table",
    "run_simulation",
]

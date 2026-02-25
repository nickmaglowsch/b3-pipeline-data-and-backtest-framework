"""
Core metrics and evaluation utilities for backtesting.
"""

import numpy as np
import pandas as pd


def cumret(ret: pd.Series) -> pd.Series:
    return (1 + ret).cumprod()


def value_to_ret(values: pd.Series) -> pd.Series:
    """Convert BRL equity curve to simple period returns."""
    return values.pct_change().fillna(0)


def ann_return(ret: pd.Series, periods_per_year: int = 12) -> float:
    n = len(ret) / periods_per_year
    return (1 + ret).prod() ** (1 / n) - 1 if n > 0 else 0.0


def ann_vol(ret: pd.Series, periods_per_year: int = 12) -> float:
    return ret.std() * np.sqrt(periods_per_year)


def sharpe(ret: pd.Series, risk_free: float = 0.0, periods_per_year: int = 12) -> float:
    mean_ret = ret.mean() - (risk_free / periods_per_year)
    return (mean_ret / ret.std()) * np.sqrt(periods_per_year) if ret.std() != 0 else 0.0


def max_dd(ret: pd.Series) -> float:
    cum = cumret(ret)
    return (cum / cum.cummax() - 1).min()


def calmar(ret: pd.Series, periods_per_year: int = 12) -> float:
    mdd = abs(max_dd(ret))
    return ann_return(ret, periods_per_year) / mdd if mdd != 0 else 0.0


def build_metrics(ret: pd.Series, label: str, periods_per_year: int = 12) -> dict:
    """Standardized performance dictionary"""
    return {
        "Strategy": label,
        "Ann. Return (%)": round(ann_return(ret, periods_per_year) * 100, 2),
        "Ann. Volatility (%)": round(ann_vol(ret, periods_per_year) * 100, 2),
        "Sharpe": round(sharpe(ret, periods_per_year=periods_per_year), 2),
        "Max Drawdown (%)": round(max_dd(ret) * 100, 2),
        "Calmar": round(calmar(ret, periods_per_year), 2),
    }


def display_metrics_table(metrics_list: list):
    """Print performance metrics to console"""
    n_cols = len(metrics_list)
    dash_length = 30 + 16 * n_cols

    print("\n" + "-" * dash_length)

    headers = ["Metric"] + [m["Strategy"] for m in metrics_list]
    header_fmt = "  {0:<26}  " + "  ".join(
        [f"{{{i}:>14}}" for i in range(1, n_cols + 1)]
    )
    print(header_fmt.format(*headers))
    print("-" * dash_length)

    col_labels = list(metrics_list[0].keys())
    for key in col_labels:
        if key == "Strategy":
            continue
        vals = [key] + [str(m[key]) for m in metrics_list]
        print(header_fmt.format(*vals))
    print("-" * dash_length + "\n")

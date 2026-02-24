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


def ann_return(ret: pd.Series) -> float:
    n = len(ret) / 12
    return (1 + ret).prod() ** (1 / n) - 1 if n > 0 else 0.0


def ann_vol(ret: pd.Series) -> float:
    return ret.std() * np.sqrt(12)


def sharpe(ret: pd.Series, risk_free: float = 0.0) -> float:
    mean_ret = ret.mean() - (risk_free / 12)
    return (mean_ret / ret.std()) * np.sqrt(12) if ret.std() != 0 else 0.0


def max_dd(ret: pd.Series) -> float:
    cum = cumret(ret)
    return (cum / cum.cummax() - 1).min()


def calmar(ret: pd.Series) -> float:
    mdd = abs(max_dd(ret))
    return ann_return(ret) / mdd if mdd != 0 else 0.0


def build_metrics(ret: pd.Series, label: str) -> dict:
    """Standardized performance dictionary"""
    return {
        "Strategy": label,
        "Ann. Return (%)": round(ann_return(ret) * 100, 2),
        "Ann. Volatility (%)": round(ann_vol(ret) * 100, 2),
        "Sharpe": round(sharpe(ret), 2),
        "Max Drawdown (%)": round(max_dd(ret) * 100, 2),
        "Calmar": round(calmar(ret), 2),
    }


def display_metrics_table(metrics_list: list):
    """Print performance metrics to console"""
    print("\n" + "-" * 65)
    print(
        f"  {'Metric':<26}  {metrics_list[0]['Strategy']:>12}  {metrics_list[1]['Strategy']:>12}  {metrics_list[2]['Strategy']:>8}"
    )
    print("-" * 65)

    col_labels = list(metrics_list[0].keys())
    for key in col_labels:
        if key == "Strategy":
            continue
        m1 = str(metrics_list[0][key])
        m2 = str(metrics_list[1][key])
        m3 = str(metrics_list[2][key])
        print(f"  {key:<26}  {m1:>12}  {m2:>12}  {m3:>8}")
    print("-" * 65 + "\n")

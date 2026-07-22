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


def reconstruct_daily_values(
    target_weights: pd.DataFrame,
    daily_ret: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """Daily mark-to-market NAV implied by rebalance-date target weights.

    run_simulation records NAV only on rebalance dates, so max_dd / ann_vol are
    blind to intra-period lows — a quarterly curve never sees a mid-quarter
    crash, and even a monthly/weekly curve misses the trough between marks. This
    reconstructs the buy-and-hold daily path: on each rebalance date the
    portfolio is reset to that row's target weights, then drifts with daily
    asset returns until the next rebalance. Tax and slippage are ignored — they
    bite only on rebalance dates and are second-order for a risk-path estimate.

    ``daily_ret`` must carry a column for every ticker with a non-zero weight
    (missing columns are treated as flat / zero return, e.g. an uninvested
    cash residual).
    """
    cols = [c for c in target_weights.columns if c in daily_ret.columns]
    if not cols:
        return pd.Series(dtype=float)
    # Guard against silent wrongness: if any held sleeve has no daily series
    # (e.g. a downloaded ETF the caller didn't supply), bail so the caller falls
    # back to cadence DD rather than a path that zeros that sleeve.
    missing = [c for c in target_weights.columns if c not in daily_ret.columns]
    if missing and float(target_weights[missing].abs().sum(axis=1).max()) > 1e-4:
        return pd.Series(dtype=float)
    tw = target_weights[cols].fillna(0.0)
    live = tw.index[tw.abs().sum(axis=1) > 0]
    if len(live) == 0:
        return pd.Series(dtype=float)
    tw = tw.loc[live[0]:]
    reb = tw.index
    dr = daily_ret[cols]

    segments = [pd.Series([float(initial_capital)], index=[reb[0]])]
    nav = float(initial_capital)
    for i, d0 in enumerate(reb):
        d1 = reb[i + 1] if i + 1 < len(reb) else dr.index[-1]
        # Trading days strictly after this rebalance, up to and including the
        # next one — a clean partition even when d0/d1 are calendar month-ends
        # (resample labels) that aren't themselves trading days.
        seg = dr.loc[(dr.index > d0) & (dr.index <= d1)]
        if seg.empty:
            continue
        w = tw.loc[d0].to_numpy()
        resid = 1.0 - float(w.sum())     # uninvested fraction: hold flat as cash,
                                         # don't vaporize it (else NAV fake-drops
                                         # to Σw at every not-fully-invested rebalance)
        # Clip daily returns to the simulator's own guard band (_apply_returns)
        # so an unrecorded-split print can't crater the reconstructed path.
        growth = (1.0 + seg.fillna(0.0).clip(-0.90, 5.0)).cumprod()   # per-asset cum growth
        seg_nav = pd.Series(((growth.to_numpy() * w).sum(axis=1) + resid) * nav, index=seg.index)
        segments.append(seg_nav)
        nav = float(seg_nav.iloc[-1])

    out = pd.concat(segments)
    return out[~out.index.duplicated(keep="first")]


def strategy_daily_values(
    shared: dict,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000.0,
) -> pd.Series:
    """Daily NAV path for a strategy's rebalance-date weights, built from the
    daily price data already in ``shared``.

    Single source of truth so Max Drawdown / Calmar reflect intra-rebalance lows
    for EVERY strategy and rebalance regime — the rebalance-cadence equity curve
    from run_simulation is blind to mid-period crashes (see
    :func:`reconstruct_daily_values`). Returns an empty Series if the required
    daily data isn't present, in which case callers fall back to cadence DD.
    """
    daily_close = shared.get("split_adj_close")
    if daily_close is None:
        return pd.Series(dtype=float)
    daily_ret = daily_close.pct_change()
    cdi_daily = shared.get("cdi_daily")
    if cdi_daily is not None:
        daily_ret["CDI_ASSET"] = cdi_daily                 # cash sleeve
    ibov_px = shared.get("ibov_px")
    if "IBOV" in target_weights.columns and ibov_px is not None:
        daily_ret["IBOV"] = ibov_px.pct_change()
    # Daily returns for downloaded ETF sleeves (IVVB11, DIVO11, …) that the
    # FixedWeight blend engine stashed here — not in the B3 price matrix.
    for col, series in (shared.get("_daily_asset_ret") or {}).items():
        daily_ret[col] = series
    return reconstruct_daily_values(target_weights, daily_ret, initial_capital)


def build_metrics(
    ret: pd.Series,
    label: str,
    periods_per_year: int = 12,
    daily_values: pd.Series | None = None,
) -> dict:
    """Standardized performance dictionary.

    When ``daily_values`` (a daily NAV path, e.g. from
    :func:`reconstruct_daily_values`) is supplied, Max Drawdown and Calmar are
    computed on it so they reflect intra-rebalance lows; otherwise they fall
    back to the rebalance-cadence ``ret`` (which understates drawdown). Return,
    volatility and Sharpe always use ``ret`` (annualized-from-periodic is the
    standard convention).
    """
    if daily_values is not None and daily_values.dropna().shape[0] > 1:
        mdd = max_dd(value_to_ret(daily_values.dropna()))
    else:
        mdd = max_dd(ret)
    ar = ann_return(ret, periods_per_year)
    cal = ar / abs(mdd) if mdd != 0 else 0.0
    return {
        "Strategy": label,
        "Ann. Return (%)": round(ar * 100, 2),
        "Ann. Volatility (%)": round(ann_vol(ret, periods_per_year) * 100, 2),
        "Sharpe": round(sharpe(ret, periods_per_year=periods_per_year), 2),
        "Max Drawdown (%)": round(mdd * 100, 2),
        "Calmar": round(cal, 2),
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

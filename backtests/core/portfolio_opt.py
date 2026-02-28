"""
Portfolio Optimization Utilities
==================================
Reusable weight computation functions for meta-portfolio construction over
strategy return streams. All functions accept a DataFrame of monthly returns
(rows=dates, cols=strategy_names) and return a pd.Series of weights summing to 1.0.

Functions:
    inverse_vol_weights()            -- Naive risk parity (inverse volatility)
    equal_risk_contribution_weights() -- Full ERC risk parity (scipy solver)
    hrp_weights()                    -- Hierarchical Risk Parity (Lopez de Prado)
    rolling_sharpe_weights()         -- Weight proportional to trailing Sharpe
    regime_conditional_weights()     -- Pre-defined mixes based on macro regime

All functions fall back to equal-weight when there is insufficient history.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

# ─────────────────────────────────────────────────────────────────────────────
#  Risk Parity Utilities
# ─────────────────────────────────────────────────────────────────────────────

def equal_weights(n: int, columns) -> pd.Series:
    """Return an equal-weight Series."""
    return pd.Series(1.0 / n, index=columns)


def inverse_vol_weights(returns_df: pd.DataFrame, lookback: int = 36) -> pd.Series:
    """
    Naive risk parity: weight inversely proportional to trailing volatility.

    Parameters
    ----------
    returns_df : DataFrame
        Monthly return series; columns = strategy names.
    lookback : int
        Number of trailing months to use for volatility estimation.

    Returns
    -------
    pd.Series of weights summing to 1.0 (all non-negative).
    Falls back to equal-weight if fewer than 2 observations are available
    for any strategy.
    """
    window = returns_df.tail(lookback)
    if len(window) < 2:
        return equal_weights(len(returns_df.columns), returns_df.columns)

    vols = window.std()
    # Replace zero or NaN vols with the mean vol to avoid division by zero
    mean_vol = vols[vols > 0].mean()
    if np.isnan(mean_vol) or mean_vol == 0:
        return equal_weights(len(returns_df.columns), returns_df.columns)
    vols = vols.replace(0, mean_vol).fillna(mean_vol)

    inv_vols = 1.0 / vols
    weights = inv_vols / inv_vols.sum()
    return weights


def equal_risk_contribution_weights(
    returns_df: pd.DataFrame, lookback: int = 36
) -> pd.Series:
    """
    Full risk parity: solve for weights where each asset's marginal risk
    contribution is equal (Equal Risk Contribution / ERC portfolio).

    Uses scipy.optimize.minimize with SLSQP to solve:
        min  sum_i (RC_i - RC_target)^2
        where RC_i = w_i * (Sigma @ w)_i / sqrt(w' Sigma w)

    Parameters
    ----------
    returns_df : DataFrame
        Monthly return series; columns = strategy names.
    lookback : int
        Trailing months for covariance estimation.

    Returns
    -------
    pd.Series of weights summing to 1.0 (all non-negative).
    Falls back to inverse_vol_weights() if the solver fails.
    """
    window = returns_df.tail(lookback)
    if len(window) < 3:
        return inverse_vol_weights(returns_df, lookback)

    cov = window.cov().values
    n = cov.shape[0]
    cols = returns_df.columns

    # Add a small regularization term to avoid singular covariance
    cov += np.eye(n) * 1e-8

    def risk_contribution(w, cov_matrix):
        port_var = w @ cov_matrix @ w
        if port_var <= 0:
            return np.zeros(n)
        marginal = cov_matrix @ w
        rc = w * marginal / np.sqrt(port_var)
        return rc

    def objective(w):
        rc = risk_contribution(w, cov)
        target = rc.sum() / n
        return np.sum((rc - target) ** 2)

    # Initial guess: inverse vol weights
    w0 = inverse_vol_weights(returns_df, lookback).values

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-6, 1.0)] * n

    try:
        res = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 500},
        )
        if res.success and np.all(np.array(res.x) >= 0):
            w_opt = np.array(res.x)
            w_opt = np.maximum(w_opt, 0)
            w_opt /= w_opt.sum()
            return pd.Series(w_opt, index=cols)
    except Exception:
        pass

    # Fallback
    return inverse_vol_weights(returns_df, lookback)


# ─────────────────────────────────────────────────────────────────────────────
#  Hierarchical Risk Parity
# ─────────────────────────────────────────────────────────────────────────────

def _get_cluster_var(cov: np.ndarray, cluster_items: list) -> float:
    """Compute variance of a cluster (equal-weight within cluster)."""
    cov_slice = cov[np.ix_(cluster_items, cluster_items)]
    n = len(cluster_items)
    w = np.ones(n) / n
    return float(w @ cov_slice @ w)


def _recursive_bisect(sorted_items: list, cov: np.ndarray) -> np.ndarray:
    """
    Recursively bisect the sorted list of assets and allocate weight
    proportional to inverse cluster variance at each split.
    """
    n = len(sorted_items)
    weights = np.ones(n)

    def _bisect(items):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]
        var_left = _get_cluster_var(cov, left)
        var_right = _get_cluster_var(cov, right)
        total = var_left + var_right
        if total == 0:
            alloc_left = 0.5
        else:
            alloc_left = var_right / total  # inverse variance
        alloc_right = 1.0 - alloc_left
        weights[left] *= alloc_left
        weights[right] *= alloc_right
        _bisect(left)
        _bisect(right)

    _bisect(list(range(n)))
    return weights


def hrp_weights(returns_df: pd.DataFrame, lookback: int = 36) -> pd.Series:
    """
    Hierarchical Risk Parity (Lopez de Prado 2016).

    Steps:
      1. Correlation -> distance matrix: d(i,j) = sqrt(0.5 * (1 - corr(i,j)))
      2. Hierarchical clustering (Ward linkage)
      3. Quasi-diagonalization via dendrogram leaf ordering
      4. Recursive bisection with inverse-variance allocation

    Parameters
    ----------
    returns_df : DataFrame
        Monthly return series; columns = strategy names.
    lookback : int
        Trailing months for correlation/covariance estimation.

    Returns
    -------
    pd.Series of weights summing to 1.0 (all non-negative).
    Falls back to equal-weight if insufficient data.
    """
    window = returns_df.tail(lookback)
    cols = returns_df.columns
    n = len(cols)

    if len(window) < max(3, n):
        return equal_weights(n, cols)

    corr = window.corr().values
    cov = window.cov().values

    # Ensure no NaNs in the correlation matrix
    if np.any(np.isnan(corr)):
        return inverse_vol_weights(returns_df, lookback)

    # Clip diagonal to exactly 1 (floating point)
    np.fill_diagonal(corr, 1.0)

    # Distance matrix
    dist = np.sqrt(np.maximum(0.5 * (1.0 - corr), 0.0))
    np.fill_diagonal(dist, 0.0)

    # Condensed form for scipy
    condensed = squareform(dist, checks=False)

    try:
        linkage_matrix = linkage(condensed, method="ward")
        order = leaves_list(linkage_matrix)
    except Exception:
        return inverse_vol_weights(returns_df, lookback)

    # Reorder covariance matrix according to dendrogram
    sorted_cov = cov[np.ix_(order, order)]

    # Recursive bisection on sorted indices
    w_sorted = _recursive_bisect(list(range(n)), sorted_cov)

    # Map back to original column order
    weights = np.zeros(n)
    for rank_i, orig_i in enumerate(order):
        weights[orig_i] = w_sorted[rank_i]

    # Normalize (should already sum to 1 but be safe)
    total = weights.sum()
    if total <= 0:
        return equal_weights(n, cols)
    weights /= total

    return pd.Series(weights, index=cols)


# ─────────────────────────────────────────────────────────────────────────────
#  Dynamic Allocation
# ─────────────────────────────────────────────────────────────────────────────

def rolling_sharpe_weights(
    returns_df: pd.DataFrame,
    lookback: int = 12,
    min_sharpe: float = 0.0,
    cdi_col: str = "CDI",
) -> pd.Series:
    """
    Weight each strategy proportional to its trailing Sharpe ratio.
    Strategies with Sharpe < min_sharpe get zero weight.
    If CDI column exists, any residual weight (from excluded strategies)
    is allocated to CDI.

    Parameters
    ----------
    returns_df : DataFrame
        Monthly return series; columns include strategy names and optionally CDI.
    lookback : int
        Trailing months for Sharpe computation.
    min_sharpe : float
        Strategies with trailing Sharpe below this get zero weight.
    cdi_col : str
        Column name for CDI (safe haven when strategies fail Sharpe filter).

    Returns
    -------
    pd.Series of weights summing to 1.0 (all non-negative).
    """
    window = returns_df.tail(lookback)
    cols = returns_df.columns

    if len(window) < 3:
        return equal_weights(len(cols), cols)

    sharpes = {}
    for col in cols:
        s = window[col].dropna()
        if len(s) < 3 or s.std() == 0:
            sharpes[col] = 0.0
        else:
            sharpes[col] = float(s.mean() / s.std() * np.sqrt(12))

    # Filter by minimum Sharpe
    positive = {k: max(0.0, v) for k, v in sharpes.items() if v >= min_sharpe}

    if not positive:
        # All strategies fail: park in CDI if available, else equal-weight
        if cdi_col in cols:
            w = pd.Series(0.0, index=cols)
            w[cdi_col] = 1.0
            return w
        return equal_weights(len(cols), cols)

    total_sharpe = sum(positive.values())
    if total_sharpe == 0:
        # All qualifying strategies have Sharpe == 0 (tie), use equal weight among them
        n_pos = len(positive)
        w = pd.Series(0.0, index=cols)
        for k in positive:
            w[k] = 1.0 / n_pos
        return w

    w = pd.Series(0.0, index=cols)
    for k, v in positive.items():
        w[k] = v / total_sharpe

    return w


# ── Regime-conditional allocation presets ────────────────────────────────────

# Equity budget per macro regime (is_easing x is_calm)
REGIME_EQUITY_BUDGET = {
    (True, True):   0.80,  # Easing + Calm: aggressive
    (True, False):  0.50,  # Easing + Stressed: moderate
    (False, True):  0.30,  # Tightening + Calm: defensive
    (False, False): 0.00,  # Tightening + Stressed: full risk-off
}

# Equity strategies (non-CDI) to consider for each regime
_EQUITY_STRATS = [
    "CDI+MA200", "Res.MultiFactor", "RegimeSwitching",
    "COPOM Easing", "MultiFactor", "SmallcapMom", "LowVol", "MomSharpe",
]


def regime_conditional_weights(
    is_easing: bool,
    is_calm: bool,
    strategy_names: list,
    cdi_col: str = "CDI",
    equity_strats: list = None,
) -> pd.Series:
    """
    Pre-defined weight mixes based on macro regime (COPOM x IBOV volatility).

    Within the equity budget, allocate equal weight to all equity strategies
    present in strategy_names.

    Parameters
    ----------
    is_easing : bool
        True if COPOM is currently easing (CDI 3m trend).
    is_calm : bool
        True if IBOV volatility is in the lower 70th percentile.
    strategy_names : list
        All available strategy columns (including CDI).
    cdi_col : str
        Column name for the CDI (risk-free) asset.
    equity_strats : list, optional
        List of strategy names considered "equity" for regime budgeting.
        Defaults to the module-level _EQUITY_STRATS list.

    Returns
    -------
    pd.Series of weights summing to 1.0.
    """
    if equity_strats is None:
        equity_strats = _EQUITY_STRATS

    equity_budget = REGIME_EQUITY_BUDGET.get((is_easing, is_calm), 0.0)
    cdi_budget = 1.0 - equity_budget

    # Select which equity strategies are available
    available_equity = [s for s in strategy_names if s in equity_strats]

    w = pd.Series(0.0, index=strategy_names)

    if cdi_col in strategy_names:
        w[cdi_col] = cdi_budget

    if available_equity and equity_budget > 0:
        eq_per_strat = equity_budget / len(available_equity)
        for s in available_equity:
            w[s] = eq_per_strat
    else:
        # No equity strategies or zero equity budget; all goes to CDI
        if cdi_col in strategy_names:
            w[cdi_col] = 1.0

    # Normalize just in case
    total = w.sum()
    if total > 0:
        w /= total

    return w


# ─────────────────────────────────────────────────────────────────────────────
#  Portfolio Backtest Helpers (shared across portfolio_*.py scripts)
# ─────────────────────────────────────────────────────────────────────────────

def compute_portfolio_returns(
    returns_df: "pd.DataFrame",
    strategy_cols: list,
    weight_fn,
    lookback: int = 36,
) -> "tuple[pd.Series, pd.DataFrame]":
    """
    Compute time-series of portfolio returns using a rolling weight function.

    Parameters
    ----------
    returns_df : DataFrame with strategy return columns
    strategy_cols : list of strategy names to include
    weight_fn : callable(window_df) -> pd.Series of weights
    lookback : months of history before weights are computed (equal-weight before)

    Returns
    -------
    (port_returns_series, weights_history_df)
    """
    sub_df = returns_df[strategy_cols].copy()
    port_rets = []
    dates = []
    weights_history = []

    for i in range(len(sub_df)):
        date = sub_df.index[i]
        if i < lookback:
            w = equal_weights(len(strategy_cols), sub_df.columns)
        else:
            window = sub_df.iloc[:i]
            w = weight_fn(window)

        row = sub_df.iloc[i].fillna(0.0)
        port_ret = (w * row).sum()
        port_rets.append(port_ret)
        dates.append(date)
        weights_history.append(w)

    port_series = pd.Series(port_rets, index=dates)
    weights_df = pd.DataFrame(weights_history, index=dates)
    return port_series, weights_df


def compute_regime_portfolio(
    returns_df: "pd.DataFrame",
    strategy_cols: list,
    regime_signals: dict,
    mode: str = "rolling_sharpe",
    lookback_sharpe: int = 12,
) -> "tuple[pd.Series, pd.DataFrame]":
    """
    Compute portfolio returns using regime-based or dynamic Sharpe allocation.

    Parameters
    ----------
    returns_df : DataFrame with strategy + benchmark columns
    strategy_cols : list of equity strategy names
    regime_signals : dict with 'is_easing' and 'ibov_calm' Series
    mode : one of 'rolling_sharpe', 'regime_only', 'combined'
    lookback_sharpe : trailing months for Sharpe computation

    Returns
    -------
    (port_returns_series, weights_history_df)
    """
    is_easing = regime_signals["is_easing"]
    ibov_calm = regime_signals["ibov_calm"]

    sub_df = returns_df[strategy_cols].copy()
    if "CDI" in returns_df.columns and "CDI" not in sub_df.columns:
        sub_df["CDI"] = returns_df["CDI"]

    all_cols = list(sub_df.columns)
    equity_cols = [c for c in strategy_cols if c != "CDI"]

    port_rets = []
    dates = []
    weights_history = []

    for i in range(len(sub_df)):
        date = sub_df.index[i]

        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        calm = bool(ibov_calm.reindex(sub_df.index).iloc[i]) if i < len(ibov_calm) else True

        if mode == "rolling_sharpe":
            if i < lookback_sharpe:
                w = equal_weights(len(all_cols), all_cols)
            else:
                window = sub_df.iloc[:i]
                w = rolling_sharpe_weights(window, lookback_sharpe, min_sharpe=0.0, cdi_col="CDI")
                if "CDI" not in w.index:
                    w = pd.concat([w, pd.Series(0.0, index=["CDI"])])
                w = w.reindex(all_cols, fill_value=0.0)
                s = w.sum()
                if s > 0:
                    w /= s

        elif mode == "regime_only":
            w = regime_conditional_weights(
                is_easing=easing, is_calm=calm,
                strategy_names=all_cols, cdi_col="CDI", equity_strats=equity_cols,
            )

        elif mode == "combined":
            equity_budget = REGIME_EQUITY_BUDGET.get((easing, calm), 0.0)
            cdi_budget = 1.0 - equity_budget
            if i < lookback_sharpe or equity_budget == 0:
                eq_w = equal_weights(len(equity_cols), equity_cols)
            else:
                eq_window = sub_df[equity_cols].iloc[:i]
                eq_w = rolling_sharpe_weights(eq_window, lookback_sharpe, min_sharpe=0.0)

            w = pd.Series(0.0, index=all_cols)
            if "CDI" in all_cols:
                w["CDI"] = cdi_budget
            for col in equity_cols:
                if col in eq_w.index:
                    w[col] = eq_w[col] * equity_budget
            s = w.sum()
            if s > 0:
                w /= s

        else:
            w = equal_weights(len(all_cols), all_cols)

        row = sub_df.iloc[i].fillna(0.0)
        port_ret = (w * row).sum()
        port_rets.append(port_ret)
        dates.append(date)
        weights_history.append(w)

    port_series = pd.Series(port_rets, index=dates)
    weights_df = pd.DataFrame(weights_history, index=dates)
    return port_series, weights_df

"""
Fundamental signals library for feature discovery.
Implements 9 fundamental signal categories with YoY delta variants.
Counterpart to base_signals.py for CVM-sourced fundamentals data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_SHARES_SCALE = 1_000.0


# ---------------------------------------------------------------------------
# Computation functions
# ---------------------------------------------------------------------------


def compute_fund_pe_ratio(
    f_shares: pd.DataFrame,
    f_net_income: pd.DataFrame,
    adj_close: pd.DataFrame,
) -> pd.DataFrame:
    """
    P/E ratio.
    market_cap = adj_close * (f_shares / _SHARES_SCALE)
    P/E = market_cap / (f_net_income * _SHARES_SCALE)
    NaN where net_income <= 0.
    """
    market_cap = adj_close * (f_shares / _SHARES_SCALE)
    ni_brl = f_net_income * _SHARES_SCALE
    # Zero/negative net income → NaN
    ni_brl = ni_brl.where(ni_brl > 0)
    return market_cap / ni_brl


def compute_fund_earnings_yield(
    f_shares: pd.DataFrame,
    f_net_income: pd.DataFrame,
    adj_close: pd.DataFrame,
) -> pd.DataFrame:
    """
    Earnings yield = 1 / P/E.
    More statistically well-behaved than P/E directly.
    NaN where market_cap <= 0.
    """
    market_cap = adj_close * (f_shares / _SHARES_SCALE)
    ni_brl = f_net_income * _SHARES_SCALE
    ni_brl = ni_brl.where(ni_brl > 0)
    pe = market_cap / ni_brl
    mc_safe = market_cap.where(market_cap > 0)
    return ni_brl / mc_safe


def compute_fund_pb_ratio(
    f_shares: pd.DataFrame,
    f_equity: pd.DataFrame,
    adj_close: pd.DataFrame,
) -> pd.DataFrame:
    """
    P/B ratio.
    market_cap = adj_close * (f_shares / _SHARES_SCALE)
    P/B = market_cap / (f_equity * _SHARES_SCALE)
    NaN where equity <= 0.
    """
    market_cap = adj_close * (f_shares / _SHARES_SCALE)
    equity_brl = f_equity * _SHARES_SCALE
    equity_brl = equity_brl.where(equity_brl > 0)
    return market_cap / equity_brl


def compute_fund_ev_ebitda(
    f_shares: pd.DataFrame,
    f_ebitda: pd.DataFrame,
    f_net_debt: pd.DataFrame,
    adj_close: pd.DataFrame,
) -> pd.DataFrame:
    """
    EV/EBITDA.
    EV = market_cap + f_net_debt * _SHARES_SCALE
    EV/EBITDA = EV / (f_ebitda * _SHARES_SCALE)
    NaN where ebitda <= 0.
    """
    market_cap = adj_close * (f_shares / _SHARES_SCALE)
    net_debt_brl = f_net_debt * _SHARES_SCALE
    ebitda_brl = f_ebitda * _SHARES_SCALE
    ev = market_cap + net_debt_brl
    ebitda_brl = ebitda_brl.where(ebitda_brl > 0)
    return ev / ebitda_brl


def compute_fund_roe(
    f_net_income: pd.DataFrame,
    f_equity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return on equity = net_income / equity.
    Scale cancels (both in thousands).
    NaN where equity == 0.
    """
    equity_safe = f_equity.where(f_equity != 0)
    return f_net_income / equity_safe


def compute_fund_roa(
    f_net_income: pd.DataFrame,
    f_total_assets: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return on assets = net_income / total_assets.
    Scale cancels.
    NaN where total_assets == 0.
    """
    assets_safe = f_total_assets.where(f_total_assets != 0)
    return f_net_income / assets_safe


def compute_fund_net_margin(
    f_net_income: pd.DataFrame,
    f_revenue: pd.DataFrame,
) -> pd.DataFrame:
    """
    Net profit margin = net_income / revenue.
    Scale cancels.
    NaN where revenue <= 0.
    """
    rev_safe = f_revenue.where(f_revenue > 0)
    return f_net_income / rev_safe


def compute_fund_revenue_growth_yoy(
    f_revenue: pd.DataFrame,
) -> pd.DataFrame:
    """
    Revenue YoY growth rate.
    = (revenue - revenue.shift(252)) / revenue.shift(252).abs()
    NaN where prior-year revenue is zero or absent.
    """
    prior = f_revenue.shift(252)
    prior_abs = prior.abs()
    prior_abs = prior_abs.where(prior_abs > 0)
    return (f_revenue - prior) / prior_abs


def compute_fund_debt_to_equity(
    f_net_debt: pd.DataFrame,
    f_equity: pd.DataFrame,
) -> pd.DataFrame:
    """
    Net debt / equity.
    NaN where equity == 0.
    Can be negative (net cash position).
    """
    equity_safe = f_equity.where(f_equity != 0)
    return f_net_debt / equity_safe


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_fundamental_base_signals(data: dict):
    """
    Generate fundamental base signals.

    Yields (feature_id, category, params, wide_df) for each signal.
    Yields nothing if fundamentals data is absent or all-empty.

    Each signal is generated in two variants:
      1. Current value (e.g. feature_id "Fund_PE_ratio")
      2. YoY delta: signal - signal.shift(252) (e.g. "Fund_PE_ratio_yoy_delta")

    Total: 9 base + 9 YoY delta = 18 signals.
    """
    f_net_income = data.get("f_net_income")
    if f_net_income is None or (isinstance(f_net_income, pd.DataFrame) and f_net_income.empty):
        return

    adj_close = data.get("adj_close", pd.DataFrame())
    f_shares = data.get("f_shares", pd.DataFrame())
    f_equity = data.get("f_equity", pd.DataFrame())
    f_total_assets = data.get("f_total_assets", pd.DataFrame())
    f_ebitda = data.get("f_ebitda", pd.DataFrame())
    f_net_debt = data.get("f_net_debt", pd.DataFrame())
    f_revenue = data.get("f_revenue", pd.DataFrame())

    signals = [
        (
            "Fund_PE_ratio",
            "valuation",
            lambda: compute_fund_pe_ratio(f_shares, f_net_income, adj_close),
        ),
        (
            "Fund_Earnings_yield",
            "valuation",
            lambda: compute_fund_earnings_yield(f_shares, f_net_income, adj_close),
        ),
        (
            "Fund_PB_ratio",
            "valuation",
            lambda: compute_fund_pb_ratio(f_shares, f_equity, adj_close),
        ),
        (
            "Fund_EV_EBITDA",
            "valuation",
            lambda: compute_fund_ev_ebitda(f_shares, f_ebitda, f_net_debt, adj_close),
        ),
        (
            "Fund_ROE",
            "quality",
            lambda: compute_fund_roe(f_net_income, f_equity),
        ),
        (
            "Fund_ROA",
            "quality",
            lambda: compute_fund_roa(f_net_income, f_total_assets),
        ),
        (
            "Fund_Net_margin",
            "quality",
            lambda: compute_fund_net_margin(f_net_income, f_revenue),
        ),
        (
            "Fund_Revenue_growth_yoy",
            "growth",
            lambda: compute_fund_revenue_growth_yoy(f_revenue),
        ),
        (
            "Fund_Debt_to_equity",
            "leverage",
            lambda: compute_fund_debt_to_equity(f_net_debt, f_equity),
        ),
    ]

    for feature_id, category, fn in signals:
        # Base signal
        try:
            base_df = fn()
            yield (feature_id, category, {}, base_df)
        except Exception as e:
            print(f"  WARNING: Failed to compute {feature_id}: {e}")
            continue

        # YoY delta variant
        try:
            if feature_id == "Fund_Revenue_growth_yoy":
                delta_id = "Fund_Revenue_growth_yoy_delta2y"
            else:
                delta_id = f"{feature_id}_yoy_delta"
            delta_df = base_df - base_df.shift(252)
            yield (delta_id, category, {}, delta_df)
        except Exception as e:
            print(f"  WARNING: Failed to compute {delta_id}: {e}")

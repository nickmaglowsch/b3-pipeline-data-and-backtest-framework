"""Ibovespa Smart Low Volatility B3 — replica of the B3 index methodology.

Per B3's "Metodologia do Índice Bovespa Smart Low Volatility B3":
  * Universe: liquid B3 stocks (proxy for Ibovespa constituents via ADTV).
  * Volatility: annualized EWMA (span=252) of daily returns.
  * Inclusion: bottom 33% by smoothed annualized vol.
  * Weighting: inverse of the vol, capped at 10% per name, renormalized.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


def ewma_annualized_vol(daily_close: pd.DataFrame, n: int = 252) -> pd.DataFrame:
    """Annualized EWMA volatility of daily returns (B3 methodology, Anexo 1)."""
    r = daily_close.pct_change()
    ewma_var = (r * r).ewm(span=n, min_periods=n).mean()
    return np.sqrt(ewma_var * 252)


def inverse_vol_capped(vol: pd.Series, cap: float = 0.10) -> pd.Series:
    """Inverse-vol weights, each capped at `cap`, renormalized to sum to 1.

    Iteratively clips names at the cap and redistributes the excess across the
    uncapped names (standard cap-and-redistribute).
    """
    w = (1.0 / vol)
    w = w / w.sum()
    for _ in range(len(w)):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if not under.any() or w[under].sum() == 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


class IbovLowVolStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "IbovLowVolB3"

    @property
    def description(self) -> str:
        return (
            "Ibovespa Smart Low Volatility B3 replica. Selects the bottom 33% of "
            "liquid B3 stocks by annualized EWMA(252) volatility, weighted by the "
            "inverse of that volatility, capped at 10% per name."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "ewma_n", "EWMA Span (days)", "int", 252,
                description="EWMA span for daily-return volatility",
                min_value=20, max_value=504, step=1,
            ),
            ParameterSpec(
                "bottom_pct", "Bottom Vol Fraction", "float", 0.33,
                description="Fraction of eligible universe kept (lowest vol)",
                min_value=0.05, max_value=1.0, step=0.01,
            ),
            ParameterSpec(
                "weight_cap", "Max Weight per Name", "float", 0.10,
                min_value=0.01, max_value=1.0, step=0.01,
            ),
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                min_value=0.0, max_value=10.0, step=0.5,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]                    # period returns (index = rebal dates)
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        daily_close = shared_data["split_adj_close"]  # daily adjusted close

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        n = int(params.get("ewma_n", 252))
        bottom_pct = params.get("bottom_pct", 0.33)
        cap = params.get("weight_cap", 0.10)

        # Annualized EWMA vol on daily data, sampled onto the rebalance calendar.
        vol_daily = ewma_annualized_vol(daily_close, n)
        vol = vol_daily.reindex(vol_daily.index.union(ret.index)).ffill().reindex(ret.index)
        vol = vol.reindex(columns=ret.columns)
        vol[shared_data["has_glitch"] == 1] = np.nan

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        start = 2
        for i in range(start, len(ret)):
            vol_row = vol.iloc[i - 1]
            mask = (adtv.iloc[i - 1] >= min_adtv) & (raw_close.iloc[i - 1] >= min_price)
            valid = vol_row[mask].dropna()
            valid = valid[valid > 0]
            if len(valid) < 5:
                continue
            k = max(5, int(len(valid) * bottom_pct))
            sel = valid.nsmallest(k)
            w = inverse_vol_capped(sel, cap)
            for t, wt in w.items():
                tw.iloc[i, tw.columns.get_loc(t)] = wt

        return ret, tw

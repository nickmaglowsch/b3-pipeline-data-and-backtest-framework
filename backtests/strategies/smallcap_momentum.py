"""Strategy 6: Smallcap Momentum (below-median ADTV stocks, 6-month momentum)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_REBALANCE_FREQ,
)


class SmallcapMomentumStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "SmallcapMom"

    @property
    def description(self) -> str:
        return (
            "Smallcap Momentum. Selects below-median-ADTV stocks (small caps) "
            "ranked by 6-month momentum, top 20%. Uses low ADTV threshold (default R$100K). "
            "WARNING: historically experienced bankruptcy losses (2009)."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "min_adtv", "Min ADTV (BRL)", "float", 100_000.0,
                description="Minimum ADTV for smallcap universe (intentionally low)",
                min_value=0.0, step=10_000.0,
            ),
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                min_value=0.0, max_value=10.0, step=0.5,
            ),
            ParameterSpec(
                "lookback", "Momentum Lookback (periods)", "int", 6,
                description="Number of periods for momentum computation",
                min_value=1, max_value=24, step=1,
            ),
            ParameterSpec(
                "top_pct", "Top Percentile", "float", 0.20,
                description="Fraction of eligible smallcap universe to select",
                min_value=0.01, max_value=0.50, step=0.01,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        log_ret = shared_data["log_ret"]
        has_glitch = shared_data["has_glitch"]

        min_adtv = params.get("min_adtv", 100_000)
        min_price = params.get("min_price", 1.0)
        lookback = params.get("lookback", 6)
        top_pct = params.get("top_pct", 0.20)

        mom = log_ret.shift(1).rolling(lookback).sum()
        mom[has_glitch == 1] = np.nan

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        r = ret.copy()

        start = lookback + 2
        for i in range(start, len(ret)):
            sig_row = mom.iloc[i - 1]
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]

            # Liquidity filter first
            liquid_mask = (adtv_r >= min_adtv) & (raw_r >= min_price)
            liquid_universe = sig_row[liquid_mask].dropna()

            if len(liquid_universe) < 5:
                continue

            # Small-cap filter: below median ADTV within the liquid universe
            liquid_adtv = adtv_r[liquid_universe.index]
            med = liquid_adtv.median()
            smallcap_mask = liquid_adtv <= med
            valid = liquid_universe[smallcap_mask]
            if len(valid) < 5:
                continue
            n = max(1, int(len(valid) * top_pct))
            sel = valid.nlargest(n).index.tolist()
            w = 1.0 / len(sel)
            for t in sel:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w

        return r, tw

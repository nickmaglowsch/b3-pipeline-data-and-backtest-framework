"""Strategy 1: CDI + MA200 Trend Filter."""
from __future__ import annotations

import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class CdiMa200Strategy(StrategyBase):
    @property
    def name(self) -> str:
        return "CDI+MA200"

    @property
    def description(self) -> str:
        return (
            "CDI + MA200 Trend Filter. "
            "During COPOM easing, hold equal-weight stocks above their 200-day MA. "
            "During tightening, hold 100% CDI."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                description="Minimum stock price to be eligible",
                min_value=0.0, max_value=10.0, step=0.5,
            ),
            ParameterSpec(
                "lookback_easing", "Easing Lookback (periods)", "int", 3,
                description="CDI shift periods to compare for easing detection (shift(1) vs shift(N))",
                min_value=2, max_value=12, step=1,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        has_glitch = shared_data["has_glitch"]
        is_easing = shared_data["is_easing"]
        above_ma200 = shared_data["above_ma200"]
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        tw["CDI_ASSET"] = 0.0
        r = ret.copy()
        r["CDI_ASSET"] = cdi_monthly
        r["IBOV"] = ibov_ret

        for i in range(13, len(ret)):
            val = is_easing.iloc[i] if i < len(is_easing) else False
            easing = bool(val) if pd.notna(val) else False
            if not easing:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue
            above = above_ma200.iloc[i - 1]
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
            mask = above.fillna(False) & (adtv_r >= min_adtv) & (raw_r >= min_price)
            if len(gl) > 0:
                mask = mask & (gl != 1)
            tickers = mask[mask].index.tolist()
            # Exclude synthetic columns
            tickers = [t for t in tickers if t in ret.columns]
            if not tickers:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            else:
                w = 1.0 / len(tickers)
                for t in tickers:
                    tw.iloc[i, tw.columns.get_loc(t)] = w

        return r, tw

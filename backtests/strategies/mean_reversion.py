"""Strategy 9: Mean Reversion (buy biggest recent losers)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class MeanReversionStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "MeanReversion"

    @property
    def description(self) -> str:
        return (
            "Short-Term Mean Reversion. Buys the biggest recent losers (1-month negative return). "
            "Top decile of the liquid universe ranked by most negative 1-period return."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "lookback", "Lookback Periods", "int", 1,
                description="Periods for mean-reversion signal (1 = last month)",
                min_value=1, max_value=6, step=1,
            ),
            ParameterSpec(
                "top_pct", "Top Percentile", "float", 0.10,
                description="Fraction of eligible universe to select (biggest losers)",
                min_value=0.01, max_value=0.50, step=0.01,
            ),
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                min_value=0.0, max_value=10.0, step=0.5,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        top_pct = params.get("top_pct", 0.10)
        lookback = params.get("lookback", 1)

        # Signal = negative return (bigger loser = higher signal)
        signal = -ret
        has_glitch = (ret > 1.0) | (ret < -0.90)
        signal[has_glitch] = np.nan

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        r = ret.copy()

        prev_sel: set = set()
        start_idx = lookback + 1

        for i in range(start_idx, len(ret)):
            sig_row = signal.iloc[i - 1]
            adtv_row = adtv.iloc[i - 1]
            raw_close_row = raw_close.iloc[i - 1]

            valid_mask = (adtv_row >= min_adtv) & (raw_close_row >= min_price)
            valid = sig_row[valid_mask].dropna()

            if len(valid) < 5:
                sel = prev_sel
            else:
                n_sel = max(1, int(len(valid) * top_pct))
                sel = set(valid.nlargest(n_sel).index)

            if not sel:
                continue

            w = 1.0 / len(sel)
            for t in sel:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w
            prev_sel = sel

        return r, tw

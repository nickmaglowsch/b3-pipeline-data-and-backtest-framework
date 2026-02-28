"""Strategy 10: Frog-in-the-Pan (Continuous Information momentum)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class FrogInPanStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "FrogInPan"

    @property
    def description(self) -> str:
        return (
            "Frog-in-the-Pan (Da, Gurun & Warachka 2014). "
            "Ranks stocks by a composite of 12-month momentum + win-rate (number of positive months). "
            "Investors under-react to small, continuous positive news."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "lookback", "Lookback Periods", "int", 12,
                description="Periods for momentum and win-rate signals",
                min_value=3, max_value=24, step=1,
            ),
            ParameterSpec(
                "top_pct", "Top Percentile", "float", 0.10,
                description="Fraction of eligible universe to select",
                min_value=0.01, max_value=0.50, step=0.01,
            ),
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                min_value=0.0, max_value=10.0, step=0.5,
            ),
            ParameterSpec(
                "mom_weight", "Momentum Weight", "float", 0.5,
                description="Weight for momentum rank (remainder goes to win-rate rank)",
                min_value=0.0, max_value=1.0, step=0.1,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        log_ret = shared_data["log_ret"]

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        lookback = params.get("lookback", 12)
        top_pct = params.get("top_pct", 0.10)
        mom_w = params.get("mom_weight", 0.5)

        skip = 1  # skip most-recent month to avoid microstructure

        mom = log_ret.shift(skip).rolling(lookback).sum()
        mom_rank = mom.rank(axis=1, pct=True)

        is_positive = (ret > 0).astype(float)
        is_positive[ret.isna()] = np.nan
        win_rate = is_positive.shift(skip).rolling(lookback).sum()
        win_rank = win_rate.rank(axis=1, pct=True)

        signal = mom_rank * mom_w + win_rank * (1 - mom_w)

        has_glitch = (
            ((ret > 1.0) | (ret < -0.90))
            .shift(skip)
            .rolling(lookback)
            .max()
        )
        signal[has_glitch == 1] = np.nan

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        r = ret.copy()

        start_idx = lookback + skip + 1
        for i in range(start_idx, len(ret)):
            sig_row = signal.iloc[i - 1]
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            mask = (adtv_r >= min_adtv) & (raw_r >= min_price)
            valid = sig_row[mask].dropna()
            if len(valid) < 5:
                continue
            n = max(1, int(len(valid) * top_pct))
            sel = valid.nlargest(n).index.tolist()
            w = 1.0 / len(sel)
            for t in sel:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w

        return r, tw

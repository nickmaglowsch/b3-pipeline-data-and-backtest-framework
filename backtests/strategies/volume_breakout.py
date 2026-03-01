"""Strategy 11: Volume Breakout (volume acceleration + positive price trend)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class VolumeBreakoutStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "VolumeBreakout"

    @property
    def description(self) -> str:
        return (
            "Volume Breakout. Detects institutional accumulation via volume acceleration "
            "(current volume / N-period avg) combined with positive price momentum. "
            "Inspired by emerging-market information asymmetry dynamics."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "vol_lookback", "Volume Lookback (periods)", "int", 6,
                description="Periods for volume moving average comparison",
                min_value=2, max_value=24, step=1,
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
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        fin_vol = shared_data["fin_vol"]

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        vol_lookback = params.get("vol_lookback", 6)
        top_pct = params.get("top_pct", 0.10)

        # Resample to same freq as ret
        # fin_vol is already daily -- use adtv (monthly mean) as volume proxy
        # For the "period volume" concept, sum works better but adtv is available
        skip = 1

        is_positive = ret.shift(skip) > 0
        vol_accel = adtv.shift(skip) / adtv.shift(skip).rolling(vol_lookback).mean()
        signal = vol_accel.copy()
        signal[~is_positive] = np.nan

        has_glitch = ((ret > 1.0) | (ret < -0.90)).shift(skip).rolling(vol_lookback).max()
        signal[has_glitch == 1] = np.nan

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        r = ret.copy()

        prev_sel: set = set()
        start_idx = vol_lookback + skip + 1
        for i in range(start_idx, len(ret)):
            sig_row = signal.iloc[i - 1]
            adtv_row = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            mask = (adtv_row >= min_adtv) & (raw_r >= min_price)
            valid = sig_row[mask].dropna()

            if len(valid) < 5:
                sel = prev_sel
            else:
                n = max(5, int(len(valid) * top_pct))
                sel = set(valid.nlargest(n).index)

            if not sel:
                continue
            w = 1.0 / len(sel)
            for t in sel:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w
            prev_sel = sel

        return r, tw

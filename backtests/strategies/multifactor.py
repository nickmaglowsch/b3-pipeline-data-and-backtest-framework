"""Strategy 5: Multifactor (50% Momentum + 50% Low Vol, always invested)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class MultifactorStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "MultiFactor"

    @property
    def description(self) -> str:
        return (
            "Multifactor (always invested). "
            "Composite = mom_weight * momentum_rank + vol_weight * low_vol_rank. "
            "Top decile of eligible liquid universe."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                min_value=0.0, max_value=10.0, step=0.5,
            ),
            ParameterSpec(
                "lookback", "Lookback Periods", "int", 12,
                description="Periods for momentum and volatility estimation",
                min_value=3, max_value=36, step=1,
            ),
            ParameterSpec(
                "top_pct", "Top Percentile", "float", 0.10,
                description="Fraction of eligible universe to select",
                min_value=0.01, max_value=0.50, step=0.01,
            ),
            ParameterSpec(
                "mom_weight", "Momentum Weight", "float", 0.5,
                description="Weight given to momentum score (1 - mom_weight goes to low-vol)",
                min_value=0.0, max_value=1.0, step=0.1,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        log_ret = shared_data["log_ret"]
        has_glitch = shared_data["has_glitch"]

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        lookback = params.get("lookback", 12)
        top_pct = params.get("top_pct", 0.10)
        mom_w = params.get("mom_weight", 0.5)
        vol_w = 1.0 - mom_w

        mom_sig = log_ret.shift(1).rolling(lookback).sum()
        mom_sig[has_glitch == 1] = np.nan
        vol_sig = -ret.shift(1).rolling(lookback).std()
        vol_sig[has_glitch == 1] = np.nan
        composite = mom_sig.rank(axis=1, pct=True) * mom_w + vol_sig.rank(axis=1, pct=True) * vol_w

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        r = ret.copy()

        for i in range(lookback + 2, len(ret)):
            sig_row = composite.iloc[i - 1]
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            mask = (adtv_r >= min_adtv) & (raw_r >= min_price)
            valid = sig_row[mask].dropna()
            if len(valid) < 5:
                continue
            n = max(5, int(len(valid) * top_pct))
            sel = valid.nlargest(n).index.tolist()
            w = 1.0 / len(sel)
            for t in sel:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w

        return r, tw

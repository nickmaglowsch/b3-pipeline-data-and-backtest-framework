"""Strategy 13: Volatility-Regime Adaptive Low Volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class AdaptiveLowVolStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "AdaptiveLowVol"

    @property
    def description(self) -> str:
        return (
            "Volatility-Regime Adaptive Low Volatility (based on ML feature importance research). "
            "Regime: CDI easing + IBOV vol. "
            "CALM+EASING -> top 15% | CALM+TIGHT -> top 10% | STRESS+EASING -> top 5% | "
            "STRESS+TIGHT -> 100% CDI."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "lookback", "Vol Lookback Periods", "int", 12,
                description="Periods for stock-level volatility signal",
                min_value=3, max_value=36, step=1,
            ),
            ParameterSpec(
                "ibov_vol_percentile", "IBOV Vol Stress Percentile", "float", 0.70,
                description="IBOV vol percentile above which market is considered 'stressed'",
                min_value=0.5, max_value=0.95, step=0.05,
            ),
            ParameterSpec(
                "pct_aggressive", "Percentile (Calm+Easing)", "float", 0.15,
                description="Top percentile when regime is calm+easing",
                min_value=0.01, max_value=0.50, step=0.01,
            ),
            ParameterSpec(
                "pct_moderate", "Percentile (Calm+Tight)", "float", 0.10,
                description="Top percentile when regime is calm+tightening",
                min_value=0.01, max_value=0.50, step=0.01,
            ),
            ParameterSpec(
                "pct_defensive", "Percentile (Stress+Easing)", "float", 0.05,
                description="Top percentile when regime is stressed+easing",
                min_value=0.01, max_value=0.25, step=0.01,
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
        has_glitch = shared_data["has_glitch"]
        is_easing = shared_data["is_easing"]
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]

        # IBOV vol regime
        ibov_vol_pctrank = shared_data.get("ibov_vol_pctrank")

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        lookback = params.get("lookback", 12)
        ibov_vol_pct_threshold = params.get("ibov_vol_percentile", 0.70)
        pct_aggressive = params.get("pct_aggressive", 0.15)
        pct_moderate = params.get("pct_moderate", 0.10)
        pct_defensive = params.get("pct_defensive", 0.05)

        # Blended vol signal: 50% 12m rolling std + 50% 2m rolling std (proxy for 20d)
        vol_60d = shared_data.get("vol_60d", ret.rolling(5).std())
        vol_20d = shared_data.get("vol_20d", ret.rolling(2).std())
        vol_signal = -(0.5 * vol_60d + 0.5 * vol_20d)
        vol_signal[has_glitch == 1] = np.nan

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        tw["CDI_ASSET"] = 0.0
        r = ret.copy()
        r["CDI_ASSET"] = cdi_monthly
        r["IBOV"] = ibov_ret

        for i in range(lookback + 1, len(ret)):
            _v = is_easing.iloc[i] if i < len(is_easing) else False
            easing = bool(_v) if pd.notna(_v) else False

            # IBOV stress regime
            if ibov_vol_pctrank is not None and i - 1 < len(ibov_vol_pctrank):
                ibov_vol_rank = float(ibov_vol_pctrank.iloc[i - 1]) if not np.isnan(ibov_vol_pctrank.iloc[i - 1]) else 0.0
            else:
                ibov_vol_rank = 0.0
            stressed = ibov_vol_rank > ibov_vol_pct_threshold

            # Determine selection percentile
            if not stressed and easing:
                top_pct = pct_aggressive
            elif not stressed and not easing:
                top_pct = pct_moderate
            elif stressed and easing:
                top_pct = pct_defensive
            else:
                # STRESS + TIGHT -> CDI
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue

            sig_row = vol_signal.iloc[i - 1]
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            mask = (adtv_r >= min_adtv) & (raw_r >= min_price)
            valid = sig_row[mask].dropna()
            if len(valid) < 5:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue
            n = max(5, int(len(valid) * top_pct))
            sel = valid.nlargest(n).index.tolist()
            w = 1.0 / len(sel)
            for t in sel:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w

        return r, tw

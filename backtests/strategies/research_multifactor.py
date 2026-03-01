"""Strategy 2: Research Multi-Factor (5 factors + regime filter)."""
from __future__ import annotations

import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class ResearchMultifactorStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "Res.MultiFactor"

    @property
    def description(self) -> str:
        return (
            "Research Multi-Factor. Requires 2-of-3 regime signals (easing, calm, uptrend). "
            "Composite score: dist_MA200 + low_vol_60d + low_ATR + low_vol_20d + high_ADTV (equal weights). "
            "Top 10% selected."
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
                "top_pct", "Top Percentile", "float", 0.10,
                description="Fraction of eligible universe to select",
                min_value=0.01, max_value=0.50, step=0.01,
            ),
            ParameterSpec(
                "regime_threshold", "Regime Threshold (signals)", "int", 2,
                description="Minimum number of regime signals (out of 3) required to be in equities",
                min_value=1, max_value=3, step=1,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        has_glitch = shared_data["has_glitch"]
        is_easing = shared_data["is_easing"]
        ibov_calm = shared_data["ibov_calm"]
        ibov_uptrend = shared_data["ibov_uptrend"]
        dist_ma200 = shared_data["dist_ma200"]
        vol_60d = shared_data["vol_60d"]
        atr_m = shared_data["atr_m"]
        vol_20d = shared_data["vol_20d"]
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        top_pct = params.get("top_pct", 0.10)
        regime_threshold = params.get("regime_threshold", 2)

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        tw["CDI_ASSET"] = 0.0
        r = ret.copy()
        r["CDI_ASSET"] = cdi_monthly
        r["IBOV"] = ibov_ret

        for i in range(14, len(ret)):
            _v = is_easing.iloc[i] if i < len(is_easing) else False
            sig_easing = bool(_v) if pd.notna(_v) else False
            _v = ibov_calm.iloc[i - 1] if i - 1 < len(ibov_calm) else False
            sig_calm = bool(_v) if pd.notna(_v) else False
            _v = ibov_uptrend.iloc[i - 1] if i - 1 < len(ibov_uptrend) else False
            sig_up = bool(_v) if pd.notna(_v) else False
            if (int(sig_easing) + int(sig_calm) + int(sig_up)) < regime_threshold:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue

            composite = (
                0.2 * dist_ma200.iloc[i - 1].rank(pct=True)
                + 0.2 * (-vol_60d.iloc[i - 1]).rank(pct=True)
                + 0.2 * (-atr_m.iloc[i - 1]).rank(pct=True)
                + 0.2 * (-vol_20d.iloc[i - 1]).rank(pct=True)
                + 0.2 * adtv.iloc[i - 1].rank(pct=True)
            )
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
            mask = (adtv_r >= min_adtv) & (raw_r >= min_price)
            if len(gl) > 0:
                mask = mask & (gl != 1)
            valid = composite[mask].dropna()
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

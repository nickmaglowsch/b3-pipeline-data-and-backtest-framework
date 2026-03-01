"""Strategy 3: Regime Switching (IBOV above N-month MA -> equities, else CDI)."""
from __future__ import annotations

import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class RegimeSwitchingStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "RegimeSwitching"

    @property
    def description(self) -> str:
        return (
            "Regime Switching. When IBOV is above its N-month moving average, "
            "invest in top-decile MultiFactor stocks. Otherwise hold 100% CDI."
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
                "lookback", "Signal Lookback (periods)", "int", 12,
                description="Lookback for momentum and vol signals",
                min_value=3, max_value=36, step=1,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        has_glitch = shared_data["has_glitch"]
        ibov_above = shared_data["ibov_above"]
        mf_composite = shared_data["mf_composite"]
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        top_pct = params.get("top_pct", 0.10)
        lookback = params.get("lookback", 12)

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        tw["CDI_ASSET"] = 0.0
        r = ret.copy()
        r["CDI_ASSET"] = cdi_monthly
        r["IBOV"] = ibov_ret

        for i in range(lookback + 2, len(ret)):
            above = bool(ibov_above.iloc[i - 1]) if i - 1 < len(ibov_above) else False
            if not above:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue
            sig_row = mf_composite.iloc[i - 1]
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

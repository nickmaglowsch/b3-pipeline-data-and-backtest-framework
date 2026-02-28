"""Strategy 4: COPOM Easing (IBOV vs CDI binary switch)."""
from __future__ import annotations

import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_REBALANCE_FREQ,
)


class CopomEasingStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "COPOM Easing"

    @property
    def description(self) -> str:
        return (
            "COPOM Easing: Simple macro regime switch. "
            "During COPOM easing (CDI declining), hold 100% IBOV. "
            "During tightening, hold 100% CDI."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "cdi_shift_short", "CDI Short Shift (periods)", "int", 1,
                description="Short lag for CDI comparison (e.g. 1 = 1 month ago)",
                min_value=1, max_value=6, step=1,
            ),
            ParameterSpec(
                "cdi_shift_long", "CDI Long Shift (periods)", "int", 4,
                description="Long lag for CDI comparison (e.g. 4 = 4 months ago)",
                min_value=2, max_value=12, step=1,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]

        shift_short = params.get("cdi_shift_short", 1)
        shift_long = params.get("cdi_shift_long", 4)

        # Rebuild easing signal with user-specified shifts
        is_easing = cdi_monthly.shift(shift_short) < cdi_monthly.shift(shift_long)

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        tw["CDI_ASSET"] = 0.0
        tw["IBOV"] = 0.0
        r = ret.copy()
        r["CDI_ASSET"] = cdi_monthly
        r["IBOV"] = ibov_ret

        start = max(shift_short, shift_long) + 1
        for i in range(start, len(ret)):
            val = is_easing.iloc[i] if i < len(is_easing) else False
            easing = bool(val) if pd.notna(val) else False
            if easing:
                tw.iloc[i, tw.columns.get_loc("IBOV")] = 1.0
            else:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0

        return r, tw

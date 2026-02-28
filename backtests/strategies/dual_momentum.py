"""Strategy 12: Dual Momentum (Brazil vs US Dollar macro allocation)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_REBALANCE_FREQ,
)
from backtests.core.data import download_benchmark


class DualMomentumStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "DualMomentum"

    @property
    def description(self) -> str:
        return (
            "Dual Momentum (Antonacci). Each month, compares N-month momentum of IBOV vs IVVB11 (S&P in BRL). "
            "Allocates 100% to the winner. If both are negative, moves to CDI (absolute momentum filter)."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "lookback_months", "Momentum Lookback (months)", "int", 3,
                description="Months of return history for the dual-momentum comparison",
                min_value=1, max_value=12, step=1,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        # DualMomentum doesn't use equity stock data; builds its own synthetic returns matrix.
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]

        lookback = params.get("lookback_months", 3)

        # Download IVVB11 (S&P 500 BRL-hedged ETF)
        start = str(cdi_monthly.index[0].date())
        end = str(cdi_monthly.index[-1].date())
        try:
            ivvb_px = download_benchmark("IVVB11.SA", start, end)
            freq = "ME"
            ivvb_ret = ivvb_px.resample(freq).last().pct_change().dropna()
        except Exception:
            # Fallback: if IVVB11 not available, use zeros
            ivvb_ret = pd.Series(0.0, index=ibov_ret.index, name="IVVB11")

        # Align
        common_idx = ibov_ret.index.intersection(cdi_monthly.index).intersection(ivvb_ret.index)
        df = pd.DataFrame({
            "IBOV_ASSET": ibov_ret.reindex(common_idx),
            "IVVB11_ASSET": ivvb_ret.reindex(common_idx),
            "CDI_ASSET": cdi_monthly.reindex(common_idx),
        }).dropna()

        mom_ibov = np.log1p(df["IBOV_ASSET"]).rolling(lookback).sum()
        mom_ivvb = np.log1p(df["IVVB11_ASSET"]).rolling(lookback).sum()

        tw = pd.DataFrame(0.0, index=df.index, columns=df.columns)

        for i in range(lookback, len(df)):
            i_mom = mom_ibov.iloc[i - 1]
            v_mom = mom_ivvb.iloc[i - 1]

            if i_mom <= 0 and v_mom <= 0:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            elif i_mom >= v_mom:
                tw.iloc[i, tw.columns.get_loc("IBOV_ASSET")] = 1.0
            else:
                tw.iloc[i, tw.columns.get_loc("IVVB11_ASSET")] = 1.0

        return df, tw

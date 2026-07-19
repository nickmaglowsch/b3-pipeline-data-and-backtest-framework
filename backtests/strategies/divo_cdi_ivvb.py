"""
Strategy: DIVO11 / CDI / IVVB11 — equal thirds, quarterly rebalance
===================================================================
The simplest three-sleeve portfolio: equal 1/3 in each of

  1/3  DIVO11   (B3 dividend-stock ETF, pulled from Yahoo)
  1/3  CDI      (Brazilian cash rate)
  1/3  IVVB11   (S&P 500 in BRL — the B3 ETF, pulled from Yahoo)

Weights are constant thirds; the simulator resets to them on each rebalance
grid row (quarter-end by default), so "QE balance" == quarterly rebalance back
to equal thirds. No screening, no fundamentals.
"""
from __future__ import annotations

import pandas as pd

from backtests.core.data import download_benchmark
from backtests.core.strategy_base import (
    StrategyBase,
    ParameterSpec,
    COMMON_START_DATE,
    COMMON_END_DATE,
)

DIVO11_TICKER = "DIVO11.SA"
IVVB11_TICKER = "IVVB11.SA"


class DivoCdiIvvbStrategy(StrategyBase):
    """Equal thirds: DIVO11 / CDI / IVVB11, rebalanced quarterly."""

    @property
    def name(self) -> str:
        return "DIVO11 / CDI / IVVB11 (thirds)"

    @property
    def description(self) -> str:
        return (
            "Quarterly-rebalanced equal thirds: 1/3 DIVO11 (B3 dividend ETF), "
            "1/3 CDI (cash), 1/3 IVVB11 (S&P 500 in BRL). Both ETFs from Yahoo."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            ParameterSpec(
                "rebalance_freq", "Rebalance Frequency", "choice", "QE",
                description="Rebalance back to equal thirds (quarterly by default)",
                choices=["ME", "QE", "W-FRI"],
            ),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        cdi = shared_data["cdi_monthly"]

        start, end = str(ret.index[0].date()), str(ret.index[-1].date())
        divo_px = download_benchmark(DIVO11_TICKER, start, end).reindex(
            ret.index, method="ffill"
        )
        ivvb_px = download_benchmark(IVVB11_TICKER, start, end).reindex(
            ret.index, method="ffill"
        )

        r = ret.copy()
        r["DIVO11"] = divo_px.pct_change().fillna(0.0)
        r["CDI_ASSET"] = cdi
        r["IVVB11"] = ivvb_px.pct_change().fillna(0.0)

        third = 1.0 / 3.0
        tw = pd.DataFrame(0.0, index=ret.index, columns=r.columns)
        tw["DIVO11"] = third
        tw["CDI_ASSET"] = third
        tw["IVVB11"] = third
        # ETFs list mid-history (DIVO11 ~2012, IVVB11 ~2014); before an ETF
        # prices, park its third in CDI so it isn't dead flat money.
        for col, px in (("DIVO11", divo_px), ("IVVB11", ivvb_px)):
            dead = px.isna()
            tw.loc[dead, col] = 0.0
            tw.loc[dead, "CDI_ASSET"] += third

        return r, tw

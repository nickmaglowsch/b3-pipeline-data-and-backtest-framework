"""
Strategy: DIVO11 / IVVB11 / CDI — 40/40/20, quarterly rebalance
================================================================
Three-sleeve portfolio with fixed weights:

  40%  DIVO11   (B3 dividend-stock ETF tracking IDIV, pulled from Yahoo)
  40%  IVVB11   (S&P 500 in BRL — the B3 ETF, pulled from Yahoo)
  20%  CDI      (Brazilian cash rate)

Weights are constant; the simulator resets to them on each rebalance grid row
(quarter-end by default). No screening, no fundamentals.
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

W_DIVO = 0.40
W_IVVB = 0.40
W_CDI = 0.20


class DivoIvvbCdi404020Strategy(StrategyBase):
    """40% DIVO11 / 40% IVVB11 / 20% CDI, rebalanced quarterly."""

    @property
    def name(self) -> str:
        return "DIVO11 / IVVB11 / CDI (40/40/20)"

    @property
    def description(self) -> str:
        return (
            "Quarterly-rebalanced fixed weights: 40% DIVO11 (B3 dividend ETF, "
            "IDIV), 40% IVVB11 (S&P 500 in BRL), 20% CDI (cash). ETFs from Yahoo."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            ParameterSpec(
                "rebalance_freq", "Rebalance Frequency", "choice", "QE",
                description="Rebalance back to fixed weights (quarterly by default)",
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

        tw = pd.DataFrame(0.0, index=ret.index, columns=r.columns)
        tw["DIVO11"] = W_DIVO
        tw["CDI_ASSET"] = W_CDI
        tw["IVVB11"] = W_IVVB
        # ETFs list mid-history (DIVO11 ~2012, IVVB11 ~2014); before an ETF
        # prices, park its weight in CDI so it isn't dead flat money.
        for col, w, px in (("DIVO11", W_DIVO, divo_px), ("IVVB11", W_IVVB, ivvb_px)):
            dead = px.isna()
            tw.loc[dead, col] = 0.0
            tw.loc[dead, "CDI_ASSET"] += w

        return r, tw

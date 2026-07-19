"""
Strategy: BOVA11 / IVVB11 / CDI — thirds (passive baseline)
==========================================================
A do-nothing baseline: equal thirds across the two big B3 ETFs and cash,
rebalanced quarterly.

  1/3  BOVA11  (Ibovespa ETF — Brazilian equity beta)
  1/3  IVVB11  (S&P 500 in BRL — US equity beta)
  1/3  CDI     (Brazilian cash rate)

No stock selection, no signals. This is the "could I have just bought the
index?" bar every active strategy should clear. ETF prices come from Yahoo
(BOVA11.SA / IVVB11.SA); CDI from shared data. Weights are the constant thirds
on every rebalance-grid row (quarterly grid -> quarterly rebalance).
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

BOVA11_TICKER = "BOVA11.SA"
IVVB11_TICKER = "IVVB11.SA"


class BaselineThirdsStrategy(StrategyBase):
    """Passive 1/3 BOVA11, 1/3 IVVB11, 1/3 CDI, rebalanced quarterly."""

    @property
    def name(self) -> str:
        return "BOVA11 / IVVB11 / CDI (thirds)"

    @property
    def description(self) -> str:
        return (
            "Passive baseline: equal thirds of BOVA11 (Ibovespa ETF), IVVB11 "
            "(S&P 500 in BRL) and CDI (cash), rebalanced quarterly. ETF prices "
            "from Yahoo. No stock selection — the index-beta bar for active "
            "strategies."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [COMMON_START_DATE, COMMON_END_DATE]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        cdi = shared_data["cdi_monthly"]
        start, end = str(ret.index[0].date()), str(ret.index[-1].date())

        def _rets(ticker: str) -> pd.Series:
            px = download_benchmark(ticker, start, end)
            return px.reindex(ret.index, method="ffill").pct_change().fillna(0.0)

        r = pd.DataFrame(index=ret.index)
        r["BOVA11"] = _rets(BOVA11_TICKER)
        r["IVVB11"] = _rets(IVVB11_TICKER)
        r["CDI_ASSET"] = cdi

        # only allocate once all three sleeves have live prices; park in CDI before
        # (AND: pre-2014 IVVB11 doesn't exist — a third parked in a dead sleeve
        # would earn 0%)
        equity_live = (r["BOVA11"] != 0) & (r["IVVB11"] != 0)
        live = equity_live.cummax()  # once true, stays true

        tw = pd.DataFrame(0.0, index=ret.index, columns=r.columns)
        third = 1.0 / 3.0
        tw.loc[live, ["BOVA11", "IVVB11", "CDI_ASSET"]] = third
        tw.loc[~live, "CDI_ASSET"] = 1.0

        return r, tw

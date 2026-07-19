"""
Strategy: Value / CDI / IVVB11 — thirds
=======================================
A three-sleeve portfolio rebalanced quarterly to equal thirds:

  1/3  Two-Leg Value  (the equity screen; see two_leg_value.py)
  1/3  CDI            (Brazilian cash rate)
  1/3  IVVB11         (S&P 500 in BRL — the B3 ETF, pulled from Yahoo)

The value sleeve is the Two-Leg Value basket, its equal-weighted names scaled
down to a 1/3 sleeve. CDI and IVVB11 each take a fixed 1/3. Weights are held
between quarterly rebalances (the value sub-strategy already rebalances mid
Apr/Jun/Sep/Dec); the simulator restates the thirds on each grid row.

Before the value basket first forms (early years with sparse fundamentals), the
whole portfolio parks in CDI.
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
from backtests.strategies.two_leg_value import TwoLegValueStrategy

IVVB11_TICKER = "IVVB11.SA"


class ValueCdiIvvbStrategy(StrategyBase):
    """Equal thirds: Two-Leg Value / CDI / IVVB11, rebalanced quarterly."""

    @property
    def name(self) -> str:
        return "Value / CDI / IVVB11 (thirds)"

    @property
    def description(self) -> str:
        return (
            "Quarterly-rebalanced equal thirds: 1/3 Two-Leg Value equity basket, "
            "1/3 CDI (cash), 1/3 IVVB11 (S&P 500 in BRL, from Yahoo). Parks in "
            "CDI until the value basket first forms."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        # Inherit the value sleeve's knobs; add only start/end at the top.
        base = TwoLegValueStrategy().get_parameter_specs()
        seen = {"start_date", "end_date"}
        return [COMMON_START_DATE, COMMON_END_DATE] + [
            p for p in base if p.name not in seen
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        cdi = shared_data["cdi_monthly"]

        # value sleeve — equal-weighted basket, rows sum to ~1.0 when invested
        _, tw_value = TwoLegValueStrategy().generate_signals(shared_data, params)
        invested = tw_value.sum(axis=1) > 0

        # IVVB11 monthly returns aligned to the rebalance grid (Yahoo)
        px = download_benchmark(
            IVVB11_TICKER, str(ret.index[0].date()), str(ret.index[-1].date())
        )
        ivvb_ret = px.reindex(ret.index, method="ffill").pct_change().fillna(0.0)

        r = ret.copy()
        r["CDI_ASSET"] = cdi
        r["IVVB11"] = ivvb_ret

        tw = pd.DataFrame(0.0, index=ret.index, columns=r.columns)
        third = 1.0 / 3.0
        # invested rows: value scaled to a third, CDI + IVVB take a third each
        tw.loc[invested, tw_value.columns] = tw_value.loc[invested].values * third
        tw.loc[invested, "CDI_ASSET"] = third
        tw.loc[invested, "IVVB11"] = third
        # pre-basket rows: park in CDI
        tw.loc[~invested, "CDI_ASSET"] = 1.0

        return r, tw

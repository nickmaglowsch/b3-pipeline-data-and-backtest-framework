"""
Strategy: Top N by Market Cap
==============================
Holds the N largest companies on B3 by point-in-time market cap. No earnings
or quality filters — pure size. Rebalance frequency is customizable (weekly,
monthly, quarterly) and weighting is market-cap or equal.

Reuses the SP-B3 machinery (backtests/strategies/sp500_b3.py): PIT
fundamentals snapshot, split-adjusted share counts, most-liquid share class
per company. Requires b3_pipeline.cvm_main to have populated fundamentals_pit.
"""
from __future__ import annotations

import pandas as pd

from backtests.core.strategy_base import (
    StrategyBase,
    ParameterSpec,
    COMMON_START_DATE,
    COMMON_END_DATE,
    COMMON_REBALANCE_FREQ,
)
from backtests.strategies.sp500_b3 import (
    DEFAULT_MIN_ADTV,
    LIQUIDITY_WINDOW,
    compute_weights,
    infer_volume_scale,
    load_fundamentals_pit_raw,
    load_stock_actions,
    select_constituents,
)


class TopMarketCapStrategy(StrategyBase):
    """Top N companies by PIT market cap, cap- or equal-weighted."""

    @property
    def name(self) -> str:
        return "Top Market Cap"

    @property
    def description(self) -> str:
        return (
            "Holds the N largest B3 companies by point-in-time market cap "
            "(split-adjusted shares x raw close, most liquid share class per "
            "company). No earnings filters. Customizable rebalance frequency; "
            "market-cap or equal weighting. Requires fundamentals_pit."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "top_n", "Number of Stocks", "int", 20,
                description="Hold the N largest companies by market cap",
                min_value=1, max_value=100, step=1,
            ),
            ParameterSpec(
                "weighting", "Weighting Scheme", "choice", "market_cap",
                description="market_cap = cap-weighted; equal = 1/N",
                choices=["market_cap", "equal"],
            ),
            ParameterSpec(
                "min_adtv", "Min Median Volume (BRL)", "float", DEFAULT_MIN_ADTV,
                description="Minimum 63-trading-day median daily financial volume",
                min_value=0.0, step=1e5,
            ),
            ParameterSpec(
                "db_path", "Database Path", "str", "b3_market_data.sqlite",
                description="Path to the B3 SQLite database (fundamentals_pit / stock_actions)",
            ),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        close_px = shared_data["close_px"]   # daily raw close (ffilled)
        fin_vol = shared_data["fin_vol"]     # daily financial volume

        top_n = int(params.get("top_n", 20))
        weighting = str(params.get("weighting", "market_cap"))
        min_adtv = float(params.get("min_adtv", DEFAULT_MIN_ADTV))
        db_path = str(params.get("db_path", "b3_market_data.sqlite"))

        fund = load_fundamentals_pit_raw(db_path)
        actions = load_stock_actions(db_path)

        scale = infer_volume_scale(fin_vol)
        med63 = fin_vol.fillna(0.0).rolling(LIQUIDITY_WINDOW).median() * scale

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        daily_idx = close_px.index
        started = False  # wait until a full basket exists (early years have
                         # sparse fundamentals coverage — a 2-stock "top 20"
                         # portfolio is not the requested strategy)

        for t_cal in ret.index:
            td = daily_idx.asof(t_cal)  # last trading day <= calendar rebalance date
            if pd.isna(td):
                continue
            sel = select_constituents(
                td, fund, close_px.loc[td], med63.loc[td], actions,
                min_market_cap=0.0, min_adtv=min_adtv, require_earnings=False,
            )
            if not started:
                if len(sel) < top_n:
                    continue
                started = True
            if sel.empty:
                continue
            sel = sel.nlargest(top_n, "market_cap")
            w = compute_weights(sel.set_index("ticker")["market_cap"], weighting)
            cols = w.index.intersection(tw.columns)
            if len(cols) == 0:
                continue
            w = w[cols] / w[cols].sum()
            # same trade-at-selection-close timing as SP500B3Strategy
            tw.loc[t_cal, cols] = w.values

        return ret, tw

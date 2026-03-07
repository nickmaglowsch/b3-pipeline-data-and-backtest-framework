"""
Strategy: LowPE — Low P/E Ratio Selection

Selects the N companies with the lowest positive P/E ratio (within a configurable
[min_pe, max_pe] band) and weights them equally.

Uses f_pe_ratio_dyn from shared_data — the monthly snapshot P/E computed with the
actual month-end closing price rather than the filing-date price. Falls back to
f_pe_ratio (filing-date-based stored P/E) when the monthly snapshot is not available
(i.e., the CVM pipeline has not yet been re-run after Task 04 integration).

Requires: shared_data["f_pe_ratio_dyn"] (or "f_pe_ratio" as fallback).
The backtest_service detects this via needs_fundamentals = True.

Important: this strategy requires the CVM fundamentals monthly snapshot table
(fundamentals_monthly) to be populated by running:
    python -m b3_pipeline.cvm_main
After the initial setup, re-run whenever new CVM DFP/ITR filings are downloaded.
"""
from __future__ import annotations

import pandas as pd

from backtests.core.strategy_base import (
    StrategyBase,
    ParameterSpec,
    COMMON_START_DATE,
    COMMON_END_DATE,
    COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE,
    COMMON_SLIPPAGE,
    COMMON_MIN_ADTV,
    COMMON_REBALANCE_FREQ,
    COMMON_MONTHLY_SALES_EXEMPTION,
)


class LowPEStrategy(StrategyBase):
    """
    Low P/E ratio selection strategy.

    For each rebalance period, selects the N companies with the lowest P/E ratio
    within the [min_pe, max_pe] band, after applying ADTV and price filters.
    Stocks are weighted equally (1/N).

    Uses f_pe_ratio_dyn (month-end-price-based P/E from the fundamentals_monthly
    snapshot) for more accurate and current valuations. Falls back to the
    filing-date-based f_pe_ratio if the monthly snapshot has not been populated.

    Requires the CVM fundamentals pipeline to be run with:
        python -m b3_pipeline.cvm_main
    The strategy will return zero weights for all periods if fundamentals data
    is missing.
    """

    # Signals to the backtest service that this strategy needs CVM fundamentals data.
    needs_fundamentals: bool = True

    @property
    def name(self) -> str:
        return "LowPE"

    @property
    def description(self) -> str:
        return (
            "Low P/E ratio selection strategy. "
            "Selects the N companies with the lowest positive P/E ratio within "
            "[min_pe, max_pe], weighted equally. "
            "Uses monthly-snapshot P/E (f_pe_ratio_dyn, computed with the actual "
            "month-end price) when available, falling back to the filing-date P/E. "
            "Requires the CVM fundamentals monthly snapshot (fundamentals_monthly "
            "table) to be populated by running the CVM pipeline."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE,
            COMMON_SLIPPAGE,
            COMMON_MIN_ADTV,
            COMMON_REBALANCE_FREQ,
            COMMON_MONTHLY_SALES_EXEMPTION,
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                description="Exclude stocks trading below this price (pennystock filter)",
                min_value=0.0, max_value=10.0, step=0.5,
            ),
            ParameterSpec(
                "n_stocks", "Number of Stocks", "int", 10,
                description="Number of lowest-P/E stocks to hold",
                min_value=1, max_value=50, step=1,
            ),
            ParameterSpec(
                "min_pe", "Min P/E Ratio", "float", 1.0,
                description=(
                    "Exclude companies with P/E below this floor. "
                    "Filters out near-zero earnings and data quality issues "
                    "(e.g., P/E of 0.1 likely reflects a one-off gain, not sustainable earnings)."
                ),
                min_value=0.0, max_value=5.0, step=0.5,
            ),
            ParameterSpec(
                "max_pe", "Max P/E Ratio", "float", 30.0,
                description="Exclude companies with P/E above this ceiling (value universe filter)",
                min_value=5.0, max_value=100.0, step=5.0,
            ),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate target weights based on lowest P/E selection.

        Returns:
            (ret, target_weights) — both DataFrames with the same DatetimeIndex
            and ticker columns.
        """
        ret = shared_data["ret"]
        adtv = shared_data.get("adtv", pd.DataFrame())
        raw_close = shared_data.get("raw_close", pd.DataFrame())

        # Use monthly-snapshot P/E if available; fall back to filing-date P/E.
        # The _dyn key is populated by build_shared_data() when fundamentals_monthly
        # has been materialized by the pipeline. The stored f_pe_ratio is the
        # filing-date-based fallback for older pipeline versions.
        if "f_pe_ratio_dyn" in shared_data and not shared_data["f_pe_ratio_dyn"].empty:
            f_pe = shared_data["f_pe_ratio_dyn"]
        else:
            # Fallback to filing-date-based stored P/E (no monthly snapshot yet)
            f_pe = shared_data.get("f_pe_ratio", pd.DataFrame())

        min_adtv = float(params.get("min_adtv", 1_000_000))
        min_price = float(params.get("min_price", 1.0))
        n_stocks = int(params.get("n_stocks", 10))
        min_pe = float(params.get("min_pe", 1.0))
        max_pe = float(params.get("max_pe", 30.0))
        min_stocks = int(params.get("min_stocks", 3))

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

        for i in range(1, len(ret)):
            prev_dt = ret.index[i - 1]

            # ── Retrieve P/E row ───────────────────────────────────────────────
            if f_pe.empty or prev_dt not in f_pe.index:
                continue
            pe_row = f_pe.loc[prev_dt].dropna()

            # ── Filter: positive P/E within [min_pe, max_pe] ──────────────────
            pe_valid = pe_row[(pe_row >= min_pe) & (pe_row <= max_pe)]

            # ── Liquidity filter ───────────────────────────────────────────────
            if not adtv.empty and prev_dt in adtv.index:
                adtv_r = adtv.loc[prev_dt]
                liq_ok = adtv_r.reindex(pe_valid.index, fill_value=0) >= min_adtv
                pe_valid = pe_valid[liq_ok]

            # ── Price filter ───────────────────────────────────────────────────
            if not raw_close.empty and prev_dt in raw_close.index:
                rc_r = raw_close.loc[prev_dt]
                price_ok = rc_r.reindex(pe_valid.index, fill_value=0) >= min_price
                pe_valid = pe_valid[price_ok]

            # ── Only keep tickers in ret.columns ──────────────────────────────
            pe_valid = pe_valid[pe_valid.index.isin(ret.columns)]

            if len(pe_valid) < min_stocks:
                continue  # not enough candidates — all weights stay at zero

            # ── Select n_stocks with lowest P/E ───────────────────────────────
            selected = pe_valid.nsmallest(n_stocks).index.tolist()

            w = 1.0 / len(selected)
            for ticker in selected:
                if ticker in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(ticker)] = w

        return ret, tw

"""
Strategy: ValueQuality — Composite Value + Quality (P/B + ROE)

Ranks stocks by:
  - Value leg: inverse P/B rank (lower P/B = cheaper = higher rank)
  - Quality leg: ROE rank (higher ROE = better quality = higher rank)
  - Composite = pb_weight * value_rank + (1 - pb_weight) * quality_rank

Requires include_fundamentals=True in build_shared_data().
The backtest_service detects this via needs_fundamentals = True.
"""
from __future__ import annotations

import numpy as np
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


class ValueQualityStrategy(StrategyBase):
    """
    Composite value + quality strategy.

    Value leg: low P/B (price-to-book) — cheaper stocks rank higher.
    Quality leg: high ROE (return on equity) — profitable companies rank higher.

    Typical fundamentals coverage is sparse (only publicly-listed Brazilian equities
    that file DFP/ITR with CVM), so a wider top_pct (default 20%) is used.

    Requires: shared_data["f_pb_ratio_dyn"], shared_data["f_net_income"], shared_data["f_equity"]
    """

    # Signals to the backtest service that this strategy needs CVM fundamentals data.
    needs_fundamentals: bool = True

    @property
    def name(self) -> str:
        return "ValueQuality"

    @property
    def description(self) -> str:
        return (
            "Composite Value + Quality strategy. "
            "Ranks stocks by P/B (value, lower is better) and ROE (quality, higher is better). "
            "Requires CVM fundamentals data (DFP/ITR) to be loaded in the DB."
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
                description="Exclude stocks trading below this price",
                min_value=0.0, max_value=10.0, step=0.5,
            ),
            ParameterSpec(
                "top_pct", "Top Percentile", "float", 0.20,
                description=(
                    "Fraction of eligible universe to select. "
                    "20% default (wider than technical strategies due to sparser fundamentals coverage)."
                ),
                min_value=0.05, max_value=0.50, step=0.05,
            ),
            ParameterSpec(
                "pb_weight", "P/B Weight (Value)", "float", 0.5,
                description="Weight given to the P/B (value) rank; (1 - pb_weight) goes to ROE (quality)",
                min_value=0.0, max_value=1.0, step=0.1,
            ),
            ParameterSpec(
                "max_pb", "Max P/B Filter", "float", 5.0,
                description="Exclude stocks with P/B ratio above this threshold (avoids overpriced financials)",
                min_value=1.0, max_value=20.0, step=0.5,
            ),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate target weights based on composite value + quality rank.

        Returns:
            (ret, target_weights) — both DataFrames with the same DatetimeIndex and ticker columns.
        """
        ret = shared_data["ret"]
        adtv = shared_data.get("adtv", pd.DataFrame())
        raw_close = shared_data.get("raw_close", pd.DataFrame())

        # Fundamentals (dynamic P/B computed from raw monthly inputs in build_shared_data)
        f_pb = shared_data.get("f_pb_ratio_dyn", pd.DataFrame())
        f_net_income = shared_data.get("f_net_income", pd.DataFrame())
        f_equity = shared_data.get("f_equity", pd.DataFrame())

        min_adtv = float(params.get("min_adtv", 1_000_000))
        min_price = float(params.get("min_price", 1.0))
        top_pct = float(params.get("top_pct", 0.20))
        pb_weight = float(params.get("pb_weight", 0.5))
        roe_weight = 1.0 - pb_weight
        max_pb = float(params.get("max_pb", 5.0))

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

        for i in range(1, len(ret)):
            dt = ret.index[i]
            prev_dt = ret.index[i - 1]

            # ── Retrieve per-period fundamentals ──────────────────────────────
            def _row(df: pd.DataFrame) -> pd.Series:
                if df.empty or prev_dt not in df.index:
                    return pd.Series(dtype=float)
                return df.loc[prev_dt]

            pb_row = _row(f_pb)
            ni_row = _row(f_net_income)
            eq_row = _row(f_equity)

            if pb_row.empty and ni_row.empty:
                continue

            # ── Compute ROE ────────────────────────────────────────────────────
            # roe = net_income / equity; handle zero/negative equity as NaN
            if not ni_row.empty and not eq_row.empty:
                # Align tickers
                common_tickers = ni_row.index.intersection(eq_row.index)
                ni_aligned = ni_row.reindex(common_tickers)
                eq_aligned = eq_row.reindex(common_tickers).replace(0, float("nan"))
                eq_aligned[eq_aligned < 0] = float("nan")
                roe_row = ni_aligned / eq_aligned
            else:
                roe_row = pd.Series(dtype=float)

            # ── Apply P/B filters ──────────────────────────────────────────────
            if not pb_row.empty:
                pb_filtered = pb_row.copy()
                pb_filtered[pb_filtered > max_pb] = float("nan")  # too expensive
                pb_filtered[pb_filtered <= 0] = float("nan")       # negative equity
            else:
                pb_filtered = pd.Series(dtype=float)

            # ── Rank signals ───────────────────────────────────────────────────
            # Value: lower P/B = higher rank (1 - pct_rank)
            if not pb_filtered.empty:
                value_rank = 1.0 - pb_filtered.rank(pct=True)
            else:
                value_rank = pd.Series(dtype=float)

            # Quality: higher ROE = higher rank
            if not roe_row.empty:
                quality_rank = roe_row.rank(pct=True)
            else:
                quality_rank = pd.Series(dtype=float)

            # ── Composite rank ─────────────────────────────────────────────────
            if not value_rank.empty and not quality_rank.empty:
                all_tickers = value_rank.index.union(quality_rank.index)
                vr = value_rank.reindex(all_tickers)
                qr = quality_rank.reindex(all_tickers)
                composite = vr.multiply(pb_weight).add(qr.multiply(roe_weight))
            elif not value_rank.empty:
                composite = value_rank
            elif not quality_rank.empty:
                composite = quality_rank
            else:
                continue

            # ── Liquidity and price filter ─────────────────────────────────────
            if not adtv.empty and prev_dt in adtv.index:
                adtv_r = adtv.loc[prev_dt]
                composite = composite[composite.index.isin(ret.columns)]
                liquidity_ok = adtv_r.reindex(composite.index, fill_value=0) >= min_adtv
                composite = composite[liquidity_ok]

            if not raw_close.empty and prev_dt in raw_close.index:
                rc_r = raw_close.loc[prev_dt]
                price_ok = rc_r.reindex(composite.index, fill_value=0) >= min_price
                composite = composite[price_ok]

            # Only keep tickers that are actually in ret columns
            composite = composite[composite.index.isin(ret.columns)].dropna()

            if len(composite) < 3:
                continue

            # ── Select top pct ─────────────────────────────────────────────────
            n = max(3, int(len(composite) * top_pct))
            selected = composite.nlargest(n).index.tolist()

            w = 1.0 / len(selected)
            for ticker in selected:
                if ticker in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(ticker)] = w

        return ret, tw

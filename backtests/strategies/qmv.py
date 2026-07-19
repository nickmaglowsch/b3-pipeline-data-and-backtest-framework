"""
Strategy: QMV — Quality Momentum with Vol-Targeted CDI Overlay
===============================================================
Three stacked, individually-evidenced components:

1. Risk-adjusted momentum (12-period log return skipping the most recent
   period, divided by volatility) — the repo's MomSharpe signal.
2. Point-in-time quality gate: positive TTM net income AND positive equity
   (fundamentals_monthly via f_net_income_ttm_m / f_equity_m). Momentum
   portfolios on B3 otherwise load up on distressed turnaround names.
3. Continuous volatility targeting (Moreira & Muir 2017): equity fraction =
   clip(target_vol / realized IBOV vol, 0, 1), remainder in CDI. In Brazil the
   cash leg earns CDI (historically 10-14% a.a.), so de-risking in vol spikes
   costs little and cuts drawdowns. Unlike the existing regime strategies this
   is a continuous dial, not a binary switch.

All defaults chosen a priori (top 10, 12m lookback, 15% vol target) — not
tuned on the backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)

TRADING_DAYS_YEAR = 252


class QMVStrategy(StrategyBase):
    needs_fundamentals: bool = True

    @property
    def name(self) -> str:
        return "QMV"

    @property
    def description(self) -> str:
        return (
            "Quality Momentum with Vol-Targeted CDI Overlay. Top-N risk-adjusted "
            "momentum stocks gated on positive PIT TTM earnings and equity; "
            "equity exposure scaled by target_vol / realized IBOV vol, "
            "remainder in CDI."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "top_n", "Number of Stocks", "int", 10,
                description="Momentum stocks held (equal weight within equity sleeve)",
                min_value=5, max_value=30, step=1,
            ),
            ParameterSpec(
                "lookback", "Momentum Lookback (periods)", "int", 12,
                description="Periods for momentum and volatility signals (skip-1 applied)",
                min_value=3, max_value=36, step=1,
            ),
            ParameterSpec(
                "target_vol", "Target Volatility (ann.)", "float", 0.15,
                description="Annualized vol target; equity fraction = target / realized IBOV vol",
                min_value=0.05, max_value=0.40, step=0.01,
            ),
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                min_value=0.0, max_value=10.0, step=0.5,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        log_ret = shared_data["log_ret"]
        has_glitch = shared_data["has_glitch"]
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]
        # 20d std of daily IBOV returns — align by label to the strategy calendar
        # (Yahoo's ^BVSP calendar can start/end offset from ret.index; positional
        # indexing would misalign the vol dial for the whole backtest)
        ibov_vol = shared_data["ibov_vol_monthly"].reindex(ret.index)
        f_ni_ttm = shared_data.get("f_net_income_ttm_m", pd.DataFrame())
        f_equity = shared_data.get("f_equity_m", pd.DataFrame())

        min_adtv = params.get("min_adtv", 2_000_000)
        min_price = params.get("min_price", 1.0)
        lookback = int(params.get("lookback", 12))
        top_n = int(params.get("top_n", 10))
        target_vol = float(params.get("target_vol", 0.15))

        # risk-adjusted momentum (skip most recent period)
        mom = log_ret.shift(1).rolling(lookback).sum()
        vol = ret.shift(1).rolling(lookback).std()
        sig = mom / vol
        sig[has_glitch == 1] = np.nan

        # map universe ticker -> fundamentals column via 4-char root
        # (f_*_m columns are one best-ADTV ticker per company)
        root_col: dict[str, str] = {}
        for c in f_ni_ttm.columns:
            root_col.setdefault(str(c)[:4], c)

        ann = np.sqrt(TRADING_DAYS_YEAR)

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        tw["CDI_ASSET"] = 0.0
        r = ret.copy()
        r["CDI_ASSET"] = cdi_monthly
        r["IBOV"] = ibov_ret

        cdi_col = tw.columns.get_loc("CDI_ASSET")

        for i in range(lookback + 2, len(ret)):
            # ── vol-targeted equity fraction (previous period's IBOV vol) ────
            eq_frac = 1.0
            v = ibov_vol.iloc[i - 1]  # safe: reindexed to ret.index above
            if pd.notna(v) and v > 0:
                eq_frac = float(np.clip(target_vol / (float(v) * ann), 0.0, 1.0))

            # ── selection ────────────────────────────────────────────────────
            sig_row = sig.iloc[i - 1]
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            mask = (adtv_r >= min_adtv) & (raw_r >= min_price)
            valid = sig_row[mask].dropna()
            valid = valid[np.isfinite(valid)]

            # quality gate: positive PIT TTM net income AND positive equity
            if not f_ni_ttm.empty and i - 1 < len(f_ni_ttm):
                ni_row = f_ni_ttm.iloc[i - 1]
                eq_row = f_equity.iloc[i - 1] if not f_equity.empty else pd.Series(dtype=float)

                def _quality(tk: str) -> bool:
                    col = root_col.get(tk[:4])
                    if col is None:
                        return False
                    ni = ni_row.get(col)
                    eq = eq_row.get(col)
                    return pd.notna(ni) and ni > 0 and pd.notna(eq) and eq > 0

                valid = valid[[_quality(t) for t in valid.index]]

            if len(valid) < 5:
                tw.iloc[i, cdi_col] = 1.0
                continue

            sel = valid.nlargest(top_n).index
            w = eq_frac / len(sel)
            for t in sel:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w
            tw.iloc[i, cdi_col] = 1.0 - eq_frac

        return r, tw

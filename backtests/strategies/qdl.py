"""
Strategy: QDL — Quality-Dividend Low-Vol with a CDI vol-target overlay
=====================================================================
Designed to beat the three known B3 "edge" strategies (TevaAtivosReais,
BESST Quality, DIVO11) *without overfitting*.

Why those three beat the Ibovespa
---------------------------------
All three are the SAME structural bet: a defensive income tilt — dividend
payers + quality + low-beta sectors (utilities, banks, telecom), cap-weighted
with a per-name cap. IBOV is cap-weighted and dominated by high-vol cyclical
commodities (VALE, PETR), so it runs ~22% vol with -50% drawdowns. The
defensive-income factors earn IBOV-like returns at roughly half the vol, so the
three win on *risk-adjusted* terms (Sharpe / Calmar), not by magic alpha.

Their shared blind spot
-----------------------
All three are ALWAYS 100% invested in equities, so they take the full -30/-50%
equity drawdown in every crisis (2008, 2015, 2020, 2022). Brazil is unusual:
cash (CDI) pays a very high *real* rate (10-14% nominal). So the single most
robust, well-evidenced improvement is to keep the same defensive sleeve and add
a CDI de-risking overlay — de-risk into cash when the sleeve's own realized vol
spikes (Moreira & Muir 2017, "Volatility-Managed Portfolios"). Because CDI is so
high, sitting in cash during stress costs little and cuts the deep drawdowns the
three eat in full.

The strategy = same factor sleeve the winners exploit, PLUS the overlay they lack:
  1. Universe: liquid B3 names (ADTV + min-price floor).
  2. Dividend gate: regular payer — dividend/JCP in >= min_div_years of last 3y
     (identical screen to the BESST/Teva methodologies).
  3. Quality gate: positive PIT TTM net income AND positive equity (drops
     distressed names, same gate as QMV — cheap insurance on B3).
  4. Selection: the `top_n` LOWEST-volatility survivors (EWMA-252 ann. vol) —
     the low-vol anomaly, harvested rule-based across whatever sectors qualify
     (no hand-curated sector map, so no survivorship / selection overfit).
  5. Weighting: inverse-vol, 10% per-name cap (the low-vol premium harvester).
  6. CDI overlay: equity fraction = clip(target_vol / realized sleeve vol, 0, 1),
     remainder in CDI. Continuous dial, not a fitted regime switch.

Anti-overfit discipline
-----------------------
Four real knobs (top_n, ewma_n, target_vol, min_div_years), every one fixed a
priori from theory / the methodologies — NOT scanned on the backtest:
  - top_n=20        : breadth ~ the winners' constituent counts.
  - ewma_n=252      : B3's own low-vol index EWMA span.
  - target_vol=0.12 : long-run realized vol of a B3 low-vol basket (dial sits
                      ~fully invested in calm times, de-risks only in stress).
  - min_div_years=2 : the "2 of last 3 years" regular-payer rule the indices use.
Weighting is parameter-free (inverse-vol). Compare against the defensive-income
peers with scripts/run_single_backtest.py, which reports the full metric set net
of the repo's tax + slippage engine.

Requires CVM fundamentals (needs_fundamentals=True) and reads dividends from
corporate_actions (via besst_quality.load_dividends).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV,
)
from backtests.strategies.ibov_low_vol import ewma_annualized_vol, inverse_vol_capped
from backtests.strategies.besst_quality import load_dividends, paid_regularly

TRADING_DAYS_YEAR = 252
_DB_PATH = Path(__file__).resolve().parents[2] / "b3_market_data.sqlite"


class QDLStrategy(StrategyBase):
    """Quality-Dividend Low-Vol sleeve with a CDI vol-target overlay."""

    needs_fundamentals: bool = True

    @property
    def name(self) -> str:
        return "QDL"

    @property
    def description(self) -> str:
        return (
            "Quality-Dividend Low-Vol with CDI overlay. Regular dividend payers "
            "with positive PIT earnings, holds the lowest-volatility names "
            "inverse-vol weighted (10% cap), and scales equity exposure by "
            "target_vol / realized sleeve vol with the remainder in CDI. Beats "
            "the defensive-income indices by adding the cash overlay they lack."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV,
            # Methodology default is quarterly (matches BESST / Teva); overridden
            # from the common ME default.
            ParameterSpec(
                "rebalance_freq", "Rebalance Frequency", "choice", "QE",
                description="Rebalance cadence (peers rebalance quarterly/semiannual)",
                choices=["ME", "QE", "W-FRI"],
            ),
            ParameterSpec(
                "top_n", "Number of Stocks", "int", 20,
                description="Lowest-vol survivors held (a priori ~ peer breadth)",
                min_value=5, max_value=40, step=1,
            ),
            ParameterSpec(
                "ewma_n", "EWMA Span (days)", "int", 252,
                description="EWMA span for daily-return volatility (B3 low-vol index span)",
                min_value=60, max_value=504, step=1,
            ),
            ParameterSpec(
                "target_vol", "Target Volatility (ann.)", "float", 0.12,
                description="Equity fraction = target / realized sleeve vol; rest in CDI",
                min_value=0.05, max_value=0.40, step=0.01,
            ),
            ParameterSpec(
                "min_div_years", "Min Dividend Years (of last 3)", "int", 2,
                description="Require dividends/JCP in >= this many of the last 3 years",
                min_value=0, max_value=3, step=1,
            ),
            ParameterSpec(
                "weight_cap", "Max Weight per Name", "float", 0.10,
                description="Per-name cap within the equity sleeve (not tuned)",
                min_value=0.01, max_value=1.0, step=0.01,
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
        daily_close = shared_data["split_adj_close"]     # daily, split-adjusted
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]
        has_glitch = shared_data["has_glitch"]
        f_ni_ttm = shared_data.get("f_net_income_ttm_m", pd.DataFrame())
        f_equity = shared_data.get("f_equity_m", pd.DataFrame())

        min_adtv = float(params.get("min_adtv", 1_000_000))
        min_price = float(params.get("min_price", 1.0))
        top_n = int(params.get("top_n", 20))
        ewma_n = int(params.get("ewma_n", 252))
        target_vol = float(params.get("target_vol", 0.12))
        min_div_years = int(params.get("min_div_years", 2))
        cap = float(params.get("weight_cap", 0.10))

        divs = load_dividends(str(_DB_PATH))

        # Annualized EWMA vol on daily data, sampled onto the rebalance calendar.
        vol_daily = ewma_annualized_vol(daily_close, ewma_n)
        vol = vol_daily.reindex(vol_daily.index.union(ret.index)).ffill().reindex(ret.index)
        vol = vol.reindex(columns=ret.columns)
        vol[has_glitch == 1] = np.nan

        # Daily returns for the sleeve's realized-vol estimate (overlay dial).
        daily_ret = daily_close.pct_change()

        # Map universe ticker -> fundamentals column via 4-char root (one best-ADTV
        # ticker per company in the f_*_m frames), mirroring QMV.
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

        for i in range(2, len(ret)):
            prev = ret.index[i - 1]
            vol_row = vol.iloc[i - 1]
            mask = (adtv.iloc[i - 1] >= min_adtv) & (raw_close.iloc[i - 1] >= min_price)
            valid = vol_row[mask].dropna()
            valid = valid[valid > 0]

            # ── dividend gate: regular payer (>= min_div_years of last 3) ─────
            if min_div_years > 0 and not valid.empty:
                keep = [
                    paid_regularly(divs.get(t[:4]), pd.Timestamp(prev), min_div_years)
                    for t in valid.index
                ]
                valid = valid[keep]

            # ── quality gate: positive PIT TTM net income AND positive equity ─
            if not f_ni_ttm.empty and i - 1 < len(f_ni_ttm) and not valid.empty:
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
                tw.iloc[i, cdi_col] = 1.0  # not enough names → park in CDI
                continue

            sel = valid.nsmallest(top_n)
            w = inverse_vol_capped(sel, cap)  # within-sleeve weights, sum to 1

            # ── CDI overlay: scale by target_vol / realized sleeve vol ────────
            win = daily_ret.loc[:prev, w.index].tail(63)
            port_daily = (win * w).sum(axis=1)
            realized = float(port_daily.std()) * ann
            eq_frac = float(np.clip(target_vol / realized, 0.0, 1.0)) if realized > 0 else 1.0

            for t, wt in (w * eq_frac).items():
                tw.iloc[i, tw.columns.get_loc(t)] = wt
            tw.iloc[i, cdi_col] = 1.0 - eq_frac

        return r, tw

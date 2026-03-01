"""
B3 Research-Driven Multi-Factor Composite Score Backtest
========================================================
Derived from Feature Importance Research (2026-02-28).

Uses ALL 8 features that were stable across both models (RF + XGB)
and all 3 target definitions:

  Stock-level factors (continuous composite score):
    1. Distance_to_MA200  — long-term trend positioning
    2. Rolling_vol_60d    — lower is better (inverted)
    3. ATR_14             — lower is better (inverted, normalized)
    4. Rank_volatility_20d — lower is better (inverted)
    5. Rank_volume         — higher is better (liquidity)

  Regime factors (binary equity/CDI switch):
    6. CDI_3m_change       — easing vs tightening
    7. Ibovespa_vol_20d    — calm vs stressed
    8. Ibovespa_return_20d — uptrend vs downtrend

Regime logic:
  If >= 2 of 3 regime signals are negative → 100% CDI
  Otherwise → top 10% stocks by composite score

Upgrade over existing multifactor_backtest.py:
  - 5 factors instead of 2 (mom + low vol)
  - Regime conditioning (CDI/IBOV based)
  - Factor weights derived from research importance rankings
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import download_benchmark, download_cdi_daily, load_b3_data
from core.metrics import build_metrics, display_metrics_table, value_to_ret
from core.plotting import plot_tax_backtest
from core.simulation import run_simulation

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
REBALANCE_FREQ = "ME"
PERIODS_PER_YEAR = 12
LOOKBACK_PERIODS = 12  # 1 year of monthly data

TOP_PCT = 0.10
TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000
MIN_PRICE = 1.0

# Factor weights: equal weight — research shows all 5 stock-level factors
# have nearly identical importance (~0.047-0.055 avg builtin importance)
W_DIST_MA200 = 0.20
W_LOW_VOL_60D = 0.20
W_LOW_ATR = 0.20
W_LOW_VOL_RANK = 0.20
W_LIQUIDITY = 0.20

# IBOV regime thresholds
IBOV_VOL_PERCENTILE = 0.70  # above = stressed
IBOV_TREND_WINDOW = 20  # trading days for IBOV return

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def generate_signals(adj_close, close_px, fin_vol, cdi_daily, ibov_px):
    """
    Generate monthly target weights using 5-factor composite score
    with 3-signal regime filter.
    """
    # Monthly resampled data
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()
    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    # ── REGIME SIGNALS ──

    # 1. CDI easing/tightening
    cdi_monthly = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1
    cdi_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # 2. IBOV volatility (calm vs stressed)
    ibov_daily_ret = ibov_px.pct_change()
    ibov_vol_20d = ibov_daily_ret.rolling(IBOV_TREND_WINDOW).std()
    ibov_vol_monthly = ibov_vol_20d.resample(REBALANCE_FREQ).last()
    ibov_vol_pctrank = ibov_vol_monthly.expanding(min_periods=12).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False
    )
    ibov_calm = ibov_vol_pctrank <= IBOV_VOL_PERCENTILE

    # 3. IBOV trend (positive 20d return)
    ibov_ret_20d = ibov_px.pct_change(IBOV_TREND_WINDOW)
    ibov_ret_monthly = ibov_ret_20d.resample(REBALANCE_FREQ).last()
    ibov_uptrend = ibov_ret_monthly > 0

    # ── STOCK-LEVEL FACTORS ──

    # Factor 1: Distance to MA200 (higher = stronger trend = better)
    ma200_daily = adj_close.rolling(200, min_periods=200).mean()
    ma200_monthly = ma200_daily.resample(REBALANCE_FREQ).last()
    dist_ma200 = px / ma200_monthly - 1

    # Factor 2: Rolling vol 60d (lower = better → invert)
    vol_60d = ret.rolling(5).std()  # 5 months ~ 60 trading days

    # Factor 3: ATR_14 normalized (lower = better → invert)
    # Approximate ATR from monthly high/low/close proxy
    # Use daily data for accuracy, resample at month-end
    daily_ret = adj_close.pct_change()
    atr_proxy = daily_ret.abs().ewm(span=14, min_periods=14).mean()
    atr_monthly = atr_proxy.resample(REBALANCE_FREQ).last()

    # Factor 4: Cross-sectional vol rank 20d (lower vol rank = better → invert)
    vol_20d = ret.rolling(2).std()  # 2 months ~ 20 trading days

    # Factor 5: Volume rank (higher = more liquid = better)
    # Already have adtv, just rank it

    # Glitch filter
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(LOOKBACK_PERIODS).max()

    # ── BUILD TARGET WEIGHTS ──
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    target_weights["CDI_ASSET"] = 0.0

    regime_log = []
    start_idx = max(LOOKBACK_PERIODS + 2, 14)

    for i in range(start_idx, len(ret)):
        # Regime check (all lagged)
        sig_easing = bool(cdi_easing.iloc[i]) if i < len(cdi_easing) else False
        sig_calm = bool(ibov_calm.iloc[i - 1]) if i - 1 < len(ibov_calm) else False
        sig_uptrend = (
            bool(ibov_uptrend.iloc[i - 1]) if i - 1 < len(ibov_uptrend) else False
        )

        # Count positive regime signals (need >= 2 of 3 to stay in equities)
        regime_score = int(sig_easing) + int(sig_calm) + int(sig_uptrend)

        if regime_score < 2:
            # Unfavorable regime: park in CDI
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            regime_log.append("cdi")
            continue

        regime_log.append("equity")

        # Compute cross-sectional ranks for each factor (using data at i-1)
        # All ranks: higher = better (pct=True gives 0-1 range)

        # Distance to MA200: higher = better
        r_ma200 = dist_ma200.iloc[i - 1].rank(pct=True)

        # Vol 60d: lower = better → invert by ranking negative
        r_vol60 = (-vol_60d.iloc[i - 1]).rank(pct=True)

        # ATR: lower = better → invert
        r_atr = (-atr_monthly.iloc[i - 1]).rank(pct=True)

        # Vol 20d rank: lower = better → invert
        r_vol20 = (-vol_20d.iloc[i - 1]).rank(pct=True)

        # Volume rank: higher = better
        r_volume = adtv.iloc[i - 1].rank(pct=True)

        # Composite score (weighted sum of percentile ranks)
        composite = (
            W_DIST_MA200 * r_ma200
            + W_LOW_VOL_60D * r_vol60
            + W_LOW_ATR * r_atr
            + W_LOW_VOL_RANK * r_vol20
            + W_LIQUIDITY * r_volume
        )

        # Apply filters
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]
        glitch_row = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()

        valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= MIN_PRICE)
        if len(glitch_row) > 0:
            valid_mask = valid_mask & (glitch_row != 1)

        valid = composite[valid_mask].dropna()

        if len(valid) < 5:
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        n_sel = max(1, int(len(valid) * TOP_PCT))
        sel = valid.nlargest(n_sel).index.tolist()

        weight = 1.0 / len(sel)
        for t in sel:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight

    return ret, target_weights, regime_log


def main():
    print("\n" + "=" * 70)
    print("  B3 RESEARCH-DRIVEN MULTI-FACTOR COMPOSITE BACKTEST (15% CGT)")
    print("  5 Stock Factors + 3 Regime Signals from Feature Importance Research")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n  Generating target weights (5-factor composite + regime filter)...")
    ret, target_weights, regime_log = generate_signals(
        adj_close, close_px, fin_vol, cdi_daily, ibov_px
    )

    # Add CDI and IBOV returns
    cdi_monthly = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1
    ret["CDI_ASSET"] = cdi_monthly
    ret["IBOV"] = ibov_ret
    ret = ret.fillna(0.0)

    print(f"\n  Running simulation ({REBALANCE_FREQ})...")
    result = run_simulation(
        returns_matrix=ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="ResearchFactor",
        monthly_sales_exemption=20_000,
    )

    common = result["pretax_values"].index.intersection(ibov_ret.index)
    pretax_val = result["pretax_values"].loc[common]
    aftertax_val = result["aftertax_values"].loc[common]
    ibov_ret = ibov_ret.loc[common]
    cdi_ret = cdi_ret.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = result["tax_paid"].sum()

    m_pretax = build_metrics(pretax_ret, "ResFactor Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret, "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    # Regime summary
    if regime_log:
        from collections import Counter

        counts = Counter(regime_log)
        total = len(regime_log)
        print(f"\n  Regime Summary ({total} months):")
        for regime in ["equity", "cdi"]:
            c = counts.get(regime, 0)
            print(f"    {regime:<8s}: {c:3d} months ({c / total * 100:4.1f}%)")

    tw_equity = target_weights.drop(columns=["CDI_ASSET"], errors="ignore")
    n_stocks = tw_equity.gt(0).sum(axis=1)
    avg_when_active = n_stocks[n_stocks > 0].mean()
    print(f"    Avg stocks when in equity: {avg_when_active:.0f}")

    plot_tax_backtest(
        title=(
            f"Research-Driven Multi-Factor Composite (B3 Native)\n"
            f"5 Factors: MA200 + LowVol60d + LowATR + VolRank + Liquidity\n"
            f"Regime: 2-of-3 (CDI Easing, IBOV Calm, IBOV Uptrend) → Equity | else CDI\n"
            f"Top {int(TOP_PCT * 100)}%  ·  R${MIN_ADTV / 1_000_000:.0f}M+ ADTV  ·  "
            f"15% CGT + {SLIPPAGE * 100:.1f}% Slippage  ·  {START_DATE[:4]}–{END_DATE[:4]}"
        ),
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=total_tax,
        out_path="research_multifactor_backtest.png",
        cdi_ret=cdi_ret,
    )


if __name__ == "__main__":
    main()

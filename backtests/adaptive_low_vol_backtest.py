"""
B3 Volatility-Regime Adaptive Low Volatility Strategy Backtest
==============================================================
Derived from Feature Importance Research (2026-02-28).

Upgrade of the unconditional low_volatility_backtest.py.
Conditions the low-vol stock selection on two regime signals:

  - CDI_3m_change (#1 ranked feature): easing vs tightening cycle
  - Ibovespa_vol_20d (#2 ranked feature): calm vs stressed market

Regime logic:
  CALM + EASING  → aggressive: top 15% lowest-vol stocks (wider net)
  CALM + TIGHT   → moderate:   top 10% lowest-vol stocks (standard)
  STRESS + EASING → defensive: top 5% lowest-vol stocks (tight filter)
  STRESS + TIGHT  → risk-off:  100% CDI

Stock-level signal uses Rolling_vol_60d (#4) and Rank_volatility_20d (#7)
via a blended volatility score.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.simulation import run_simulation

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
REBALANCE_FREQ = "ME"
PERIODS_PER_YEAR = 12
LOOKBACK_YEARS = 1
LOOKBACK_PERIODS = int(LOOKBACK_YEARS * PERIODS_PER_YEAR)

# Regime thresholds
IBOV_VOL_LOOKBACK = 20          # trading days for IBOV realized vol
IBOV_VOL_PERCENTILE = 0.70      # above this percentile = "stressed"

# Stock selection percentiles per regime
PCT_AGGRESSIVE = 0.15           # calm + easing: wider net
PCT_MODERATE = 0.10             # calm + tight: standard
PCT_DEFENSIVE = 0.05            # stress + easing: tight filter
# stress + tight: 100% CDI

TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000
MIN_PRICE = 1.0

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def generate_signals(adj_close, close_px, fin_vol, cdi_daily, ibov_px):
    """
    Generate monthly target weights with regime-adaptive low-vol selection.
    """
    # Monthly resampled data
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()
    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    # ── Regime 1: CDI easing/tightening ──
    cdi_monthly = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1
    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # ── Regime 2: IBOV volatility (calm vs stressed) ──
    ibov_daily_ret = ibov_px.pct_change()
    ibov_vol_20d = ibov_daily_ret.rolling(IBOV_VOL_LOOKBACK).std()
    ibov_vol_monthly = ibov_vol_20d.resample(REBALANCE_FREQ).last()
    # Expanding percentile rank: is current vol high relative to history?
    ibov_vol_pctrank = ibov_vol_monthly.expanding(min_periods=12).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False
    )
    is_stressed = ibov_vol_pctrank > IBOV_VOL_PERCENTILE

    # ── Stock-level volatility signal ──
    # Blend: 60-day rolling vol (time-series) + 20-day cross-sectional rank
    vol_60d = ret.rolling(5).std()  # monthly proxy for ~60 trading days
    vol_20d = ret.rolling(2).std()  # monthly proxy for ~20 trading days

    # Cross-sectional rank of 20d vol (per date, lower = better)
    rank_vol_20d = vol_20d.rank(axis=1, pct=True)

    # Blended score: lower is better (negative so nlargest picks lowest vol)
    # 60% weight on longer vol (more stable), 40% on cross-sectional rank
    blended_vol = -(0.6 * vol_60d.rank(axis=1, pct=True) + 0.4 * rank_vol_20d)

    # Glitch filter
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(LOOKBACK_PERIODS).max()
    blended_vol[has_glitch == 1] = np.nan

    # ── Build target weights ──
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    target_weights["CDI_ASSET"] = 0.0

    regime_log = []
    start_idx = max(LOOKBACK_PERIODS + 1, 14)

    for i in range(start_idx, len(ret)):
        # Regime signals (lagged: use i-1 or earlier)
        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        stressed = bool(is_stressed.iloc[i - 1]) if i - 1 < len(is_stressed) else False

        # Determine selection percentile based on regime
        if stressed and not easing:
            # Worst regime: 100% CDI
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            regime_log.append("risk-off")
            continue
        elif stressed and easing:
            pct = PCT_DEFENSIVE
            regime = "defensive"
        elif not stressed and not easing:
            pct = PCT_MODERATE
            regime = "moderate"
        else:
            # calm + easing: best regime
            pct = PCT_AGGRESSIVE
            regime = "aggressive"

        regime_log.append(regime)

        # Stock selection using data up to i-1
        sig_row = blended_vol.iloc[i - 1]
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]

        valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= MIN_PRICE)
        valid = sig_row[valid_mask].dropna()

        if len(valid) < 5:
            # Not enough stocks: fallback to CDI
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        n_sel = max(1, int(len(valid) * pct))
        sel = valid.nlargest(n_sel).index.tolist()

        weight = 1.0 / len(sel)
        for t in sel:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight

    return ret, target_weights, regime_log


def main():
    print("\n" + "=" * 70)
    print("  B3 VOLATILITY-REGIME ADAPTIVE LOW VOL BACKTEST (15% CGT)")
    print("  Derived from Feature Importance Research")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n  Generating target weights (regime-adaptive low vol)...")
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
        name="Adaptive Low Vol",
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

    m_pretax = build_metrics(pretax_ret, "AdaptLowVol Pre-Tax", PERIODS_PER_YEAR)
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
        for regime in ["aggressive", "moderate", "defensive", "risk-off"]:
            c = counts.get(regime, 0)
            print(f"    {regime:<12s}: {c:3d} months ({c/total*100:4.1f}%)")

    # Avg stocks per regime
    tw_equity = target_weights.drop(columns=["CDI_ASSET"], errors="ignore")
    n_stocks = tw_equity.gt(0).sum(axis=1)
    avg_when_active = n_stocks[n_stocks > 0].mean()
    print(f"    Avg stocks when in equity: {avg_when_active:.0f}")

    plot_tax_backtest(
        title=(
            f"Volatility-Regime Adaptive Low Vol (B3 Native)\n"
            f"Calm+Easing→Top15% | Calm+Tight→Top10% | "
            f"Stress+Easing→Top5% | Stress+Tight→CDI\n"
            f"R${MIN_ADTV/1_000_000:.0f}M+ ADTV  ·  15% CGT + {SLIPPAGE*100:.1f}% Slippage  ·  "
            f"{START_DATE[:4]}–{END_DATE[:4]}"
        ),
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=total_tax,
        out_path="adaptive_low_vol_backtest.png",
        cdi_ret=cdi_ret,
    )


if __name__ == "__main__":
    main()

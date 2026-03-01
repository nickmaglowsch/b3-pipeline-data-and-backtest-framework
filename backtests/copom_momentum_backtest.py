"""
B3 COPOM Momentum Strategy Backtest
====================================
Regime switch: Smooth Momentum portfolio during easing cycles, CDI during tightening.
Easing = current month's CDI rate is lower than 3 months ago.

During easing  → top-decile stocks ranked by Information Ratio (smooth momentum)
During tightening → 100% CDI
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
LOOKBACK_YEARS = 1

period_map = {"ME": 12, "QE": 4, "YE": 1, "W": 52, "W-FRI": 52, "W-MON": 52}
PERIODS_PER_YEAR = period_map.get(REBALANCE_FREQ, 12)
LOOKBACK_PERIODS = int(LOOKBACK_YEARS * PERIODS_PER_YEAR)
SKIP_PERIODS = 1 if REBALANCE_FREQ == "ME" else 0

TOP_DECILE = 0.10
TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def generate_signals(adj_close, close_px, fin_vol, cdi_monthly):
    """
    Build target weights that combine COPOM regime detection with smooth momentum.
    Easing months → smooth momentum stock picks (top decile by info ratio)
    Tightening months → 100% CDI
    """
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()

    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    log_ret = np.log1p(ret)

    # ── Smooth Momentum Signal ──
    mom = log_ret.shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).sum()
    vol = log_ret.shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).std()
    signal = mom / vol

    # Glitch protection
    has_glitch = (
        ((ret > 1.0) | (ret < -0.90))
        .shift(SKIP_PERIODS)
        .rolling(LOOKBACK_PERIODS)
        .max()
    )
    signal[has_glitch == 1] = np.nan

    # ── COPOM Regime Detection ──
    # Easing = current month's CDI rate is lower than 3 months ago
    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # Add CDI as a tradable asset in the returns matrix
    ret["CDI_ASSET"] = cdi_monthly

    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

    start_idx = max(LOOKBACK_PERIODS + SKIP_PERIODS + 1, 5)  # need 5 for COPOM signal
    prev_sel = set()
    easing_months = 0
    tight_months = 0

    for i in range(start_idx, len(ret)):
        if is_easing.iloc[i]:
            # ── Easing: deploy smooth momentum portfolio ──
            easing_months += 1

            sig_row = signal.iloc[i - 1]
            adtv_row = adtv.iloc[i - 1]
            raw_close_row = raw_close.iloc[i - 1]

            valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)
            valid = sig_row[valid_mask].dropna()

            if len(valid) < 5:
                sel = prev_sel
            else:
                n_sel = max(1, int(len(valid) * TOP_DECILE))
                sel = set(valid.nlargest(n_sel).index)

            if not sel:
                # Fallback to CDI if no valid stocks
                target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
                continue

            weight_per_stock = 1.0 / len(sel)
            for t in sel:
                target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight_per_stock

            prev_sel = sel
        else:
            # ── Tightening: park everything in CDI ──
            tight_months += 1
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0

    return ret, target_weights, easing_months, tight_months


def main():
    print("\n" + "=" * 70)
    print("  B3 COPOM MOMENTUM STRATEGY BACKTEST (15% CGT)")
    print("  Easing → Smooth Momentum  |  Tightening → CDI")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    cdi_monthly = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    cdi_ret = cdi_monthly.copy()

    print("\n  Generating regime-aware momentum weights...")
    ret, target_weights, easing_months, tight_months = generate_signals(
        adj_close, close_px, fin_vol, cdi_monthly
    )
    ret = ret.fillna(0.0)

    print(f"\n  Running simulation ({REBALANCE_FREQ})...")
    result = run_simulation(
        returns_matrix=ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="COPOM Momentum",
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

    m_pretax = build_metrics(pretax_ret, "COPOM Mom Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret, "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    total_months = easing_months + tight_months
    print(f"\n  Regime Summary:")
    print(f"    Easing (Momentum):    {easing_months} months ({easing_months/max(total_months,1)*100:.0f}%)")
    print(f"    Tightening (CDI):     {tight_months} months ({tight_months/max(total_months,1)*100:.0f}%)")

    plot_tax_backtest(
        title=(
            f"COPOM Momentum Strategy (B3 Native)\n"
            f"Easing → Smooth Momentum (Top {int(TOP_DECILE*100)}%)  |  Tightening → CDI\n"
            f"15% CGT + {SLIPPAGE*100:.1f}% Slippage  ·  {START_DATE[:4]}–{END_DATE[:4]}"
        ),
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=total_tax,
        out_path="copom_momentum_backtest.png",
        cdi_ret=cdi_ret,
    )


if __name__ == "__main__":
    main()

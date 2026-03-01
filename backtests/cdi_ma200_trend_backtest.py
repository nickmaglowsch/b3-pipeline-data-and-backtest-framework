"""
B3 CDI Regime + MA200 Trend Filter Strategy Backtest
=====================================================
Derived from Feature Importance Research (2026-02-28).

Signal logic (uses the #1 and #5 ranked features):
  - CDI_3m_change (#1): When CDI cumulative 3-month return is FALLING
    (easing cycle), allocate to equities. Otherwise, park in CDI.
  - Distance_to_MA200 (#5): Among equities, only buy stocks trading
    ABOVE their 200-day moving average (positive long-term trend).

Rebalances monthly. Equal-weights qualifying stocks.
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

TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000  # R$1M minimum liquidity (matches research universe)
MIN_PRICE = 1.0  # R$1.00 minimum price
MA_WINDOW = 200  # 200 trading days

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def generate_signals(adj_close, close_px, fin_vol, cdi_daily):
    """
    Generate monthly target weights.

    Regime filter (CDI):
      - Compute monthly CDI return, then check if current month's CDI
        is lower than 3 months ago (easing = True).
      - If NOT easing: 100% CDI.

    Stock selection (MA200):
      - Compute 200-day MA on daily adj_close.
      - At each month-end, select stocks where adj_close > MA200
        AND pass liquidity/price filters.
      - Equal-weight the qualifying stocks.
    """
    # Monthly resampled data
    px_monthly = adj_close.resample(REBALANCE_FREQ).last()
    ret = px_monthly.pct_change()
    raw_close_monthly = close_px.resample(REBALANCE_FREQ).last()
    adtv_monthly = fin_vol.resample(REBALANCE_FREQ).mean()

    # CDI regime: is CDI rate falling? (easing cycle)
    cdi_monthly = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1
    # Compare current month CDI to 3 months ago
    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # MA200 on daily data, sampled at month-end
    ma200_daily = adj_close.rolling(MA_WINDOW, min_periods=MA_WINDOW).mean()
    ma200_monthly = ma200_daily.resample(REBALANCE_FREQ).last()

    # Above MA200 flag (using lagged data: month-end i-1 to avoid lookahead)
    above_ma200 = px_monthly > ma200_monthly

    # Glitch filter: discard stocks with extreme single-period returns
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(6).max()

    # Build target weights
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    target_weights["CDI_ASSET"] = 0.0

    # Need enough history for MA200 + CDI lookback
    start_idx = max(13, MA_WINDOW // 20 + 2)  # ~10 months for MA200 + CDI warmup

    for i in range(start_idx, len(ret)):
        # Regime check: use lagged CDI signal (data up to i-1)
        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False

        if not easing:
            # Tightening cycle: park in CDI
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        # Easing cycle: select stocks above MA200 with liquidity filter
        # All signals use data from i-1 (no lookahead)
        above_row = above_ma200.iloc[i - 1]
        adtv_row = adtv_monthly.iloc[i - 1]
        raw_close_row = raw_close_monthly.iloc[i - 1]
        glitch_row = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()

        # Filters
        valid_mask = (
            above_row.fillna(False)
            & (adtv_row >= MIN_ADTV)
            & (raw_close_row >= MIN_PRICE)
        )
        if len(glitch_row) > 0:
            valid_mask = valid_mask & (glitch_row != 1)

        valid_tickers = valid_mask[valid_mask].index.tolist()

        if len(valid_tickers) == 0:
            # No stocks qualify: fallback to CDI
            target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        weight = 1.0 / len(valid_tickers)
        for t in valid_tickers:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight

    return ret, target_weights


def main():
    print("\n" + "=" * 70)
    print("  B3 CDI REGIME + MA200 TREND FILTER BACKTEST (15% CGT)")
    print("  Derived from Feature Importance Research")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n  Generating target weights (CDI regime + MA200 filter)...")
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol, cdi_daily)

    # Add CDI and IBOV returns to the returns matrix
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
        name="CDI+MA200",
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

    m_pretax = build_metrics(pretax_ret, "CDI+MA200 Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret, "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    # Regime summary
    tw_sum = target_weights.drop(columns=["CDI_ASSET"], errors="ignore").sum(axis=1)
    cdi_months = (tw_sum == 0).sum()
    equity_months = (tw_sum > 0).sum()
    total_months = cdi_months + equity_months
    print(f"\n  Regime Summary:")
    print(
        f"    Equity months: {equity_months} ({equity_months / max(total_months, 1) * 100:.0f}%)"
    )
    print(
        f"    CDI months:    {cdi_months} ({cdi_months / max(total_months, 1) * 100:.0f}%)"
    )
    avg_stocks = (
        target_weights.drop(columns=["CDI_ASSET"], errors="ignore").gt(0).sum(axis=1)
    )
    avg_stocks_when_active = avg_stocks[avg_stocks > 0].mean()
    print(f"    Avg stocks when in equity: {avg_stocks_when_active:.0f}")

    plot_tax_backtest(
        title=(
            f"CDI Regime + MA200 Trend Filter (B3 Native)\n"
            f"Easing → Equal-Weight Above MA200  |  Tightening → 100% CDI\n"
            f"R${MIN_ADTV / 1_000_000:.0f}M+ ADTV  ·  15% CGT + {SLIPPAGE * 100:.1f}% Slippage  ·  "
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
        out_path="cdi_ma200_trend_backtest.png",
        cdi_ret=cdi_ret,
    )


if __name__ == "__main__":
    main()

"""
B3 High Dividend Yield + Momentum Backtest
=======================================================================
Strategy: The classic "Dogs of the Dow" with a Momentum filter.
In Brazil, high dividend yield is incredibly important due to high interest rates,
but buying a high-yielding stock that is trending downwards is a "Value Trap"
(e.g., the dividend is high because the stock price crashed).

This strategy dynamically calculates the TRUE Trailing 12-Month Dividend Yield
directly from the B3 corporate action ledger. It filters for stocks yielding
> 8%, and then buys the top 10% that have the highest 12-Month Momentum.
"""

import warnings

warnings.filterwarnings("ignore")

import sqlite3
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
MOM_PERIODS = int(0.5 * PERIODS_PER_YEAR)  # 6 month momentum
SKIP_PERIODS = 1 if REBALANCE_FREQ == "ME" else 0

MIN_YIELD = 0.08  # Must yield at least 8% over the last 12 months
TOP_DECILE = 0.20  # Buy the top 20% highest momentum from the high yielders
TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 2_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def load_split_adj_prices(db_path: str, start: str, end: str):
    """We need split_adj_close to natively calculate dividend yield."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT date, ticker, split_adj_close FROM prices WHERE date >= '{start}' AND date <= '{end}' AND ((LENGTH(ticker) = 5 AND SUBSTR(ticker, 5, 1) IN ('3', '4', '5', '6')) OR (LENGTH(ticker) = 6 AND SUBSTR(ticker, 5, 2) = '11'))"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    split_adj = df.pivot(
        index="date", columns="ticker", values="split_adj_close"
    ).ffill()
    return split_adj


def generate_signals(adj_close, close_px, fin_vol, split_adj):
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()

    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    split_px = split_adj.resample(REBALANCE_FREQ).last()

    # ── Calculate Trailing 12-Month Dividend Yield ──
    # adj_close is fully adjusted (splits + divs backward).
    # split_adj_close is only adjusted for splits.
    # The ratio of split_adj / adj_close represents cumulative dividends mathematically baked in.
    # We can extract the exactly 12-month trailing yield natively!
    ratio = split_px / px
    trailing_yield = (ratio.shift(LOOKBACK_PERIODS) / ratio) - 1.0

    # ── Signal 2: 6-Month Momentum ──
    log_ret = np.log1p(ret)
    mom_signal = log_ret.shift(SKIP_PERIODS).rolling(MOM_PERIODS).sum()
    mom_glitch = (
        ((ret > 1.0) | (ret < -0.90)).shift(SKIP_PERIODS).rolling(MOM_PERIODS).max()
    )
    mom_signal[mom_glitch == 1] = np.nan

    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    start_idx = LOOKBACK_PERIODS + SKIP_PERIODS + 1

    prev_sel = set()

    for i in range(start_idx, len(ret)):
        yield_row = trailing_yield.iloc[i - 1]
        mom_row = mom_signal.iloc[i - 1]

        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]

        # Must have R$ 2M ADTV, not be a penny stock, AND have at least defined yield
        valid_mask = (
            (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0) & (yield_row >= MIN_YIELD)
        )

        valid = mom_row[valid_mask].dropna()

        if len(valid) < 5:
            sel = prev_sel
        else:
            # Out of the high yielders, pick the top 20% with the strongest upward trend
            n_sel = max(1, int(len(valid) * TOP_DECILE))
            sel = set(valid.nlargest(n_sel).index)

        if not sel:
            continue

        weight_per_stock = 1.0 / len(sel)
        for t in sel:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight_per_stock

        prev_sel = sel

    return ret, target_weights


def main():
    print("\n" + "=" * 70)
    print("  B3 HIGH YIELD + MOMENTUM (DIVIDEND TRAP AVOIDER)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n🧠 Generating target weights matrix...")
    split_adj = load_split_adj_prices(DB_PATH, START_DATE, END_DATE)
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol, split_adj)
    ret = ret.fillna(0.0)

    print(f"\n🚀 Running generic simulation engine ({REBALANCE_FREQ})...")
    result = run_simulation(
        ret, target_weights, INITIAL_CAPITAL, TAX_RATE, SLIPPAGE, name="Yield+Mom"
    )

    common = result["pretax_values"].index.intersection(ibov_ret.index)
    m_pretax = build_metrics(
        value_to_ret(result["pretax_values"].loc[common]),
        "Yield+Mom Pre-Tax",
        PERIODS_PER_YEAR,
    )
    m_aftertax = build_metrics(
        value_to_ret(result["aftertax_values"].loc[common]),
        "After-Tax 15% CGT",
        PERIODS_PER_YEAR,
    )
    m_ibov = build_metrics(ibov_ret.loc[common], "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret.loc[common], "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    plot_tax_backtest(
        title=f"Yield + Momentum  ·  Min Yield > 8% · Top {int(TOP_DECILE * 100)}% Momentum\nR$ {MIN_ADTV / 1_000_000:.0f}M+ ADTV  ·  15% CGT + {SLIPPAGE * 100}% Slippage\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=result["pretax_values"].loc[common],
        aftertax_val=result["aftertax_values"].loc[common],
        ibov_ret=ibov_ret.loc[common],
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=result["tax_paid"].sum(),
        out_path="yield_momentum_backtest.png",
        cdi_ret=cdi_ret.loc[common],
    )


if __name__ == "__main__":
    main()

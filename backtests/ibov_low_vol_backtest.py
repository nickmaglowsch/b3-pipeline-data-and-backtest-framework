"""
Ibovespa Smart Low Volatility B3 replica — Backtest (15% CGT)
=============================================================
Faithful to B3's methodology: annualized EWMA(252) daily-return vol, bottom 33%
of the liquid universe, inverse-vol weighting capped at 10% per name.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
import os
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.simulation import run_simulation
from strategies.ibov_low_vol import ewma_annualized_vol, inverse_vol_capped

REBALANCE_FREQ = "ME"
PERIODS_PER_YEAR = 12
EWMA_N = 252
BOTTOM_PCT = 0.33
WEIGHT_CAP = 0.10
TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 5_000_000
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b3_market_data.sqlite")
IBOV_INDEX = "^BVSP"


def generate_signals(adj_close, close_px, fin_vol):
    ret = adj_close.resample(REBALANCE_FREQ).last().pct_change()
    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    vol_daily = ewma_annualized_vol(adj_close, EWMA_N)
    vol = vol_daily.resample(REBALANCE_FREQ).last().reindex(ret.index).reindex(columns=ret.columns)
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(12).max()
    vol[has_glitch == 1] = np.nan

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    for i in range(2, len(ret)):
        vol_row = vol.iloc[i - 1]
        mask = (adtv.iloc[i - 1] >= MIN_ADTV) & (raw_close.iloc[i - 1] >= 1.0)
        valid = vol_row[mask].dropna()
        valid = valid[valid > 0]
        if len(valid) < 5:
            continue
        k = max(5, int(len(valid) * BOTTOM_PCT))
        w = inverse_vol_capped(valid.nsmallest(k), WEIGHT_CAP)
        for t, wt in w.items():
            tw.iloc[i, tw.columns.get_loc(t)] = wt
    return ret, tw


def main():
    print("\n" + "=" * 70)
    print("  IBOVESPA SMART LOW VOLATILITY B3 — REPLICA BACKTEST (15% CGT)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n🧠 Generating target weights (EWMA-252 inverse-vol, cap 10%)...")
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol)
    ret = ret.fillna(0.0)

    print(f"\n🚀 Running generic simulation engine ({REBALANCE_FREQ})...")
    result = run_simulation(
        returns_matrix=ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="Ibov Low Vol B3",
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

    m_pretax = build_metrics(pretax_ret, "Ibov Low Vol Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret, "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    plot_tax_backtest(
        title=(
            "Ibovespa Smart Low Volatility B3 (Replica)\n"
            f"EWMA-{EWMA_N} vol · bottom {int(BOTTOM_PCT*100)}% · inverse-vol cap "
            f"{int(WEIGHT_CAP*100)}% · R${MIN_ADTV/1_000_000:.0f}M+ ADTV · 15% CGT\n"
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
        out_path="ibov_low_vol_backtest.png",
        cdi_ret=cdi_ret,
    )


if __name__ == "__main__":
    main()

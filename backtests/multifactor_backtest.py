"""
B3 Multi-Factor Strategy Backtest (Value + Momentum)
=======================================================================
Strategy: Instead of allocating parallel capital to separate strategies,
this strategy uses a composite scoring system. Each month it ranks stocks
by both their Momentum (12-month return) and Mean Reversion (1-month drop).
It combines the ranks and buys the top 10%.
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_b3_data, download_benchmark
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.portfolio import rebalance_positions, apply_returns, compute_tax

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TOP_DECILE = 0.10
TAX_RATE = 0.15
START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def run_backtest(adj_close, close_px, fin_vol):
    monthly_px = adj_close.resample("ME").last()
    monthly_ret = monthly_px.pct_change()

    monthly_raw_close = close_px.resample("ME").last()
    monthly_adtv = fin_vol.resample("ME").mean()

    # Signal 1: 12-Month Momentum (skip 1 month)
    log_ret = np.log1p(monthly_ret)
    mom_signal = log_ret.shift(1).rolling(12).sum()
    mom_glitch = (
        ((monthly_ret > 1.0) | (monthly_ret < -0.90)).shift(1).rolling(12).max()
    )
    mom_signal[mom_glitch == 1] = np.nan

    # Signal 2: Low Volatility (12-month trailing standard deviation)
    vol_signal = -monthly_ret.shift(1).rolling(12).std()
    vol_glitch = (
        ((monthly_ret > 1.0) | (monthly_ret < -0.90)).shift(1).rolling(12).max()
    )
    vol_signal[vol_glitch == 1] = np.nan

    # Convert signals to cross-sectional percentile ranks (higher = better)
    mom_rank = mom_signal.rank(axis=1, pct=True)
    vol_rank = vol_signal.rank(axis=1, pct=True)

    # Combined Composite Signal (50% Mom, 50% Low Vol)
    # We only score stocks where both signals are valid
    composite_signal = (mom_rank * 0.5) + (vol_rank * 0.5)

    start_idx = 14

    pretax_positions, aftertax_positions = {}, {}
    loss_carryforward = 0.0

    pretax_values, aftertax_values = [], []
    tax_paid_list, loss_cf_list, turnover_list, dates = [], [], [], []
    prev_selected = set()
    initialized = False

    for i in range(start_idx, len(monthly_ret)):
        date = monthly_ret.index[i]

        sig_row = composite_signal.iloc[i - 1]

        adtv_row = monthly_adtv.iloc[i - 1]
        raw_close_row = monthly_raw_close.iloc[i - 1]
        valid_universe_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)

        valid = sig_row[valid_universe_mask].dropna()

        if len(valid) < 5:
            new_selected = prev_selected
        else:
            n_select = max(1, int(len(valid) * TOP_DECILE))
            new_selected = set(valid.nlargest(n_select).index.tolist())

        ret_row = monthly_ret.iloc[i]

        if not initialized and len(valid) >= 5:
            alloc = INITIAL_CAPITAL / len(new_selected)
            for t in new_selected:
                pretax_positions[t] = {"cost_basis": alloc, "current_value": alloc}
                aftertax_positions[t] = {"cost_basis": alloc, "current_value": alloc}
            prev_selected = new_selected
            initialized = True

            pretax_values.append(INITIAL_CAPITAL)
            aftertax_values.append(INITIAL_CAPITAL)
            tax_paid_list.append(0.0)
            loss_cf_list.append(0.0)
            turnover_list.append(1.0)
            dates.append(date)
            continue

        if not initialized:
            continue

        apply_returns(pretax_positions, ret_row)
        apply_returns(aftertax_positions, ret_row)

        exiting = prev_selected - new_selected
        entering = new_selected - prev_selected
        n_universe = len(new_selected | prev_selected)
        turnover_list.append(len(exiting | entering) / max(n_universe, 1))

        tax, loss_carryforward = compute_tax(
            exiting, aftertax_positions, loss_carryforward, TAX_RATE
        )

        rebalance_positions(pretax_positions, exiting, entering, ret_row, tax=0.0)
        rebalance_positions(aftertax_positions, exiting, entering, ret_row, tax=tax)

        pretax_values.append(sum(p["current_value"] for p in pretax_positions.values()))
        aftertax_values.append(
            sum(p["current_value"] for p in aftertax_positions.values())
        )
        tax_paid_list.append(tax)
        loss_cf_list.append(loss_carryforward)
        dates.append(date)

        prev_selected = new_selected

    idx = pd.DatetimeIndex(dates)
    return {
        "pretax_values": pd.Series(
            pretax_values, index=idx, name="Multi-Factor (Pre-Tax)"
        ),
        "aftertax_values": pd.Series(
            aftertax_values, index=idx, name="Multi-Factor (After-Tax)"
        ),
        "tax_paid": pd.Series(tax_paid_list, index=idx, name="Tax Paid (BRL)"),
        "loss_carryforward": pd.Series(
            loss_cf_list, index=idx, name="Loss Carryforward (BRL)"
        ),
        "turnover": pd.Series(turnover_list, index=idx, name="Turnover"),
    }


def main():
    print("\n" + "=" * 70)
    print("  B3 NATIVE MULTI-FACTOR BACKTEST (15% CGT)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    ibov_monthly = ibov_px.resample("ME").last().pct_change().dropna()
    ibov_monthly.name = "IBOV"

    print("\n🚀 Running dynamic backtest with tax engine...")
    result = run_backtest(adj_close, close_px, fin_vol)

    common = result["pretax_values"].index.intersection(ibov_monthly.index)
    pretax_val = result["pretax_values"].loc[common]
    aftertax_val = result["aftertax_values"].loc[common]
    ibov_ret = ibov_monthly.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = result["tax_paid"].sum()

    m_pretax = build_metrics(pretax_ret, "Factor Pre-Tax")
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT")
    m_ibov = build_metrics(ibov_ret, "IBOV")

    display_metrics_table([m_pretax, m_aftertax, m_ibov])

    plot_tax_backtest(
        title=f"Multi-Factor Rank: 50% Mom + 50% Low Vol\nTop 10% Selected  ·  R$ {MIN_ADTV / 1_000_000:.0f}M+ ADTV\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov],
        total_tax_brl=total_tax,
        out_path="multifactor_backtest.png",
    )


if __name__ == "__main__":
    main()

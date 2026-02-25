"""
B3 Short-Term Mean Reversion Strategy Backtest
=======================================================================
Strategy: Each month, rank all eligible stocks by their return over the past 1 month.
Buy the K% of stocks with the LOWEST returns (the biggest losers).
Equal-weight, rebalance monthly.

Universe:
  • Sourced natively from local B3 SQLite database (b3_market_data.sqlite).
  • Restricted to standard stocks/units ending in 3, 4, 5, 6, 11.
  • Dynamically filtered each month to only include stocks with an Average
    Daily Traded Volume (ADTV) >= R$ 1,000,000 in the preceding month.
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add parent directory to path so we can import from core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_b3_data, download_benchmark
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.portfolio import rebalance_positions, apply_returns, compute_tax

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
TOP_DECILE = 0.10  # fraction of valid universe selected each rebalance
TAX_RATE = 0.15  # Brazilian capital gains tax rate
START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000  # BRL
MIN_ADTV = 1_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


# ─────────────────────────────────────────────
#  SIGNAL
# ─────────────────────────────────────────────
def mean_reversion_signal(monthly_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate 1-month mean reversion signal.
    We return the NEGATIVE 1-month return so that .nlargest() picks the
    stocks that dropped the most (the biggest losers).
    """
    signal = -monthly_ret.shift(1)

    # Detect data glitches (monthly return > 100% or < -90%)
    has_glitch = ((monthly_ret > 1.0) | (monthly_ret < -0.90)).shift(1)

    # Invalidate signal if there's a glitch in the lookback window
    signal[has_glitch == True] = np.nan

    return signal


# ─────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────
def run_backtest(
    adj_close: pd.DataFrame,
    close_px: pd.DataFrame,
    fin_vol: pd.DataFrame,
    top_pct: float = TOP_DECILE,
) -> dict:
    monthly_px = adj_close.resample("ME").last()
    monthly_ret = monthly_px.pct_change()

    monthly_raw_close = close_px.resample("ME").last()
    monthly_adtv = fin_vol.resample("ME").mean()

    signal = mean_reversion_signal(monthly_ret)

    # Need at least 2 months of data to start (1 for return, 1 for shift)
    start_idx = 2

    pretax_positions = {}
    aftertax_positions = {}
    loss_carryforward = 0.0

    pretax_values = []
    aftertax_values = []
    tax_paid_list = []
    loss_cf_list = []
    turnover_list = []
    dates = []
    prev_selected = set()
    initialized = False

    for i in range(start_idx, len(monthly_ret)):
        date = monthly_ret.index[i]

        sig_row = signal.iloc[i - 1]

        # Liquidity & Penny Stock Filter
        adtv_row = monthly_adtv.iloc[i - 1]
        raw_close_row = monthly_raw_close.iloc[i - 1]
        valid_universe_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)

        valid = sig_row[valid_universe_mask].dropna()

        if len(valid) < 5:
            new_selected = prev_selected
        else:
            n_select = max(1, int(len(valid) * top_pct))
            new_selected = set(valid.nlargest(n_select).index.tolist())

        ret_row = monthly_ret.iloc[i]

        # ── First month: open equal-weight positions ───
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
        "pretax_values": pd.Series(pretax_values, index=idx, name="Reversal (Pre-Tax)"),
        "aftertax_values": pd.Series(
            aftertax_values, index=idx, name="Reversal (After-Tax)"
        ),
        "tax_paid": pd.Series(tax_paid_list, index=idx, name="Tax Paid (BRL)"),
        "loss_carryforward": pd.Series(
            loss_cf_list, index=idx, name="Loss Carryforward (BRL)"
        ),
        "turnover": pd.Series(turnover_list, index=idx, name="Turnover"),
    }


def main():
    print("\n" + "=" * 70)
    print("  B3 NATIVE MEAN REVERSION BACKTEST (15% CGT)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    ibov_monthly = ibov_px.resample("ME").last().pct_change().dropna()
    ibov_monthly.name = "IBOV"

    print("\n🚀 Running dynamic backtest with tax engine...")
    result = run_backtest(adj_close, close_px, fin_vol)

    pretax_val = result["pretax_values"]
    aftertax_val = result["aftertax_values"]
    tax_paid = result["tax_paid"]
    loss_cf = result["loss_carryforward"]
    turnover = result["turnover"]

    common = pretax_val.index.intersection(ibov_monthly.index)
    pretax_val = pretax_val.loc[common]
    aftertax_val = aftertax_val.loc[common]
    tax_paid = tax_paid.loc[common]
    loss_cf = loss_cf.loc[common]
    turnover = turnover.loc[common]
    ibov_ret = ibov_monthly.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = tax_paid.sum()

    print(f"\n   Period          : {common[0].date()} → {common[-1].date()}")
    print(f"   Total months    : {len(common)}")
    print(f"   Total tax paid  : R$ {total_tax:,.2f}")
    print(f"   Final loss C/F  : R$ {loss_cf.iloc[-1]:,.2f}")
    print(f"   Avg turnover/mo : {turnover.mean() * 100:.1f}%")

    m_pretax = build_metrics(pretax_ret, "Reversal Pre-Tax")
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT")
    m_ibov = build_metrics(ibov_ret, "IBOV")

    display_metrics_table([m_pretax, m_aftertax, m_ibov])

    title = (
        f"Dynamic Mean Reversion (B3 Native)  ·  1M Lookback\n"
        f"Top {int(TOP_DECILE * 100)}% of Losers  ·  R$ {MIN_ADTV / 1_000_000:.0f}M+ ADTV  ·  15% CGT\n"
        f"{START_DATE[:4]}–{END_DATE[:4]}"
    )

    plot_tax_backtest(
        title=title,
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=tax_paid,
        loss_cf=loss_cf,
        turnover=turnover,
        metrics=[m_pretax, m_aftertax, m_ibov],
        total_tax_brl=total_tax,
        out_path="mean_reversion_backtest.png",
    )


if __name__ == "__main__":
    main()

"""
B3 Dynamic Momentum Strategy Backtest — with Brazilian Capital Gains Tax
=======================================================================
"""

import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

# Import framework components
from core.data import load_b3_data, download_benchmark
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.portfolio import rebalance_positions, apply_returns, compute_tax

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
LOOKBACK_MONTHS = 12  # momentum signal window (months)
SKIP_MONTHS = 1  # skip most-recent month (avoid short-term reversal)
TOP_DECILE = 0.10  # fraction of valid universe selected each rebalance
TAX_RATE = 0.15  # Brazilian capital gains tax rate
START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000  # BRL
MIN_ADTV = 1_000_000  # Minimum Average Daily Traded Volume in BRL

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


# ─────────────────────────────────────────────
#  SIGNAL
# ─────────────────────────────────────────────
def momentum_signal(
    monthly_ret: pd.DataFrame, lookback: int, skip: int
) -> pd.DataFrame:
    """Log-cumulative return with data glitch protection."""
    log_ret = np.log1p(monthly_ret)
    signal = log_ret.shift(skip).rolling(lookback).sum()

    has_glitch = (
        ((monthly_ret > 1.0) | (monthly_ret < -0.90))
        .shift(skip)
        .rolling(lookback)
        .max()
    )
    signal[has_glitch == 1] = np.nan
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
    """Simulate two portfolios in parallel using dynamic universe filtering."""
    monthly_px = adj_close.resample("ME").last()
    monthly_ret = monthly_px.pct_change()
    monthly_raw_close = close_px.resample("ME").last()
    monthly_adtv = fin_vol.resample("ME").mean()

    signal = momentum_signal(monthly_ret, LOOKBACK_MONTHS, SKIP_MONTHS)
    start_idx = LOOKBACK_MONTHS + SKIP_MONTHS + 1

    pretax_positions, aftertax_positions = {}, {}
    loss_carryforward = 0.0

    pretax_values, aftertax_values = [], []
    tax_paid_list, loss_cf_list, turnover_list, dates = [], [], [], []

    prev_selected = set()
    initialized = False

    for i in range(start_idx, len(monthly_ret)):
        date = monthly_ret.index[i]

        sig_row = signal.iloc[i - 1]
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

        # Step 1: Apply returns
        apply_returns(pretax_positions, ret_row)
        apply_returns(aftertax_positions, ret_row)

        # Step 2: Classify positions
        exiting = prev_selected - new_selected
        entering = new_selected - prev_selected
        n_universe = len(new_selected | prev_selected)
        turnover_list.append(len(exiting | entering) / max(n_universe, 1))

        # Step 3: Compute tax
        tax, loss_carryforward = compute_tax(
            exiting, aftertax_positions, loss_carryforward, TAX_RATE
        )

        # Step 4: Rebalance ledgers
        rebalance_positions(pretax_positions, exiting, entering, ret_row, tax=0.0)
        rebalance_positions(aftertax_positions, exiting, entering, ret_row, tax=tax)

        # Step 5: Record state
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
        "pretax_values": pd.Series(pretax_values, index=idx, name="Momentum (Pre-Tax)"),
        "aftertax_values": pd.Series(
            aftertax_values, index=idx, name="Momentum (After-Tax)"
        ),
        "tax_paid": pd.Series(tax_paid_list, index=idx, name="Tax Paid (BRL)"),
        "loss_carryforward": pd.Series(
            loss_cf_list, index=idx, name="Loss Carryforward (BRL)"
        ),
        "turnover": pd.Series(turnover_list, index=idx, name="Turnover"),
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  B3 NATIVE DYNAMIC MOMENTUM BACKTEST (15% CGT)")
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
    total_tax = result["tax_paid"].loc[common].sum()

    print(f"\n   Period          : {common[0].date()} → {common[-1].date()}")
    print(f"   Total months    : {len(common)}")
    print(f"   Total tax paid  : R$ {total_tax:,.2f}")
    print(
        f"   Final loss C/F  : R$ {result['loss_carryforward'].loc[common].iloc[-1]:,.2f}"
    )
    print(f"   Avg turnover/mo : {result['turnover'].loc[common].mean() * 100:.1f}%")

    m_pretax = build_metrics(pretax_ret, "Momentum Pre-Tax")
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT")
    m_ibov = build_metrics(ibov_ret, "IBOV")

    display_metrics_table([m_pretax, m_aftertax, m_ibov])

    plot_tax_backtest(
        title=f"Dynamic Momentum (B3 Native)  ·  {LOOKBACK_MONTHS}M Lookback · Skip {SKIP_MONTHS}M\nTop {int(TOP_DECILE * 100)}% of R$ {MIN_ADTV / 1_000_000:.0f}M+ ADTV  ·  15% CGT with Loss Carryforward\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov],
        total_tax_brl=total_tax,
        out_path="momentum_dynamic_backtest.png",
    )


if __name__ == "__main__":
    main()

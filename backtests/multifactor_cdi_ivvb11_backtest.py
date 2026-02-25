"""
B3 33/33/33 Multi-Factor + CDI + IVVB11 Strategy Backtest
=======================================================================
Strategy: Runs a 33/33/33 portfolio split equally between:
  1. Multi-Factor (50% Momentum + 50% Low Volatility)
  2. CDI (Risk-free rate)
  3. IVVB11 (S&P 500 ETF in BRL)

The portfolio is rebalanced to equal weights every month.
Returns are subject to a unified 15% CGT + loss carryforward pool.
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
from core.portfolio import apply_returns, compute_tax

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
# Target Portfolio Weights (Must sum to 1.0)
WEIGHT_EQUITY = 0.9
WEIGHT_CDI = 0.1
WEIGHT_IVVB11 = 0.0

BORROW_SPREAD = 0.02  # 2% annualized spread on margin loans

REBALANCE_FREQ = "W-FRI"  # ME=Monthly, QE=Quarterly, YE=Yearly, W-FRI=Weekly on Fridays

# We define the strategy lookback in Calendar Years.
LOOKBACK_YEARS = 1

# Automatically calculate how many periods this lookback represents
period_map = {"ME": 12, "QE": 4, "YE": 1, "W": 52, "W-FRI": 52, "W-MON": 52}
PERIODS_PER_YEAR = period_map.get(REBALANCE_FREQ, 12)
LOOKBACK_PERIODS = int(LOOKBACK_YEARS * PERIODS_PER_YEAR)

TOP_DECILE = 1.00
TAX_RATE = 0.15
START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"
IVVB11_TICKER = "IVVB11.SA"  # Fetch from Yahoo since it represents external asset


def run_backtest(adj_close, close_px, fin_vol, cdi_ret, ivvb_ret):
    # Resample all core series to exactly match the configured Rebalance Frequency
    monthly_px = adj_close.resample(REBALANCE_FREQ).last()
    monthly_ret = monthly_px.pct_change()

    monthly_raw_close = close_px.resample(REBALANCE_FREQ).last()
    monthly_adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    # ── Signal Generation (Multi-Factor) ──
    log_ret = np.log1p(monthly_ret)
    mom_signal = log_ret.shift(1).rolling(LOOKBACK_PERIODS).sum()
    mom_glitch = (
        ((monthly_ret > 1.0) | (monthly_ret < -0.90))
        .shift(1)
        .rolling(LOOKBACK_PERIODS)
        .max()
    )
    mom_signal[mom_glitch == 1] = np.nan

    vol_signal = -monthly_ret.shift(1).rolling(LOOKBACK_PERIODS).std()
    vol_glitch = (
        ((monthly_ret > 1.0) | (monthly_ret < -0.90))
        .shift(1)
        .rolling(LOOKBACK_PERIODS)
        .max()
    )
    vol_signal[vol_glitch == 1] = np.nan

    mom_rank = mom_signal.rank(axis=1, pct=True)
    vol_rank = vol_signal.rank(axis=1, pct=True)

    composite_signal = (mom_rank * 0.5) + (vol_rank * 0.5)

    start_idx = LOOKBACK_PERIODS + 2

    pt_pos = {}
    at_pos = {}

    loss_carryforward = 0.0

    pretax_values, aftertax_values = [], []
    tax_paid_list, loss_cf_list, turnover_list, dates = [], [], [], []

    prev_sel_eq = set()
    initialized = False

    for i in range(start_idx, len(monthly_ret)):
        date = monthly_ret.index[i]

        # 1. Selection: Multi-Factor Equities
        sig_row = composite_signal.iloc[i - 1]
        adtv_row = monthly_adtv.iloc[i - 1]
        raw_close_row = monthly_raw_close.iloc[i - 1]
        valid_universe_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)

        valid_eq = sig_row[valid_universe_mask].dropna()
        if len(valid_eq) >= 5:
            n_sel_eq = max(1, int(len(valid_eq) * TOP_DECILE))
            new_sel_eq = set(valid_eq.nlargest(n_sel_eq).index.tolist())
        else:
            new_sel_eq = prev_sel_eq

        ret_row = monthly_ret.iloc[i].copy()

        # Add CDI and IVVB11 returns for this month
        current_cdi = cdi_ret.get(date, 0.0)
        if pd.isna(current_cdi):
            current_cdi = 0.0

        # If CDI is used as a margin loan, add the borrowing spread
        if WEIGHT_CDI < 0:
            current_cdi += BORROW_SPREAD / 12.0

        ret_row["CDI_ASSET"] = current_cdi

        current_ivvb = ivvb_ret.get(date, 0.0)
        if pd.isna(current_ivvb):
            current_ivvb = 0.0
        ret_row["IVVB11_ASSET"] = current_ivvb

        # ── First month initialization ───
        if not initialized and len(valid_eq) >= 5:
            eq_sleeve_capital = INITIAL_CAPITAL * WEIGHT_EQUITY
            cdi_sleeve_capital = INITIAL_CAPITAL * WEIGHT_CDI
            ivvb_sleeve_capital = INITIAL_CAPITAL * WEIGHT_IVVB11

            eq_alloc = eq_sleeve_capital / len(new_sel_eq)

            for t in new_sel_eq:
                pt_pos[t] = {"cost_basis": eq_alloc, "current_value": eq_alloc}
                at_pos[t] = {"cost_basis": eq_alloc, "current_value": eq_alloc}

            pt_pos["CDI_ASSET"] = {
                "cost_basis": cdi_sleeve_capital,
                "current_value": cdi_sleeve_capital,
            }
            at_pos["CDI_ASSET"] = {
                "cost_basis": cdi_sleeve_capital,
                "current_value": cdi_sleeve_capital,
            }

            pt_pos["IVVB11_ASSET"] = {
                "cost_basis": ivvb_sleeve_capital,
                "current_value": ivvb_sleeve_capital,
            }
            at_pos["IVVB11_ASSET"] = {
                "cost_basis": ivvb_sleeve_capital,
                "current_value": ivvb_sleeve_capital,
            }

            prev_sel_eq = new_sel_eq
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

        # ── Step 1: Apply Returns ────
        apply_returns(pt_pos, ret_row)
        apply_returns(at_pos, ret_row)

        pt_total = sum(p["current_value"] for p in pt_pos.values())
        at_total = sum(p["current_value"] for p in at_pos.values())

        if pt_total <= 0 or at_total <= 0:
            print(f"\n💀 MARGIN CALL! Portfolio went bankrupt on {date.date()}")
            break

        exit_eq = prev_sel_eq - new_sel_eq
        enter_eq = new_sel_eq - prev_sel_eq
        turnover_list.append(
            len(exit_eq | enter_eq) / max(len(prev_sel_eq | new_sel_eq), 1)
        )

        # ── Step 3: Compute unified tax for after-tax ledgers ───────────
        tax, loss_carryforward = compute_tax(
            set(at_pos.keys()), at_pos, loss_carryforward, TAX_RATE
        )

        # Liquidate
        pt_cash = pt_total
        at_cash = at_total - tax

        pt_pos.clear()
        at_pos.clear()

        # ── Step 4: Rebalance according to config weights ─────────────────
        pt_eq_sleeve = pt_cash * WEIGHT_EQUITY
        pt_cdi_sleeve = pt_cash * WEIGHT_CDI
        pt_ivvb_sleeve = pt_cash * WEIGHT_IVVB11

        at_eq_sleeve = at_cash * WEIGHT_EQUITY
        at_cdi_sleeve = at_cash * WEIGHT_CDI
        at_ivvb_sleeve = at_cash * WEIGHT_IVVB11

        pt_eq_alloc = pt_eq_sleeve / len(new_sel_eq)
        at_eq_alloc = at_eq_sleeve / len(new_sel_eq)

        for t in new_sel_eq:
            pt_pos[t] = {"cost_basis": pt_eq_alloc, "current_value": pt_eq_alloc}
            at_pos[t] = {"cost_basis": at_eq_alloc, "current_value": at_eq_alloc}

        pt_pos["CDI_ASSET"] = {
            "cost_basis": pt_cdi_sleeve,
            "current_value": pt_cdi_sleeve,
        }
        at_pos["CDI_ASSET"] = {
            "cost_basis": at_cdi_sleeve,
            "current_value": at_cdi_sleeve,
        }

        pt_pos["IVVB11_ASSET"] = {
            "cost_basis": pt_ivvb_sleeve,
            "current_value": pt_ivvb_sleeve,
        }
        at_pos["IVVB11_ASSET"] = {
            "cost_basis": at_ivvb_sleeve,
            "current_value": at_ivvb_sleeve,
        }

        # ── Step 5: Record state ───────────────────────────────────────
        pretax_values.append(pt_total)
        aftertax_values.append(at_cash)
        tax_paid_list.append(tax)
        loss_cf_list.append(loss_carryforward)
        dates.append(date)

        prev_sel_eq = new_sel_eq

    idx = pd.DatetimeIndex(dates)
    return {
        "pretax_values": pd.Series(
            pretax_values, index=idx, name="33/33/33 Mix (Pre-Tax)"
        ),
        "aftertax_values": pd.Series(
            aftertax_values, index=idx, name="33/33/33 Mix (After-Tax)"
        ),
        "tax_paid": pd.Series(tax_paid_list, index=idx, name="Tax Paid (BRL)"),
        "loss_carryforward": pd.Series(
            loss_cf_list, index=idx, name="Loss Carryforward (BRL)"
        ),
        "turnover": pd.Series(turnover_list, index=idx, name="Turnover"),
    }


def main():
    print("\n" + "=" * 70)
    print("  B3 NATIVE MULTI-FACTOR + CDI + IVVB11 BACKTEST (15% CGT)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    # Download raw benchmarks
    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ivvb_px = download_benchmark(IVVB11_TICKER, START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    # Resample benchmarks to match the configured rebalance frequency
    # For price series, we just take the last price of the period and pct_change
    ivvb_ret = ivvb_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"

    # For CDI daily returns, we compound them correctly over the requested frequency
    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n🚀 Running dynamic backtest with tax engine...")
    result = run_backtest(adj_close, close_px, fin_vol, cdi_ret, ivvb_ret)

    common = result["pretax_values"].index.intersection(ibov_ret.index)
    pretax_val = result["pretax_values"].loc[common]
    aftertax_val = result["aftertax_values"].loc[common]
    ibov_ret = ibov_ret.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = result["tax_paid"].sum()

    m_pretax = build_metrics(
        pretax_ret,
        f"{int(WEIGHT_EQUITY * 100)}/{int(WEIGHT_CDI * 100)}/{int(WEIGHT_IVVB11 * 100)} Pre-Tax",
    )
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT")
    m_ibov = build_metrics(ibov_ret, "IBOV")
    m_cdi = build_metrics(cdi_ret.loc[common], "CDI")

    # Calculate how many periods are in a year to adjust the multiplier
    # for metric annualization (so Vol/Sharpe are perfectly accurate).
    period_map = {"ME": 12, "QE": 4, "YE": 1, "W": 52, "W-FRI": 52, "W-MON": 52}
    periods_per_year = period_map.get(REBALANCE_FREQ, 12)

    m_pretax = build_metrics(
        pretax_ret,
        f"{int(WEIGHT_EQUITY * 100)}/{int(WEIGHT_CDI * 100)}/{int(WEIGHT_IVVB11 * 100)} Pre-Tax",
        periods_per_year,
    )
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", periods_per_year)
    m_ibov = build_metrics(ibov_ret, "IBOV", periods_per_year)
    m_cdi = build_metrics(cdi_ret.loc[common], "CDI", periods_per_year)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    plot_tax_backtest(
        title=f"{int(WEIGHT_EQUITY * 100)}% Multi-Factor / {int(WEIGHT_CDI * 100)}% CDI / {int(WEIGHT_IVVB11 * 100)}% IVVB11\n{REBALANCE_FREQ} Rebalance  ·  15% CGT with Shared Loss Carryforward\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=total_tax,
        out_path="multifactor_cdi_ivvb11_backtest.png",
        cdi_ret=cdi_ret.loc[common],
    )


if __name__ == "__main__":
    main()

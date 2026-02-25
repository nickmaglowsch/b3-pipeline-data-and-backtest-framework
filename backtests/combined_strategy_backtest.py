"""
B3 Combined Strategy Backtest — 50% Momentum / 50% Low Volatility
=======================================================================
Strategy: Runs multiple quantitative sub-strategies (sleeves) in a single
taxable portfolio. Shares a single Capital Gains Tax (CGT) loss carryforward
pool, which accurately reflects a real Brazilian investor's CPF account.

Universe:
  • Sourced natively from local B3 SQLite database (b3_market_data.sqlite).
  • Restricted to standard stocks/units ending in 3, 4, 5, 6, 11.
  • ADTV >= R$ 5,000,000.
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

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
TOP_DECILE = 0.10
TAX_RATE = 0.15
START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000  # R$ 50k to Mom, R$ 50k to Low Vol
MIN_ADTV = 5_000_000

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


# ─────────────────────────────────────────────
#  SIGNALS
# ─────────────────────────────────────────────
def momentum_signal(
    monthly_ret: pd.DataFrame, lookback: int, skip: int
) -> pd.DataFrame:
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


def low_vol_signal(monthly_ret: pd.DataFrame, lookback: int) -> pd.DataFrame:
    volatility = monthly_ret.rolling(lookback).std()
    has_glitch = ((monthly_ret > 1.0) | (monthly_ret < -0.90)).rolling(lookback).max()
    volatility[has_glitch == 1] = np.nan
    return -volatility  # Negative so nlargest() picks lowest vol


# ─────────────────────────────────────────────
#  MULTI-STRATEGY REBALANCING ENGINE
# ─────────────────────────────────────────────
def _rebalance_sleeve(
    positions: dict, exiting: set, entering: set, cash_available: float
) -> None:
    """Deploy cash into a specific strategy sleeve"""
    if cash_available <= 0:
        return

    if entering:
        alloc_each = cash_available / len(entering)
        for t in entering:
            positions[t] = {"cost_basis": alloc_each, "current_value": alloc_each}
    else:
        continuing_total = sum(p["current_value"] for p in positions.values())
        if continuing_total > 0:
            for pos in positions.values():
                injected = cash_available * pos["current_value"] / continuing_total
                pos["cost_basis"] += injected
                pos["current_value"] += injected


def apply_returns(
    positions: dict, ret_row: pd.Series, max_ret: float = 1.0, min_ret: float = -0.90
) -> None:
    for t, pos in positions.items():
        r = ret_row.get(t, pd.NA)
        if pd.isna(r):
            r = 0.0
        r = min(max(r, min_ret), max_ret)
        pos["current_value"] *= 1.0 + r


def run_combined_backtest(
    adj_close: pd.DataFrame,
    close_px: pd.DataFrame,
    fin_vol: pd.DataFrame,
) -> dict:
    monthly_px = adj_close.resample("ME").last()
    monthly_ret = monthly_px.pct_change()
    monthly_raw_close = close_px.resample("ME").last()
    monthly_adtv = fin_vol.resample("ME").mean()

    sig_mom = momentum_signal(monthly_ret, LOOKBACK_MONTHS, SKIP_MONTHS)
    sig_vol = low_vol_signal(monthly_ret.shift(1), LOOKBACK_MONTHS)

    start_idx = LOOKBACK_MONTHS + SKIP_MONTHS + 1

    # We maintain separate position ledgers for each strategy sleeve
    # but they share a single tax pool
    pt_mom_pos, pt_vol_pos = {}, {}
    at_mom_pos, at_vol_pos = {}, {}

    loss_carryforward = 0.0

    pretax_values, aftertax_values = [], []
    tax_paid_list, loss_cf_list, turnover_list, dates = [], [], [], []

    prev_sel_mom, prev_sel_vol = set(), set()
    initialized = False

    for i in range(start_idx, len(monthly_ret)):
        date = monthly_ret.index[i]

        adtv_row = monthly_adtv.iloc[i - 1]
        raw_close_row = monthly_raw_close.iloc[i - 1]
        valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)

        # 1. Selection: Momentum
        valid_mom = sig_mom.iloc[i - 1][valid_mask].dropna()
        if len(valid_mom) >= 5:
            n_sel_mom = max(1, int(len(valid_mom) * TOP_DECILE))
            new_sel_mom = set(valid_mom.nlargest(n_sel_mom).index.tolist())
        else:
            new_sel_mom = prev_sel_mom

        # 2. Selection: Low Vol
        valid_vol = sig_vol.iloc[i][valid_mask].dropna()
        if len(valid_vol) >= 5:
            n_sel_vol = max(1, int(len(valid_vol) * TOP_DECILE))
            new_sel_vol = set(valid_vol.nlargest(n_sel_vol).index.tolist())
        else:
            new_sel_vol = prev_sel_vol

        ret_row = monthly_ret.iloc[i]

        # ── First month initialization ───
        if not initialized and len(valid_mom) >= 5 and len(valid_vol) >= 5:
            # Split capital 50/50 between the two strategies
            sleeve_cap = INITIAL_CAPITAL / 2.0

            alloc_mom = sleeve_cap / len(new_sel_mom)
            for t in new_sel_mom:
                pt_mom_pos[t] = {"cost_basis": alloc_mom, "current_value": alloc_mom}
                at_mom_pos[t] = {"cost_basis": alloc_mom, "current_value": alloc_mom}

            alloc_vol = sleeve_cap / len(new_sel_vol)
            for t in new_sel_vol:
                pt_vol_pos[t] = {"cost_basis": alloc_vol, "current_value": alloc_vol}
                at_vol_pos[t] = {"cost_basis": alloc_vol, "current_value": alloc_vol}

            prev_sel_mom = new_sel_mom
            prev_sel_vol = new_sel_vol
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
        apply_returns(pt_mom_pos, ret_row)
        apply_returns(pt_vol_pos, ret_row)
        apply_returns(at_mom_pos, ret_row)
        apply_returns(at_vol_pos, ret_row)

        # ── Step 2: Classify positions ─────────────────────────────────
        exit_mom = prev_sel_mom - new_sel_mom
        enter_mom = new_sel_mom - prev_sel_mom

        exit_vol = prev_sel_vol - new_sel_vol
        enter_vol = new_sel_vol - prev_sel_vol

        # Calculate combined turnover
        total_exits = len(exit_mom) + len(exit_vol)
        total_enters = len(enter_mom) + len(enter_vol)
        total_universe = len(prev_sel_mom | new_sel_mom) + len(
            prev_sel_vol | new_sel_vol
        )
        turnover_list.append((total_exits + total_enters) / max(total_universe, 1))

        # ── Step 3: Compute unified tax for after-tax ledgers ───────────
        gross_gain = 0.0
        gross_loss = 0.0

        # Collect PnL from both sleeves
        for ledger, exits in [(at_mom_pos, exit_mom), (at_vol_pos, exit_vol)]:
            for t in exits:
                if t not in ledger:
                    continue
                pnl = ledger[t]["current_value"] - ledger[t]["cost_basis"]
                if pnl > 0:
                    gross_gain += pnl
                else:
                    gross_loss += abs(pnl)

        net_pnl = gross_gain - gross_loss

        if net_pnl > 0:
            net_after_cf = net_pnl - loss_carryforward
            if net_after_cf > 0:
                tax = TAX_RATE * net_after_cf
                loss_carryforward = 0.0
            else:
                tax = 0.0
                loss_carryforward = abs(net_after_cf)
        else:
            tax = 0.0
            loss_carryforward += abs(net_pnl)

        # ── Step 4: Extract Cash & Distribute Tax ───────────────────────
        pt_cash_mom = sum(
            pt_mom_pos.pop(t)["current_value"] for t in exit_mom if t in pt_mom_pos
        )
        pt_cash_vol = sum(
            pt_vol_pos.pop(t)["current_value"] for t in exit_vol if t in pt_vol_pos
        )

        at_cash_mom = sum(
            at_mom_pos.pop(t)["current_value"] for t in exit_mom if t in at_mom_pos
        )
        at_cash_vol = sum(
            at_vol_pos.pop(t)["current_value"] for t in exit_vol if t in at_vol_pos
        )

        # Pro-rata tax distribution to the sleeves based on freed cash
        # This keeps the 50/50 sleeve balancing relatively intact over time
        total_at_cash = at_cash_mom + at_cash_vol
        if total_at_cash > 0:
            tax_mom = tax * (at_cash_mom / total_at_cash)
            tax_vol = tax * (at_cash_vol / total_at_cash)
        else:
            tax_mom = tax_vol = 0.0

        at_cash_mom -= tax_mom
        at_cash_vol -= tax_vol

        # ── Step 5: Rebalance Sleeves ──────────────────────────────────
        _rebalance_sleeve(pt_mom_pos, exit_mom, enter_mom, pt_cash_mom)
        _rebalance_sleeve(pt_vol_pos, exit_vol, enter_vol, pt_cash_vol)

        _rebalance_sleeve(at_mom_pos, exit_mom, enter_mom, at_cash_mom)
        _rebalance_sleeve(at_vol_pos, exit_vol, enter_vol, at_cash_vol)

        # ── Step 6: Record state ───────────────────────────────────────
        pt_total = sum(p["current_value"] for p in pt_mom_pos.values()) + sum(
            p["current_value"] for p in pt_vol_pos.values()
        )
        at_total = sum(p["current_value"] for p in at_mom_pos.values()) + sum(
            p["current_value"] for p in at_vol_pos.values()
        )

        pretax_values.append(pt_total)
        aftertax_values.append(at_total)
        tax_paid_list.append(tax)
        loss_cf_list.append(loss_carryforward)
        dates.append(date)

        prev_sel_mom = new_sel_mom
        prev_sel_vol = new_sel_vol

    idx = pd.DatetimeIndex(dates)

    return {
        "pretax_values": pd.Series(
            pretax_values, index=idx, name="50/50 Mix (Pre-Tax)"
        ),
        "aftertax_values": pd.Series(
            aftertax_values, index=idx, name="50/50 Mix (After-Tax)"
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
    print("  B3 NATIVE COMBINED STRATEGY BACKTEST (15% CGT)")
    print("  50% Momentum + 50% Low Volatility")
    print("=" * 70)

    # 1. Load native B3 adjusted data
    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    # 2. Download benchmark
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    ibov_monthly = ibov_px.resample("ME").last().pct_change().dropna()
    ibov_monthly.name = "IBOV"

    # 3. Run Strategy
    print("\n🚀 Running multi-sleeve dynamic backtest with unified tax engine...")
    result = run_combined_backtest(adj_close, close_px, fin_vol)

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

    m_pretax = build_metrics(pretax_ret, "50/50 Mix Pre-Tax")
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT")
    m_ibov = build_metrics(ibov_ret, "IBOV")

    display_metrics_table([m_pretax, m_aftertax, m_ibov])

    title = (
        f"Combined 50/50 Strategy (B3 Native)  ·  Mom + Low Vol\n"
        f"Top 10% Selected  ·  R$ {MIN_ADTV / 1_000_000:.0f}M+ ADTV  ·  15% CGT with Shared Loss Carryforward\n"
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
        out_path="combined_strategy_backtest.png",
    )


if __name__ == "__main__":
    main()

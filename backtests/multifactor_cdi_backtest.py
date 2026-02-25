"""
B3 50/50 Multi-Factor + CDI Strategy Backtest
=======================================================================
Strategy: Runs a 50/50 portfolio where half the capital is allocated to 
the Multi-Factor strategy (Mom + Low Vol) and the other half is allocated 
to the risk-free rate (CDI). The portfolio is rebalanced to exactly 50/50 
every month. 

As requested, the CDI returns are subject to the same 15% CGT + loss 
carryforward logic as the equities to simulate a single unified tax pool.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_b3_data, download_benchmark, download_cdi_monthly
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.portfolio import apply_returns, compute_tax

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

def run_backtest(adj_close, close_px, fin_vol, cdi_ret):
    monthly_px = adj_close.resample("ME").last()
    monthly_ret = monthly_px.pct_change()
    
    monthly_raw_close = close_px.resample("ME").last()
    monthly_adtv = fin_vol.resample("ME").mean()
    
    # ── Signal Generation (Multi-Factor) ──
    log_ret = np.log1p(monthly_ret)
    mom_signal = log_ret.shift(1).rolling(12).sum()
    mom_glitch = ((monthly_ret > 1.0) | (monthly_ret < -0.90)).shift(1).rolling(12).max()
    mom_signal[mom_glitch == 1] = np.nan
    
    vol_signal = -monthly_ret.shift(1).rolling(12).std()
    vol_glitch = ((monthly_ret > 1.0) | (monthly_ret < -0.90)).shift(1).rolling(12).max()
    vol_signal[vol_glitch == 1] = np.nan
    
    mom_rank = mom_signal.rank(axis=1, pct=True)
    vol_rank = vol_signal.rank(axis=1, pct=True)
    
    composite_signal = (mom_rank * 0.5) + (vol_rank * 0.5)
    
    start_idx = 14

    # We maintain positions as dictionaries mapping ticker to details.
    # We will use the string 'CDI_ASSET' to represent the CDI position.
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
            
        # The target portfolio state is the chosen equities PLUS the CDI asset
        new_target_portfolio = new_sel_eq | {"CDI_ASSET"}

        ret_row = monthly_ret.iloc[i].copy()
        
        # Add the CDI return for this month to the return row so apply_returns works natively
        current_cdi = cdi_ret.get(date, 0.0)
        if pd.isna(current_cdi): current_cdi = 0.0
        ret_row["CDI_ASSET"] = current_cdi

        # ── First month initialization ───
        if not initialized and len(valid_eq) >= 5:
            eq_alloc = (INITIAL_CAPITAL / 2.0) / len(new_sel_eq)
            cdi_alloc = INITIAL_CAPITAL / 2.0
            
            for t in new_sel_eq:
                pt_pos[t] = {"cost_basis": eq_alloc, "current_value": eq_alloc}
                at_pos[t] = {"cost_basis": eq_alloc, "current_value": eq_alloc}
                
            pt_pos["CDI_ASSET"] = {"cost_basis": cdi_alloc, "current_value": cdi_alloc}
            at_pos["CDI_ASSET"] = {"cost_basis": cdi_alloc, "current_value": cdi_alloc}
                
            prev_sel_eq = new_sel_eq
            initialized = True

            pretax_values.append(INITIAL_CAPITAL)
            aftertax_values.append(INITIAL_CAPITAL)
            tax_paid_list.append(0.0)
            loss_cf_list.append(0.0)
            turnover_list.append(1.0)
            dates.append(date)
            continue
            
        if not initialized: continue

        # ── Step 1: Apply Returns ────
        apply_returns(pt_pos, ret_row)
        apply_returns(at_pos, ret_row)

        # ── Step 2: Liquidate Entire Portfolio to Rebalance 50/50 ──
        # Because we want exactly 50/50 between the entire equity sleeve and the CDI sleeve,
        # we treat this as a full portfolio rebalance (selling everything that deviates and rebuying).
        # To avoid penalizing turnover visually, we'll calculate real turnover below.
        
        # But for tax, any position leaving the portfolio or being resized down is technically a tax event.
        # To keep it identical to the requested framework mechanics, we will fully "close" the portfolio 
        # to a cash pool, pay tax on all gains, and redeploy 50/50.
        
        # Since standard momentum strategy rebalances fully into equal weights anyway among survivors,
        # fully liquidating makes the math perfect.
        
        pt_total = sum(p["current_value"] for p in pt_pos.values())
        at_total = sum(p["current_value"] for p in at_pos.values())
        
        # Calculate Turnover (What actually changed)
        # Exiting equities + Entering equities + shift between Equity/CDI ratio
        exit_eq = prev_sel_eq - new_sel_eq
        enter_eq = new_sel_eq - prev_sel_eq
        turnover_list.append(len(exit_eq | enter_eq) / max(len(prev_sel_eq | new_sel_eq), 1))

        # ── Step 3: Compute unified tax for after-tax ledgers ───────────
        tax, loss_carryforward = compute_tax(set(at_pos.keys()), at_pos, loss_carryforward, TAX_RATE)
        
        # Liquidate
        pt_cash = pt_total
        at_cash = at_total - tax
        
        pt_pos.clear()
        at_pos.clear()

        # ── Step 4: Rebalance 50/50 ──────────────────────────────────
        pt_eq_alloc = (pt_cash / 2.0) / len(new_sel_eq)
        pt_cdi_alloc = pt_cash / 2.0
        
        at_eq_alloc = (at_cash / 2.0) / len(new_sel_eq)
        at_cdi_alloc = at_cash / 2.0
        
        for t in new_sel_eq:
            pt_pos[t] = {"cost_basis": pt_eq_alloc, "current_value": pt_eq_alloc}
            at_pos[t] = {"cost_basis": at_eq_alloc, "current_value": at_eq_alloc}
            
        pt_pos["CDI_ASSET"] = {"cost_basis": pt_cdi_alloc, "current_value": pt_cdi_alloc}
        at_pos["CDI_ASSET"] = {"cost_basis": at_cdi_alloc, "current_value": at_cdi_alloc}

        # ── Step 5: Record state ───────────────────────────────────────
        pretax_values.append(pt_total)
        aftertax_values.append(at_cash)
        tax_paid_list.append(tax)
        loss_cf_list.append(loss_carryforward)
        dates.append(date)

        prev_sel_eq = new_sel_eq

    idx = pd.DatetimeIndex(dates)
    return {
        "pretax_values": pd.Series(pretax_values, index=idx, name="50% Factor / 50% CDI (Pre-Tax)"),
        "aftertax_values": pd.Series(aftertax_values, index=idx, name="50% Factor / 50% CDI (After-Tax)"),
        "tax_paid": pd.Series(tax_paid_list, index=idx, name="Tax Paid (BRL)"),
        "loss_carryforward": pd.Series(loss_cf_list, index=idx, name="Loss Carryforward (BRL)"),
        "turnover": pd.Series(turnover_list, index=idx, name="Turnover"),
    }

def main():
    print("\n" + "=" * 70)
    print("  B3 NATIVE MULTI-FACTOR + CDI BACKTEST (15% CGT)")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)
    
    cdi_ret = download_cdi_monthly(START_DATE, END_DATE)
    
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)
    ibov_monthly = ibov_px.resample("ME").last().pct_change().dropna()
    ibov_monthly.name = "IBOV"

    print("\n🚀 Running dynamic backtest with tax engine...")
    result = run_backtest(adj_close, close_px, fin_vol, cdi_ret)

    common = result["pretax_values"].index.intersection(ibov_monthly.index)
    pretax_val = result["pretax_values"].loc[common]
    aftertax_val = result["aftertax_values"].loc[common]
    ibov_ret = ibov_monthly.loc[common]
    
    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = result["tax_paid"].sum()

    m_pretax = build_metrics(pretax_ret, "Factor+CDI Pre-Tax")
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT")
    m_ibov = build_metrics(ibov_ret, "IBOV")

    display_metrics_table([m_pretax, m_aftertax, m_ibov])

    plot_tax_backtest(
        title=f"50% Multi-Factor + 50% CDI Risk-Free\nMonthly Rebalanced  ·  15% CGT with Shared Loss Carryforward\n{START_DATE[:4]}–{END_DATE[:4]}",
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov],
        total_tax_brl=total_tax,
        out_path="multifactor_cdi_backtest.png"
    )

if __name__ == "__main__":
    main()

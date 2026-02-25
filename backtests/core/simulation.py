"""
Generic backtest simulation engine.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _apply_returns(
    positions: dict, ret_row: pd.Series, max_ret: float = 1.0, min_ret: float = -0.90
) -> None:
    """Apply returns to all open positions with clipping for safety against artifacts."""
    for t, pos in positions.items():
        if t not in ret_row.index:
            r = 0.0
        else:
            try:
                r = float(ret_row.loc[t])
                if math.isnan(r):
                    r = 0.0
            except (ValueError, TypeError):
                r = 0.0

        r = min(max(r, min_ret), max_ret)
        pos["current_value"] *= 1.0 + r


def _compute_tax(
    sales: Dict[str, float],
    positions: dict,
    loss_carryforward: float,
    tax_rate: float,
    ignore_short_losses: bool = True,
) -> Tuple[float, float]:
    """
    Compute Brazilian capital gains tax (with loss carryforward) on partial or full sales.

    Args:
        sales: Dict of {ticker: amount_sold_in_cash}
        positions: The current ledger (must contain cost_basis and current_value)
    Returns:
        (tax_paid, new_loss_carryforward)
    """
    gross_gain = 0.0
    gross_loss = 0.0

    for t, amount_sold in sales.items():
        if t not in positions or amount_sold <= 0:
            continue

        pos = positions[t]

        # Calculate what percentage of the position is being sold
        if pos["current_value"] <= 0:
            continue

        fraction_sold = min(1.0, amount_sold / pos["current_value"])

        # Calculate the proportional PnL for this sale
        cost_of_sold_portion = pos["cost_basis"] * fraction_sold
        pnl = amount_sold - cost_of_sold_portion

        # Receita Federal does not allow deducting margin interest/short losses from capital gains
        if ignore_short_losses and cost_of_sold_portion < 0 and pnl < 0:
            pass  # ignore
        else:
            if pnl > 0:
                gross_gain += pnl
            else:
                gross_loss += abs(pnl)

    net_pnl = gross_gain - gross_loss

    if net_pnl > 0:
        net_after_cf = net_pnl - loss_carryforward
        if net_after_cf > 0:
            tax = tax_rate * net_after_cf
            loss_carryforward = 0.0
        else:
            tax = 0.0
            loss_carryforward = abs(net_after_cf)
    else:
        tax = 0.0
        loss_carryforward += abs(net_pnl)

    return tax, loss_carryforward


def _execute_rebalance(
    positions: dict,
    target_weights: pd.Series,
    portfolio_nav: float,
    slippage: float,
    tax_rate: float,
    loss_carryforward: float,
) -> Tuple[float, float, float, float]:
    """
    Execute a strict rebalance to match target weights.
    Returns (tax_paid, new_loss_carryforward, cash_drag, turnover)
    """
    # 1. Identify current weights and determine buys/sells
    sells = {}  # {ticker: cash_to_raise}
    buys = {}  # {ticker: cash_to_deploy}

    target_tickers = set(target_weights[target_weights != 0].index)
    current_tickers = set(positions.keys())
    all_tickers = target_tickers | current_tickers

    turnover_cash = 0.0

    for t in all_tickers:
        current_val = positions.get(t, {}).get("current_value", 0.0)
        target_w = target_weights.get(t, 0.0)
        if pd.isna(target_w):
            target_w = 0.0

        target_val = portfolio_nav * target_w

        diff = target_val - current_val

        # If we need to sell (or short more)
        if diff < -0.01:
            sell_amount = abs(diff)
            sells[t] = sell_amount
            turnover_cash += sell_amount

        # If we need to buy (or cover short)
        elif diff > 0.01:
            buy_amount = diff
            buys[t] = buy_amount
            turnover_cash += buy_amount

    # 2. Calculate Tax on Sales
    tax_paid, loss_carryforward = _compute_tax(
        sells, positions, loss_carryforward, tax_rate
    )

    # We must pay tax out of the cash we raised. If there isn't enough cash,
    # we simulate an automatic drag on the portfolio NAV.
    # We also deduct slippage from all trades.
    total_slippage = turnover_cash * slippage
    cash_drag = tax_paid + total_slippage

    # 3. Update the Ledger
    for t, sell_amount in sells.items():
        pos = positions[t]
        fraction_sold = (
            min(1.0, sell_amount / pos["current_value"])
            if pos["current_value"] > 0
            else 1.0
        )

        if fraction_sold >= 0.999:
            del positions[t]
        else:
            pos["current_value"] -= sell_amount
            pos["cost_basis"] -= pos["cost_basis"] * fraction_sold

    # Deploy buys (after reducing them proportionately by cash_drag if we are fully invested)
    # If the portfolio is not 100% invested, this cash drag is just a reduction in NAV.
    # To keep it mathematically simple: we scale all buys down slightly so we don't over-leverage to pay taxes.
    total_buy_cash = sum(buys.values())
    buy_adjustment_factor = 1.0
    if total_buy_cash > 0 and cash_drag > 0:
        if cash_drag < total_buy_cash:
            buy_adjustment_factor = (total_buy_cash - cash_drag) / total_buy_cash
        else:
            buy_adjustment_factor = 0.0  # Taxes wiped out all buy liquidity

    for t, buy_amount in buys.items():
        actual_deploy = buy_amount * buy_adjustment_factor
        if t not in positions:
            positions[t] = {"cost_basis": actual_deploy, "current_value": actual_deploy}
        else:
            positions[t]["cost_basis"] += actual_deploy
            positions[t]["current_value"] += actual_deploy

    turnover_pct = turnover_cash / max(portfolio_nav, 1.0)

    return tax_paid, loss_carryforward, cash_drag, turnover_pct


def run_simulation(
    returns_matrix: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000,
    tax_rate: float = 0.15,
    slippage: float = 0.0,
    name: str = "Strategy",
) -> dict:
    """
    Run a generic, institutional-grade portfolio simulation.

    Args:
        returns_matrix: DataFrame of asset returns (rows=dates, cols=tickers)
        target_weights: DataFrame of target weights (rows=dates, cols=tickers).
                       Rows must sum to 1.0 (or >1.0 if using negative leverage).
        initial_capital: Starting BRL
        tax_rate: Capital gains tax rate applied to net realized gains
        slippage: Transaction cost applied per trade (e.g., 0.001 for 0.1%)

    Returns:
        Dictionary containing pre-tax and after-tax equity curves, tax ledgers, etc.
    """
    # Initialize the ledgers
    pt_pos = {}  # Pre-tax ledger
    at_pos = {}  # After-tax ledger

    loss_carryforward = 0.0

    pretax_values, aftertax_values = [], []
    tax_paid_list, loss_cf_list, turnover_list, dates = [], [], [], []

    initialized = False

    for i in range(len(returns_matrix)):
        date = returns_matrix.index[i]

        # 1. Look up the Target Weights for THIS period
        # Target weights should have been calculated using data from i-1 (no lookahead)
        t_weights = target_weights.iloc[i].fillna(0.0)

        # If target weights sum to 0, it means we sit in cash (or the strategy hasn't started yet)
        if t_weights.abs().sum() == 0:
            if not initialized:
                continue

        ret_row = returns_matrix.iloc[i]

        # ── First month initialization ───
        if not initialized:
            for t, w in t_weights.items():
                if w != 0:
                    alloc = initial_capital * w
                    pt_pos[t] = {"cost_basis": alloc, "current_value": alloc}
                    at_pos[t] = {"cost_basis": alloc, "current_value": alloc}

            initialized = True

            pretax_values.append(initial_capital)
            aftertax_values.append(initial_capital)
            tax_paid_list.append(0.0)
            loss_cf_list.append(0.0)
            turnover_list.append(1.0)
            dates.append(date)
            continue

        # ── Step 1: Apply Returns ────
        _apply_returns(pt_pos, ret_row)
        _apply_returns(at_pos, ret_row)

        pt_nav = sum(p["current_value"] for p in pt_pos.values())
        at_nav = sum(p["current_value"] for p in at_pos.values())

        if pt_nav <= 0 or at_nav <= 0:
            print(f"\n💀 MARGIN CALL! Portfolio went bankrupt on {date.date()}")
            break

        # ── Step 2: Strict Rebalance (Pre-Tax Ledger: Zero Tax) ──
        _, _, pt_drag, _ = _execute_rebalance(
            pt_pos,
            t_weights,
            pt_nav,
            slippage=slippage,
            tax_rate=0.0,
            loss_carryforward=0.0,
        )

        # ── Step 3: Strict Rebalance (After-Tax Ledger: Real Tax) ──
        tax_paid, loss_carryforward, at_drag, turnover = _execute_rebalance(
            at_pos,
            t_weights,
            at_nav,
            slippage=slippage,
            tax_rate=tax_rate,
            loss_carryforward=loss_carryforward,
        )

        pt_final_nav = sum(p["current_value"] for p in pt_pos.values())
        at_final_nav = sum(p["current_value"] for p in at_pos.values())

        # ── Step 4: Record state ───────────────────────────────────────
        pretax_values.append(pt_final_nav)
        aftertax_values.append(at_final_nav)
        tax_paid_list.append(tax_paid)
        loss_cf_list.append(loss_carryforward)
        turnover_list.append(turnover)
        dates.append(date)

    idx = pd.DatetimeIndex(dates)
    return {
        "pretax_values": pd.Series(pretax_values, index=idx, name=f"{name} (Pre-Tax)"),
        "aftertax_values": pd.Series(
            aftertax_values, index=idx, name=f"{name} (After-Tax)"
        ),
        "tax_paid": pd.Series(tax_paid_list, index=idx, name="Tax Paid (BRL)"),
        "loss_carryforward": pd.Series(
            loss_cf_list, index=idx, name="Loss Carryforward (BRL)"
        ),
        "turnover": pd.Series(turnover_list, index=idx, name="Turnover"),
    }

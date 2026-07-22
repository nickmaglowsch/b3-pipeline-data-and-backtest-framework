"""
Generic backtest simulation engine.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def _apply_returns(
    positions: dict, ret_row: pd.Series, max_ret: float = 5.0, min_ret: float = -0.90
) -> None:
    """Apply returns to all open positions with clipping for safety against artifacts.

    max_ret is a last-resort guard against data artifacts (e.g. unadjusted
    reverse splits showing as +N00%); genuine multi-hundred-percent periods
    below +500% pass through uncapped. Glitch filtering proper belongs
    upstream (has_glitch / split detection).
    """
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

        # If it is a short position (negative current value), apply a forced Stop Loss.
        # If the stock goes up more than 50% in a single period, the broker forcefully
        # closes the short position. We simulate this by hard-capping the short loss.
        if pos["current_value"] < 0 and r > 0.50:
            r = 0.50

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
        if pos["current_value"] == 0:
            continue

        fraction_sold = min(1.0, amount_sold / abs(pos["current_value"]))

        # Calculate the proportional PnL for this sale
        # If short, cost_basis is negative. Covering a short means buying back the shares.
        # Say cost basis is -100. Current value is -80 (it dropped, we made money).
        # We sell/cover all 80. amount_sold = 80.
        # Pnl = cost_basis - current_value = (-100) - (-80) = -20
        # Actually standard long: amount_sold=80, cost=100 -> PnL = 80-100 = -20.
        # So PnL is ALWAYS the difference between extracted value and cost basis.

        cost_of_sold_portion = pos["cost_basis"] * fraction_sold

        if pos["current_value"] > 0:
            # Long position
            pnl = amount_sold - cost_of_sold_portion
        else:
            # Short position. We spent 'amount_sold' cash to cover.
            # cost_basis is negative (the cash we originally received).
            # So PnL = abs(cost_basis) - amount_sold (cash spent)
            pnl = abs(cost_of_sold_portion) - amount_sold

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
    monthly_sales_exemption: float = 0.0,
    defer_tax: bool = False,
    pending_tax_payment: float = 0.0,
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

        # If diff is negative, we want less exposure than we currently have.
        # This means either SELLING a long position, or SHORTING MORE (increasing a negative position).
        if diff < -0.01:
            sell_amount = abs(diff)
            # If we are initiating a new short, it goes straight to buys/deploy logic
            # (we don't "sell" an asset we don't own to calculate taxes, we just allocate negative capital)
            if t not in positions and target_w < 0:
                buys[t] = diff  # This will be negative capital deployment
                turnover_cash += sell_amount
            else:
                sells[t] = sell_amount
                turnover_cash += sell_amount

        # If diff is positive, we want more exposure than we currently have.
        # This means either BUYING more of a long position, or COVERING a short position.
        elif diff > 0.01:
            buy_amount = diff

            # If we are covering a short back towards 0, that is technically a "sale" (closing an active position)
            # which needs to be run through the tax engine!
            if t in positions and positions[t]["current_value"] < 0:
                sells[t] = buy_amount  # To trigger tax on the covered portion
                turnover_cash += buy_amount
            else:
                buys[t] = buy_amount
                turnover_cash += buy_amount

    # 2. Calculate Tax on Sales
    # If total sell proceeds <= exemption threshold, skip tax entirely.
    # Under Brazilian law, exempt months don't consume loss carryforward.
    exempt = (
        monthly_sales_exemption > 0
        and sum(sells.values()) <= monthly_sales_exemption
    )

    if exempt:
        tax_paid = 0.0
    else:
        tax_paid, loss_carryforward = _compute_tax(
            sells, positions, loss_carryforward, tax_rate
        )

    # We must pay tax out of the cash we raised. If there isn't enough cash,
    # we simulate an automatic drag on the portfolio NAV.
    # We also deduct slippage from all trades.
    total_slippage = turnover_cash * slippage
    if defer_tax:
        # Don't pay current tax now; pay last month's deferred tax instead
        cash_drag = total_slippage + pending_tax_payment
    else:
        cash_drag = tax_paid + total_slippage

    # 3. Update the Ledger
    for t, sell_amount in sells.items():
        pos = positions[t]

        # Determine the fraction sold based on whether it is a long or short position
        if pos["current_value"] > 0:
            fraction_sold = min(1.0, sell_amount / pos["current_value"])
        elif pos["current_value"] < 0:
            # We are covering a short. "sell_amount" is positive here representing the cash required to cover.
            fraction_sold = min(1.0, sell_amount / abs(pos["current_value"]))
        else:
            fraction_sold = 1.0

        if fraction_sold >= 0.999:
            del positions[t]
        else:
            if pos["current_value"] > 0:
                pos["current_value"] -= sell_amount
            else:
                # Covering a short moves current value closer to 0
                pos["current_value"] += sell_amount

            pos["cost_basis"] -= pos["cost_basis"] * fraction_sold

    # Deploy buys (after reducing them proportionately by cash_drag if we are fully invested)
    # If the portfolio is not 100% invested, this cash drag is just a reduction in NAV.
    # To keep it mathematically simple: we scale all buys down slightly so we don't over-leverage to pay taxes.
    total_buy_cash = sum(v for v in buys.values() if v > 0)
    buy_adjustment_factor = 1.0
    residual_drag = 0.0
    if cash_drag > 0:
        if cash_drag < total_buy_cash:
            buy_adjustment_factor = (total_buy_cash - cash_drag) / total_buy_cash
        else:
            buy_adjustment_factor = 0.0  # Taxes wiped out all buy liquidity
            residual_drag = cash_drag - total_buy_cash

    for t, buy_amount in buys.items():
        actual_deploy = buy_amount * buy_adjustment_factor
        if t not in positions:
            positions[t] = {"cost_basis": actual_deploy, "current_value": actual_deploy}
        else:
            positions[t]["cost_basis"] += actual_deploy
            positions[t]["current_value"] += actual_deploy

    # Any drag not absorbed by scaling down buys must still be paid — deduct it
    # from NAV via a proportional reduction across the final positions
    # (previously the excess silently vanished).
    if residual_drag > 0:
        nav_after = sum(p["current_value"] for p in positions.values())
        if nav_after > 0:
            scale = max(0.0, 1.0 - residual_drag / nav_after)
            for p in positions.values():
                p["current_value"] *= scale

    turnover_pct = turnover_cash / max(portfolio_nav, 1.0)

    return tax_paid, loss_carryforward, cash_drag, turnover_pct


def run_simulation(
    returns_matrix: pd.DataFrame,
    target_weights: pd.DataFrame,
    initial_capital: float = 100_000,
    tax_rate: float = 0.15,
    slippage: float = 0.0,
    name: str = "Strategy",
    monthly_sales_exemption: float = 0.0,
    defer_tax: bool = False,
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
    pending_tax = 0.0

    pretax_values, aftertax_values = [], []
    tax_paid_list, loss_cf_list, turnover_list, dates = [], [], [], []

    initialized = False

    # Contract guard: NAV is defined as the sum of position values and this engine
    # does NOT hold uninvested cash. So any live rebalance row whose target weights
    # sum to < 1 has that gap silently dropped from NAV — a fake drawdown that
    # compounds every rebalance. Warn loudly rather than fail silently; strategies
    # must park the residual in CDI_ASSET. (Rows that sum to 0 are held; rows > 1
    # are the separate leverage case, not flagged here.)
    _row_sums = target_weights.fillna(0.0).sum(axis=1)
    _under = (_row_sums > 1e-9) & (_row_sums < 1.0 - 1e-4)
    if bool(_under.any()):
        _off = _under[_under].index
        print(f"\n⚠️  {name}: {len(_off)} live rebalance row(s) with target weights "
              f"summing < 1 (first {_off[0].date()}, sum={_row_sums.loc[_off[0]]:.3f}); "
              f"run_simulation drops that uninvested weight from NAV (fake drawdown). "
              f"Park the residual in CDI_ASSET.")

    for i in range(len(returns_matrix)):
        date = returns_matrix.index[i]

        # 1. Look up the Target Weights for THIS period
        # Target weights should have been calculated using data from i-1 (no lookahead)
        t_weights = target_weights.iloc[i].fillna(0.0)

        # Before inception, an all-zero row means the strategy hasn't started yet
        if t_weights.abs().sum() == 0 and not initialized:
            continue

        ret_row = returns_matrix.iloc[i]

        # ── First month initialization ───
        if not initialized:
            for t, w in t_weights.items():
                if w != 0:
                    # Charge slippage on the initial deployment (turnover = 1.0)
                    alloc = initial_capital * w * (1.0 - slippage)
                    pt_pos[t] = {"cost_basis": alloc, "current_value": alloc}
                    at_pos[t] = {"cost_basis": alloc, "current_value": alloc}

            initialized = True

            nav0 = sum(p["current_value"] for p in pt_pos.values())
            pretax_values.append(nav0)
            aftertax_values.append(nav0)
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

        # After inception, an all-zero target row means "no valid targets this
        # period" — hold current positions. Rebalancing to zero would liquidate
        # everything, and since sale proceeds aren't tracked as cash
        # (NAV = sum of position values), NAV would collapse to 0 and fake a
        # margin call on the next step.
        if t_weights.abs().sum() == 0:
            pretax_values.append(pt_nav)
            aftertax_values.append(at_nav)
            tax_paid_list.append(0.0)
            loss_cf_list.append(loss_carryforward)
            turnover_list.append(0.0)
            dates.append(date)
            continue

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
            monthly_sales_exemption=monthly_sales_exemption,
            defer_tax=defer_tax,
            pending_tax_payment=pending_tax if defer_tax else 0.0,
        )

        # Update pending tax for next period
        if defer_tax:
            pending_tax = tax_paid

        pt_final_nav = sum(p["current_value"] for p in pt_pos.values())
        at_final_nav = sum(p["current_value"] for p in at_pos.values())

        # ── Step 4: Record state ───────────────────────────────────────
        pretax_values.append(pt_final_nav)
        aftertax_values.append(at_final_nav)
        tax_paid_list.append(tax_paid)
        loss_cf_list.append(loss_carryforward)
        turnover_list.append(turnover)
        dates.append(date)

    # Deduct any remaining deferred tax from final NAV
    if defer_tax and pending_tax > 0:
        aftertax_values[-1] -= pending_tax

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

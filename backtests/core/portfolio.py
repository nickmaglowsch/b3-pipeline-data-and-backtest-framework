"""
Portfolio management and rebalancing mechanics.
"""

import pandas as pd


def rebalance_positions(
    positions: dict, exiting: set, entering: set, ret_row: pd.Series, tax: float
) -> None:
    """
    Shared rebalancing logic used by both ledgers.

    1. Remove exiting positions and collect their liquidation cash.
    2. Deduct tax from available cash.
    3. Deploy cash into new entrants. If there are no new entrants,
       distribute freed cash pro-rata to surviving continuing positions
       so no capital leaks out.
    """
    exit_cash = 0.0
    for t in exiting:
        if t in positions:
            exit_cash += positions[t]["current_value"]
            del positions[t]

    cash_available = exit_cash - tax

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
    """Apply monthly returns to all open positions with optional clipping for safety."""
    for t, pos in positions.items():
        r = ret_row.get(t, pd.NA)
        if pd.isna(r):
            r = 0.0

        r = min(max(r, min_ret), max_ret)
        pos["current_value"] *= 1.0 + r


def compute_tax(
    exiting: set, positions: dict, loss_carryforward: float, tax_rate: float
) -> tuple[float, float]:
    """
    Compute Brazilian capital gains tax (with loss carryforward) on exiting positions.
    Returns (tax_paid, new_loss_carryforward).
    """
    gross_gain = 0.0
    gross_loss = 0.0

    for t in exiting:
        if t not in positions:
            continue
        cv = positions[t]["current_value"]
        cb = positions[t]["cost_basis"]
        pnl = cv - cb
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

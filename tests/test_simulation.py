"""Tests for the generic simulation engine (backtests/core/simulation.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests.core.metrics import value_to_ret
from backtests.core.simulation import run_simulation, _apply_returns, _execute_rebalance

IDX = pd.date_range("2020-01-31", periods=5, freq="ME")
COLS = ["A", "B"]


def _run(weights_rows, returns_rows, **kw):
    rm = pd.DataFrame(returns_rows, index=IDX[: len(returns_rows)], columns=COLS)
    tw = pd.DataFrame(weights_rows, index=rm.index, columns=COLS)
    kw.setdefault("initial_capital", 100.0)
    kw.setdefault("tax_rate", 0.0)
    return run_simulation(rm, tw, **kw)


def test_zero_weight_row_after_inception_holds_positions():
    # Month 3 emits an all-zero row (e.g. <5 valid names that period) — the
    # engine must hold current positions, not liquidate to nothing (which
    # zeroed NAV and faked a margin call).
    tw_rows = [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    rets = [[0.0, 0.0], [0.10, 0.0], [0.10, 0.0], [0.10, 0.0], [0.10, 0.0]]
    res = _run(tw_rows, rets)
    vals = res["aftertax_values"]
    assert len(vals) == 5, "must not truncate with a bogus margin call"
    assert np.allclose(vals.values, 100.0 * 1.1 ** np.arange(5))
    assert res["turnover"].iloc[2] == 0.0


def test_underinvested_row_warns():
    # A live row summing < 1 must warn (its uninvested weight is dropped from NAV).
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run([[0.5, 0.0], [0.5, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    assert "summing < 1" in buf.getvalue()


def test_fully_invested_row_does_not_warn():
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run([[0.6, 0.4], [0.6, 0.4]], [[0.0, 0.0], [0.0, 0.0]])
    assert "summing < 1" not in buf.getvalue()


def test_initial_deployment_charged_slippage():
    res = _run([[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], slippage=0.01)
    assert res["aftertax_values"].iloc[0] == pytest.approx(99.0)
    assert res["pretax_values"].iloc[0] == pytest.approx(99.0)


def test_excess_drag_deducted_from_nav():
    # Sell 10, buy 10, slippage 60% -> drag 12 exceeds buy liquidity 10.
    # Buys scale to zero and the residual 2 must come out of NAV (previously
    # it silently vanished).
    positions = {"A": {"cost_basis": 100.0, "current_value": 100.0}}
    tw = pd.Series({"A": 0.9, "B": 0.1})
    _execute_rebalance(
        positions, tw, 100.0, slippage=0.6, tax_rate=0.0, loss_carryforward=0.0
    )
    nav = sum(p["current_value"] for p in positions.values())
    assert nav == pytest.approx(90.0 - 2.0)


def test_apply_returns_allows_large_genuine_gains():
    pos = {"A": {"cost_basis": 10.0, "current_value": 10.0}}
    _apply_returns(pos, pd.Series({"A": 2.0}))  # +200% must not be clipped
    assert pos["A"]["current_value"] == pytest.approx(30.0)


# ── Periodic buy-ins (aportes) ────────────────────────────────────────────────

FLAT = [[0.0, 0.0]] * 5
HALF = [[0.5, 0.5]] * 5


def test_contribution_lands_in_nav():
    # 5 monthly rows, zero returns: 4 deposits (none on the inception row).
    res = _run(HALF, FLAT, contribution=1_000.0)
    assert res["aftertax_values"].iloc[-1] == pytest.approx(100.0 + 4 * 1_000.0)
    assert res["contributions"].tolist() == [0.0] + [1_000.0] * 4
    assert res["invested"].iloc[-1] == pytest.approx(100.0 + 4 * 1_000.0)


def test_contribution_buys_more_of_the_underweight():
    # THE headline property: A doubled to 100 while B stayed at 50 against a
    # 50/50 target, so B is farther below target and must take more of the new
    # cash. Asserted per leg — NAV and turnover totals are identical for an
    # even 50/50 split, so they cannot discriminate a broken implementation.
    pos = {"A": {"cost_basis": 50.0, "current_value": 100.0},
           "B": {"cost_basis": 50.0, "current_value": 50.0}}
    before = {t: p["current_value"] for t, p in pos.items()}
    _execute_rebalance(
        pos, pd.Series({"A": 0.5, "B": 0.5}), 150.0 + 100.0,   # NAV + buy-in
        slippage=0.0, tax_rate=0.0, loss_carryforward=0.0,
    )
    bought = {t: pos[t]["current_value"] - before[t] for t in pos}
    assert bought["B"] == pytest.approx(75.0)
    assert bought["A"] == pytest.approx(25.0)
    assert bought["B"] == pytest.approx(3 * bought["A"])   # not an even split

    # And the same through the full loop: the deposit reaches the book.
    res = run_simulation(
        pd.DataFrame([[0.0, 0.0], [1.0, 0.0]], index=IDX[:2], columns=COLS),
        pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=IDX[:2], columns=COLS),
        initial_capital=100.0, tax_rate=0.0, contribution=100.0,
    )
    assert res["aftertax_values"].iloc[-1] == pytest.approx(250.0)


def test_withdrawal_sells_and_is_taxed():
    # Negative contribution = withdrawal: it must come out of NAV and the forced
    # sells must go through the tax engine (not silently escape it).
    rets = [[0.0, 0.0], [1.0, 0.0]]        # A doubles -> selling realizes a gain
    tw = pd.DataFrame([[0.5, 0.5], [0.5, 0.5]], index=IDX[:2], columns=COLS)
    res = run_simulation(
        pd.DataFrame(rets, index=IDX[:2], columns=COLS), tw,
        initial_capital=100.0, tax_rate=0.15, contribution=-30.0,
    )
    assert res["contributions"].iloc[-1] == pytest.approx(-30.0)
    assert res["invested"].iloc[-1] == pytest.approx(70.0)
    assert res["tax_paid"].iloc[-1] > 0          # the forced sell was taxed
    # NAV 150, withdraw 30 -> 120 before tax, minus the tax on the realized gain.
    assert res["aftertax_values"].iloc[-1] == pytest.approx(120.0 - res["tax_paid"].iloc[-1])


def test_contributions_reduce_realized_tax():
    # The economic reason DCA beats lump sum after tax: new cash does the
    # rebalancing, so less has to be sold and less gain is realized.
    rets = [[0.0, 0.0]] + [[0.30, 0.0]] * 4      # A keeps running away from target
    kw = dict(tax_rate=0.15, slippage=0.001)
    lump = _run(HALF, rets, **kw)
    dca = _run(HALF, rets, contribution=50.0, **kw)
    assert dca["tax_paid"].sum() < lump["tax_paid"].sum()


def test_weekly_cadence_deposits_once_per_month():
    # W-FRI: the buy-in lands on the first weekly rebalance of each new month.
    wk = pd.date_range("2020-01-03", periods=10, freq="W-FRI")   # Jan x5, Feb x4, Mar
    res = run_simulation(
        pd.DataFrame(0.0, index=wk, columns=COLS),
        pd.DataFrame(0.5, index=wk, columns=COLS),
        initial_capital=100.0, tax_rate=0.0, contribution=1_000.0,
    )
    months = pd.DatetimeIndex(res["contributions"].index).to_period("M")
    paid = res["contributions"] > 0
    assert paid.sum() == 2                                   # Feb and Mar
    assert not months[paid].duplicated().any()               # never twice in a month
    assert res["contributions"][paid].eq(1_000.0).all()


def test_pure_dca_from_zero_capital():
    # initial_capital=0: no lump sum at all, the book is built by buy-ins alone.
    res = _run(HALF, FLAT, initial_capital=0.0, contribution=1_000.0)
    assert res["aftertax_values"].tolist() == pytest.approx([0.0, 1_000.0, 2_000.0, 3_000.0, 4_000.0])
    assert res["invested"].iloc[-1] == pytest.approx(4_000.0)


def test_no_money_at_all_is_rejected():
    with pytest.raises(ValueError, match="No money"):
        _run(HALF, FLAT, initial_capital=0.0, contribution=0.0)


def test_contribution_is_time_weighted_neutral():
    # Same weights and returns; deposits must not show up as performance.
    rets = [[0.02, 0.01], [-0.03, 0.01], [0.05, 0.01], [0.0, 0.01], [0.01, 0.01]]
    base = _run(HALF, rets)
    dca = _run(HALF, rets, contribution=1_000.0)
    r_base = value_to_ret(base["aftertax_values"])
    r_dca = value_to_ret(dca["aftertax_values"], dca["contributions"])
    assert np.allclose(r_base.values, r_dca.values, atol=1e-9)
    # ... and the naive pct_change would have lied (deposits dwarf the returns).
    assert value_to_ret(dca["aftertax_values"]).iloc[1] > 1.0


def test_contribution_accrues_by_calendar_month():
    # Quarterly grid -> each rebalance collects 3 months of buy-ins.
    q = pd.date_range("2020-03-31", periods=3, freq="QE")
    res = run_simulation(
        pd.DataFrame(0.0, index=q, columns=COLS),
        pd.DataFrame(0.5, index=q, columns=COLS),
        initial_capital=100.0, tax_rate=0.0, contribution=1_000.0,
    )
    assert res["contributions"].tolist() == [0.0, 3_000.0, 3_000.0]


def test_contribution_defers_over_hold_rows():
    # Month 3 has no valid targets (all-zero row): nothing is deposited that
    # month, and the next live rebalance collects both months.
    tw = [[0.5, 0.5], [0.5, 0.5], [0.0, 0.0], [0.5, 0.5], [0.5, 0.5]]
    res = _run(tw, FLAT, contribution=1_000.0)
    assert res["contributions"].tolist() == [0.0, 1_000.0, 0.0, 2_000.0, 1_000.0]
    assert res["aftertax_values"].iloc[-1] == pytest.approx(100.0 + 4 * 1_000.0)
